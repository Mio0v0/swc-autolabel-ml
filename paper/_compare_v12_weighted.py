#!/usr/bin/env python3
"""Evaluate v12_weighted on the same v12 test split and compare to v12 baseline.

Runs the v12_weighted pipeline (weighted Stage 2 + class-weighted GNN)
on the held-out test cells, computes node-level + per-cell metrics, and
prints a side-by-side comparison against the v12 baseline.

Saves:
    paper/results/v12_weighted_test_metrics.json
    paper/results/v12_weighted_per_branch.csv
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.features import parse_swc                             # noqa: E402
from hybrid.pipeline import run_pipeline_on_nodes                 # noqa: E402
from hybrid.confidence import (                                   # noqa: E402
    ConfidenceConfig, summarize_confidence,
)
from paper.gnn_inference import load_gnn                          # noqa: E402

DATA_DIR    = ROOT / "data" / "v12_uncurated"
SPLIT_JSON  = ROOT / "paper" / "models" / "v12_weighted" / "train_test_split.json"
W_DIR       = ROOT / "paper" / "models" / "v12_weighted"
W_S1        = W_DIR / "cell_type_classifier.pkl"
W_S2        = W_DIR / "branch_classifier.pkl"
W_GNN       = W_DIR / "gnn_apical_basal.pt"
OUT_JSON    = ROOT / "paper" / "results" / "v12_weighted_test_metrics.json"
OUT_CSV     = ROOT / "paper" / "results" / "v12_weighted_per_branch.csv"
V12_PER_BRANCH_CSV = ROOT / "paper" / "results" / "flag_calibration_curves.csv"


def _neurite_macro_f1(gt, pred):
    if not gt:
        return 0.0
    return float(f1_score(gt, pred, labels=[2, 3, 4], average="macro", zero_division=0))


def _evaluate_model(label: str, s1, s2, gnn_path, test_files):
    """Run inference on all test files and return aggregated metrics + per-branch rows."""
    print(f"\n=== Evaluating: {label} ===")
    print(f"  Stage 1: {s1.name}")
    print(f"  Stage 2: {s2.name}")
    print(f"  GNN    : {gnn_path.name}")
    gnn_state = load_gnn(gnn_path)
    print(f"  GNN loaded on {gnn_state.device}")

    per_branch_rows = []
    per_cell = []
    t0 = time.perf_counter()
    for i, (ct_gt, path) in enumerate(test_files):
        try:
            nodes = parse_swc(path)
            if not nodes:
                continue
            pr = run_pipeline_on_nodes(
                nodes, file_path=str(path),
                stage1_model=s1, stage2_model=s2,
                gnn_state=gnn_state, use_subtree_stage2=True,
            )
        except Exception as exc:
            print(f"  WARN: {path.name}: {exc}")
            continue

        gt = [n.type for n in nodes]
        pred = pr.node_labels
        confs = pr.node_confidences

        # Per-cell metrics
        nf1 = _neurite_macro_f1(gt, pred)
        n_correct = sum(1 for g, p in zip(gt, pred) if g == p)
        acc = n_correct / max(1, len(gt))
        per_cell.append({
            "file":         path.name,
            "cell_type_gt": ct_gt,
            "cell_type_pred": pr.stage1.cell_type,
            "n_nodes":      len(gt),
            "accuracy":     acc,
            "neurite_f1":   nf1,
        })

        # Per-branch rows (for confusion matrix + flag analysis)
        cfg = ConfidenceConfig.default()
        branches, _ = summarize_confidence(
            nodes, pred, confs, pr.stage1.cell_type, float(pr.stage1.confidence), cfg,
        )
        for b in branches:
            gt_labs = [gt[idx] for idx in b.node_indices]
            gt_maj = Counter(gt_labs).most_common(1)[0][0]
            n_cor = sum(1 for idx in b.node_indices if pred[idx] == gt[idx])
            per_branch_rows.append({
                "file":               path.name,
                "cell_type_gt":       ct_gt,
                "n_nodes":            b.n_nodes,
                "pred_label":         b.predicted_label,
                "gt_label":           int(gt_maj),
                "is_wrong":           int(gt_maj != b.predicted_label),
                "frac_correct_nodes": n_cor / b.n_nodes,
                "mean_conf":          b.mean_confidence,
                "min_conf":           b.min_confidence,
            })
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(test_files)}  (elapsed {(time.perf_counter()-t0)/60:.1f} min)")
    print(f"  inference elapsed: {(time.perf_counter()-t0)/60:.1f} min")
    return per_branch_rows, per_cell


def _compute_metrics(per_branch_rows, per_cell):
    n_total = len(per_branch_rows)
    n_wrong = sum(1 for r in per_branch_rows if r["is_wrong"])
    total_nodes  = sum(r["n_nodes"] for r in per_branch_rows)
    correct_nodes = sum(r["n_nodes"] * r["frac_correct_nodes"] for r in per_branch_rows)
    per_node_acc = correct_nodes / max(1, total_nodes)
    branch_acc = 1 - n_wrong / max(1, n_total)

    # Per-class F1 (branch-weighted approximation)
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for r in per_branch_rows:
        pred = int(r["pred_label"]); gt = int(r["gt_label"]); n = int(r["n_nodes"])
        for c in [1, 2, 3, 4]:
            if pred == c and gt == c: tp[c] += n
            elif pred == c and gt != c: fp[c] += n
            elif pred != c and gt == c: fn[c] += n
    per_class = {}
    for c in [1, 2, 3, 4]:
        p = tp[c] / max(1, tp[c] + fp[c])
        r = tp[c] / max(1, tp[c] + fn[c])
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        per_class[c] = {"precision": p, "recall": r, "f1": f1, "support_nodes": tp[c] + fn[c]}
    neurite_f1 = float(np.mean([per_class[c]["f1"] for c in [2, 3, 4]]))

    # Per-cell distribution (accuracy + neurite F1)
    accs = np.array([c["accuracy"] for c in per_cell])
    nf1s = np.array([c["neurite_f1"] for c in per_cell])

    return {
        "n_cells":               len(per_cell),
        "n_nodes":               int(total_nodes),
        "n_branches":            n_total,
        "per_node_accuracy":     float(per_node_acc),
        "branch_accuracy":       float(branch_acc),
        "neurite_macro_f1":      neurite_f1,
        "per_class": {
            "soma":           per_class[1],
            "axon":           per_class[2],
            "basal/dendrite": per_class[3],
            "apical":         per_class[4],
        },
        "per_cell_accuracy": {
            "mean":   float(accs.mean()),
            "median": float(np.median(accs)),
            "p10":    float(np.percentile(accs, 10)),
            "p25":    float(np.percentile(accs, 25)),
            "min":    float(accs.min()),
        },
        "per_cell_neurite_f1": {
            "mean":   float(nf1s.mean()),
            "median": float(np.median(nf1s)),
            "p10":    float(np.percentile(nf1s, 10)),
            "p25":    float(np.percentile(nf1s, 25)),
            "min":    float(nf1s.min()),
        },
    }


def _load_baseline_metrics() -> dict | None:
    """Re-derive v12 baseline metrics from the existing per-branch CSV."""
    if not V12_PER_BRANCH_CSV.is_file():
        return None
    rows = list(csv.DictReader(V12_PER_BRANCH_CSV.open("r", encoding="utf-8")))
    # The baseline CSV uses string types; cast
    for r in rows:
        r["pred_label"] = int(r["pred_label"])
        r["gt_label"]   = int(r["gt_label"])
        r["is_wrong"]   = int(r["is_wrong"])
        r["n_nodes"]    = int(r["n_nodes"])
        r["frac_correct_nodes"] = float(r["frac_correct_nodes"])
    # Per-cell accuracy
    per_cell = defaultdict(lambda: {"n_nodes": 0, "n_correct": 0.0, "cell_type": ""})
    for r in rows:
        f = r["file"]; n = r["n_nodes"]; fc = r["frac_correct_nodes"]
        per_cell[f]["n_nodes"] += n
        per_cell[f]["n_correct"] += n * fc
        per_cell[f]["cell_type"] = r["cell_type_gt"]
    cell_list = [{"file": f, "cell_type_gt": v["cell_type"], "n_nodes": v["n_nodes"],
                  "accuracy": v["n_correct"] / max(1, v["n_nodes"]),
                  # neurite F1 can't be derived from branch CSV; leave as accuracy proxy
                  "neurite_f1": v["n_correct"] / max(1, v["n_nodes"])}
                 for f, v in per_cell.items()]
    return _compute_metrics(rows, cell_list)


def _print_comparison(baseline: dict | None, weighted: dict):
    print()
    print("=" * 78)
    print("v12 baseline  vs  v12_weighted  (held-out test, same 2,468 cells)")
    print("=" * 78)
    if baseline is None:
        print("(no baseline metrics on disk to compare)")
        print(json.dumps(weighted, indent=2))
        return

    def line(label, b, w, fmt="{:.4f}", delta_fmt="{:+.4f}"):
        if isinstance(b, (int, float)) and isinstance(w, (int, float)):
            d = w - b
            print(f"  {label:<32} {fmt.format(b):>12}  {fmt.format(w):>12}  {delta_fmt.format(d):>11}")
        else:
            print(f"  {label:<32} {str(b):>12}  {str(w):>12}")

    print(f"  {'metric':<32} {'baseline':>12}  {'weighted':>12}  {'delta':>11}")
    print(f"  {'-'*32} {'-'*12}  {'-'*12}  {'-'*11}")
    line("per-node accuracy",      baseline["per_node_accuracy"],       weighted["per_node_accuracy"])
    line("branch accuracy",        baseline["branch_accuracy"],         weighted["branch_accuracy"])
    line("neurite macro-F1",       baseline["neurite_macro_f1"],        weighted["neurite_macro_f1"])
    print()
    print(f"  Per-class F1:")
    for cls in ("axon", "basal/dendrite", "apical"):
        line(f"    {cls}",
             baseline["per_class"][cls]["f1"],
             weighted["per_class"][cls]["f1"])
    print()
    print(f"  Per-cell accuracy:")
    for k in ("mean", "median", "p10", "p25"):
        line(f"    {k}",
             baseline["per_cell_accuracy"][k],
             weighted["per_cell_accuracy"][k])
    if "neurite_f1" in weighted["per_cell_neurite_f1"]:
        pass  # placeholder
    print()
    print(f"  Per-cell neurite-F1 (only v12_weighted has true F1):")
    for k in ("mean", "median", "p10", "p25"):
        v = weighted["per_cell_neurite_f1"][k]
        print(f"    {k:<28} {'(n/a)':>12}  {v:>12.4f}")
    print()


def main() -> int:
    # Load test files
    if not SPLIT_JSON.is_file():
        print(f"ERROR: split file not found: {SPLIT_JSON}", file=sys.stderr)
        return 2
    split = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    test_files = []
    for ct in ("pyramidal", "interneuron"):
        for name in split["test"].get(ct, []):
            p = DATA_DIR / ct / "swc" / name
            if p.is_file():
                test_files.append((ct, p))
    print(f"Test split: {len(test_files)} cells")

    # Evaluate weighted
    branch_rows, per_cell = _evaluate_model(
        "v12_weighted", W_S1, W_S2, W_GNN, test_files,
    )
    metrics = _compute_metrics(branch_rows, per_cell)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nMetrics -> {OUT_JSON}")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if branch_rows:
        with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(branch_rows[0].keys()))
            w.writeheader()
            for r in branch_rows:
                w.writerow(r)
    print(f"Per-branch -> {OUT_CSV}")

    # Compare
    baseline = _load_baseline_metrics()
    _print_comparison(baseline, metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
