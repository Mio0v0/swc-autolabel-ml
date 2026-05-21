#!/usr/bin/env python3
"""Definitive ensemble evaluation on the full v12 held-out test set.

Computes every metric correctly:

  Per-NODE level (aggregate over the corpus):
    - per_node_accuracy
    - per_class confusion matrix
    - per_class precision/recall/F1 (corpus-wide)
    - neurite_macro_f1 over {axon, basal, apical} weighted by support

  Per-CELL level (one row per cell, with cell-type-aware metric):
    - per_cell_accuracy: nodes_correct / total
    - per_cell_neurite_f1: macro F1 over the cell's CELL-TYPE'S valid
      neurite labels (pyramidal={2,3,4}, interneuron={2,3}) —
      same convention as v11_final's reported per-file F1.

  Per-cell-type breakdown:
    - same metrics split out for pyramidal vs interneuron separately

  Ensemble validation:
    - disagreement-vs-wrongness Spearman correlation
    - disagreement-binned wrongness rate
    - training-contamination tracking: for each test cell, count how
      many of the N models had it in their TRAIN set (vs test).
      Numbers should be interpreted in light of this — non-zero
      contamination optimistically biases the ensemble eval.

Outputs:
    paper/results/ensemble_eval_full.json     (all summary stats)
    paper/results/ensemble_eval_full_per_cell.csv   (per-cell rows)
    paper/results/ensemble_eval_full_per_branch.csv (per-branch w/ flag info)

Wall time: ~110 min on RTX 4080 (4 models × 2.7 s/cell × 2,468 cells).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.features import parse_swc                       # noqa: E402
from hybrid.pipeline import run_pipeline_on_nodes           # noqa: E402
from hybrid.confidence import _segment_branches             # noqa: E402
from paper.gnn_inference import load_gnn                    # noqa: E402

DATA_DIR = ROOT / "data" / "v12_uncurated"
EVAL_SPLIT_JSON = ROOT / "paper" / "models" / "v12_weighted_gentle" / "train_test_split.json"

MODEL_DIRS = [
    ROOT / "paper" / "models" / "v12_weighted_gentle",
    ROOT / "paper" / "models" / "v12_gentle_seed123",
    ROOT / "paper" / "models" / "v12_gentle_seed456",
    ROOT / "paper" / "models" / "v12_gentle_seed789",
]
OUT_JSON   = ROOT / "paper" / "results" / "ensemble_eval_full.json"
PER_CELL   = ROOT / "paper" / "results" / "ensemble_eval_full_per_cell.csv"
PER_BRANCH = ROOT / "paper" / "results" / "ensemble_eval_full_per_branch.csv"


# ---------------------------------------------------------------------------
# Metric helpers — every one of these is the cell-type-aware version
# ---------------------------------------------------------------------------
def neurite_labels_for(cell_type: str) -> list[int]:
    if cell_type == "pyramidal":
        return [2, 3, 4]
    if cell_type == "interneuron":
        return [2, 3]
    return [2, 3, 4]  # default to most permissive


def per_cell_neurite_f1(gt: list[int], pred: list[int], cell_type: str) -> float:
    """Cell-type-aware per-cell macro F1 — matches v11_final's convention."""
    labels = neurite_labels_for(cell_type)
    return float(f1_score(gt, pred, labels=labels, average="macro", zero_division=0))


def per_cell_accuracy(gt: list[int], pred: list[int]) -> float:
    if not gt:
        return 0.0
    return sum(1 for g, p in zip(gt, pred) if g == p) / len(gt)


# ---------------------------------------------------------------------------
# Models + contamination tracking
# ---------------------------------------------------------------------------
def load_models() -> tuple[list[dict], dict[str, set[str]]]:
    """Load each model + its TRAIN set (for contamination tracking)."""
    models = []
    train_sets: dict[str, set[str]] = {}
    for d in MODEL_DIRS:
        print(f"  Loading {d.name}: ", end="", flush=True)
        models.append({
            "tag":       d.name,
            "s1":        d / "cell_type_classifier.pkl",
            "s2":        d / "branch_classifier.pkl",
            "gnn_state": load_gnn(d / "gnn_apical_basal.pt"),
        })
        split_path = d / "train_test_split.json"
        if split_path.is_file():
            sp = json.loads(split_path.read_text(encoding="utf-8"))
            ts: set[str] = set()
            for ct in ("pyramidal", "interneuron"):
                ts.update(sp.get("train", {}).get(ct, []))
            train_sets[d.name] = ts
        else:
            train_sets[d.name] = set()
        print("OK")
    return models, train_sets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None,
                    help="Optional: limit to first N test cells (debug)")
    args = ap.parse_args()

    print(f"Loading {len(MODEL_DIRS)} models...")
    models, train_sets = load_models()
    n_models = len(models)

    # Use seed=42's test split as the evaluation set (matches the v12_gentle baseline)
    split = json.loads(EVAL_SPLIT_JSON.read_text(encoding="utf-8"))
    test_files: list[tuple[str, Path]] = []
    for ct in ("pyramidal", "interneuron"):
        for name in split["test"].get(ct, []):
            p = DATA_DIR / ct / "swc" / name
            if p.is_file():
                test_files.append((ct, p))
    if args.sample:
        test_files = test_files[: args.sample]
    print(f"\nTest split: {len(test_files)} cells "
          f"({sum(1 for c,_ in test_files if c=='pyramidal')} pyramidal, "
          f"{sum(1 for c,_ in test_files if c=='interneuron')} interneuron)")
    print()

    # Pre-compute contamination per test file: how many of the n_models had it in TRAIN?
    contamination_per_file: dict[str, int] = {}
    for _, p in test_files:
        n_in_train = sum(1 for tag, ts in train_sets.items() if p.name in ts)
        contamination_per_file[p.name] = n_in_train
    contam_dist = Counter(contamination_per_file.values())
    print(f"  Training contamination distribution (n test cells with k models having it in TRAIN):")
    for k in range(n_models + 1):
        print(f"    {k} of {n_models} models trained on it: {contam_dist.get(k, 0)} cells")
    print()

    # ---- Inference loop ----
    per_cell_rows: list[dict] = []
    per_branch_rows: list[dict] = []
    # Corpus-level: collect all per-node (gt, pred_baseline, pred_ensemble) and (disagreement, correct_ensemble)
    all_gt: list[int] = []
    all_pred_baseline: list[int] = []
    all_pred_ensemble: list[int] = []
    all_disagreement: list[float] = []

    t0 = time.perf_counter()
    for i, (ct_gt, path) in enumerate(test_files):
        try:
            nodes = parse_swc(path)
            if not nodes:
                continue
            gt = [n.type for n in nodes]
        except Exception as exc:
            print(f"  WARN {path.name}: {exc}")
            continue

        per_model_labels = []
        cell_type_preds = []
        for m in models:
            pr = run_pipeline_on_nodes(
                nodes, file_path="", stage1_model=m["s1"], stage2_model=m["s2"],
                gnn_state=m["gnn_state"], use_subtree_stage2=True,
            )
            per_model_labels.append(list(pr.node_labels))
            cell_type_preds.append(pr.stage1.cell_type)

        # Baseline = first model (v12_weighted_gentle)
        baseline_pred = per_model_labels[0]
        # Ensemble per-node majority + disagreement
        ens_pred = []
        disag = []
        for n_i in range(len(nodes)):
            votes = [per_model_labels[k][n_i] for k in range(n_models)]
            mc, count = Counter(votes).most_common(1)[0]
            ens_pred.append(int(mc))
            disag.append(1.0 - count / n_models)

        # Per-cell metrics — cell-type-aware
        bl_acc = per_cell_accuracy(gt, baseline_pred)
        en_acc = per_cell_accuracy(gt, ens_pred)
        bl_f1  = per_cell_neurite_f1(gt, baseline_pred, ct_gt)
        en_f1  = per_cell_neurite_f1(gt, ens_pred, ct_gt)

        per_cell_rows.append({
            "file": path.name, "cell_type_gt": ct_gt, "n_nodes": len(gt),
            "n_models_trained_on_it": contamination_per_file[path.name],
            "baseline_accuracy": bl_acc,
            "ensemble_accuracy": en_acc,
            "baseline_neurite_f1": bl_f1,
            "ensemble_neurite_f1": en_f1,
            "mean_disagreement": float(np.mean(disag)),
            "frac_disagreeing": float(np.mean([1 if d > 0 else 0 for d in disag])),
            "cell_type_agreement": float(
                sum(1 for c in cell_type_preds if c == cell_type_preds[0]) / len(cell_type_preds)
            ),
        })

        # Per-branch flag info
        segs = _segment_branches(nodes, ens_pred)
        for bid, idxs in enumerate(segs):
            if not idxs:
                continue
            b_disag = [disag[ii] for ii in idxs]
            b_pred  = [ens_pred[ii] for ii in idxs]
            b_gt    = [gt[ii] for ii in idxs]
            pred_majority = Counter(b_pred).most_common(1)[0][0]
            gt_majority   = Counter(b_gt).most_common(1)[0][0]
            n_correct = sum(1 for ii in idxs if ens_pred[ii] == gt[ii])
            per_branch_rows.append({
                "file": path.name, "branch_id": bid, "n_nodes": len(idxs),
                "pred_label": int(pred_majority), "gt_label": int(gt_majority),
                "is_wrong": int(gt_majority != pred_majority),
                "frac_correct_nodes": n_correct / len(idxs),
                "mean_disagreement": float(np.mean(b_disag)),
                "max_disagreement":  float(np.max(b_disag)),
            })

        all_gt.extend(gt)
        all_pred_baseline.extend(baseline_pred)
        all_pred_ensemble.extend(ens_pred)
        all_disagreement.extend(disag)

        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{len(test_files)}  ({(time.perf_counter()-t0)/60:.1f} min)")

    elapsed_min = (time.perf_counter() - t0) / 60.0
    print(f"\nInference: {elapsed_min:.1f} min")

    # ============================================================
    # Aggregates
    # ============================================================
    # Corpus-level accuracy
    n_total = len(all_gt)
    n_correct_bl = sum(1 for g, p in zip(all_gt, all_pred_baseline) if g == p)
    n_correct_en = sum(1 for g, p in zip(all_gt, all_pred_ensemble) if g == p)
    per_node_acc_bl = n_correct_bl / max(1, n_total)
    per_node_acc_en = n_correct_en / max(1, n_total)

    # Per-class metrics (corpus-wide)
    def _per_class_metrics(gt_list, pred_list):
        out = {}
        for c, name in [(1, "soma"), (2, "axon"), (3, "basal/dendrite"), (4, "apical")]:
            tp = sum(1 for g, p in zip(gt_list, pred_list) if g == c and p == c)
            fp = sum(1 for g, p in zip(gt_list, pred_list) if g != c and p == c)
            fn = sum(1 for g, p in zip(gt_list, pred_list) if g == c and p != c)
            prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            out[name] = {"precision": prec, "recall": rec, "f1": f1,
                         "support_gt_nodes": tp + fn}
        return out

    pc_bl = _per_class_metrics(all_gt, all_pred_baseline)
    pc_en = _per_class_metrics(all_gt, all_pred_ensemble)
    neurite_bl = float(np.mean([pc_bl[k]["f1"] for k in ("axon", "basal/dendrite", "apical")]))
    neurite_en = float(np.mean([pc_en[k]["f1"] for k in ("axon", "basal/dendrite", "apical")]))

    # Confusion matrix
    cm = defaultdict(lambda: defaultdict(int))
    for g, p in zip(all_gt, all_pred_ensemble):
        cm[g][p] += 1

    # Per-cell distributions
    bl_acc_arr = np.array([r["baseline_accuracy"] for r in per_cell_rows])
    en_acc_arr = np.array([r["ensemble_accuracy"] for r in per_cell_rows])
    bl_f1_arr  = np.array([r["baseline_neurite_f1"] for r in per_cell_rows])
    en_f1_arr  = np.array([r["ensemble_neurite_f1"] for r in per_cell_rows])

    def _dist(arr):
        return {"mean": float(arr.mean()), "median": float(np.median(arr)),
                "p10":  float(np.percentile(arr, 10)),
                "p25":  float(np.percentile(arr, 25)),
                "min":  float(arr.min())}

    # Per-cell-type breakdown
    by_ct: dict[str, dict] = {}
    for ct in ("pyramidal", "interneuron"):
        idx = [i for i, r in enumerate(per_cell_rows) if r["cell_type_gt"] == ct]
        if not idx:
            continue
        bl_a = bl_acc_arr[idx]; en_a = en_acc_arr[idx]
        bl_f = bl_f1_arr[idx];  en_f = en_f1_arr[idx]
        by_ct[ct] = {
            "n_cells": len(idx),
            "baseline_accuracy":   _dist(bl_a),
            "ensemble_accuracy":   _dist(en_a),
            "baseline_neurite_f1": _dist(bl_f),
            "ensemble_neurite_f1": _dist(en_f),
        }

    # Disagreement-vs-wrong analysis
    disag_arr = np.array(all_disagreement)
    wrong_arr = np.array([0 if g == p else 1 for g, p in zip(all_gt, all_pred_ensemble)],
                         dtype=np.int64)
    rho = spearmanr(disag_arr, wrong_arr)
    bins = [(0, 0.001), (0.001, 0.26), (0.25, 0.51), (0.5, 0.76), (0.75, 1.001)]
    bin_stats = []
    for lo, hi in bins:
        mask = (disag_arr >= lo) & (disag_arr < hi)
        n = int(mask.sum())
        wn = int(wrong_arr[mask].sum()) if n else 0
        bin_stats.append({
            "range": [round(lo, 3), round(hi, 3)],
            "n_nodes": n,
            "n_wrong": wn,
            "wrongness": (wn / n) if n else 0.0,
        })

    # ============================================================
    # Save outputs
    # ============================================================
    PER_CELL.parent.mkdir(parents=True, exist_ok=True)
    with PER_CELL.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_cell_rows[0].keys()))
        w.writeheader()
        for r in per_cell_rows:
            w.writerow(r)
    with PER_BRANCH.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_branch_rows[0].keys()))
        w.writeheader()
        for r in per_branch_rows:
            w.writerow(r)

    summary = {
        "n_test_cells":    len(per_cell_rows),
        "n_test_nodes":    n_total,
        "n_models":        n_models,
        "model_dirs":      [str(d) for d in MODEL_DIRS],
        "inference_min":   elapsed_min,
        "training_contamination": {str(k): contam_dist.get(k, 0) for k in range(n_models + 1)},
        "baseline_per_node_accuracy": per_node_acc_bl,
        "ensemble_per_node_accuracy": per_node_acc_en,
        "baseline_neurite_macro_f1_corpus": neurite_bl,
        "ensemble_neurite_macro_f1_corpus": neurite_en,
        "per_class_baseline":        pc_bl,
        "per_class_ensemble":        pc_en,
        "confusion_matrix_ensemble": {
            str(g): {str(p): int(cm[g][p]) for p in (1, 2, 3, 4)} for g in (1, 2, 3, 4)
        },
        "per_cell_accuracy_baseline":   _dist(bl_acc_arr),
        "per_cell_accuracy_ensemble":   _dist(en_acc_arr),
        "per_cell_neurite_f1_baseline": _dist(bl_f1_arr),
        "per_cell_neurite_f1_ensemble": _dist(en_f1_arr),
        "per_cell_type_breakdown":      by_ct,
        "disagreement_vs_wrongness_spearman": float(rho.statistic),
        "disagreement_bins": bin_stats,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ============================================================
    # Console report
    # ============================================================
    print()
    print("=" * 72)
    print(f"  v12_gentle ENSEMBLE EVAL — full v12 test set (n={len(per_cell_rows)} cells)")
    print("=" * 72)

    print(f"\n  CORPUS-LEVEL ACCURACY:")
    print(f"    baseline (1 model)     : {per_node_acc_bl:.4f}")
    print(f"    ensemble (4 models)    : {per_node_acc_en:.4f}   (delta {per_node_acc_en - per_node_acc_bl:+.4f})")

    print(f"\n  NEURITE MACRO-F1 (corpus, over axon/basal/apical):")
    print(f"    baseline               : {neurite_bl:.4f}")
    print(f"    ensemble               : {neurite_en:.4f}   (delta {neurite_en - neurite_bl:+.4f})")

    print(f"\n  PER-CLASS F1 (corpus):")
    print(f"    {'class':<16} {'baseline':>10} {'ensemble':>10} {'delta':>10}")
    for cls in ("soma", "axon", "basal/dendrite", "apical"):
        print(f"    {cls:<16} {pc_bl[cls]['f1']:>10.4f} {pc_en[cls]['f1']:>10.4f} "
              f"{pc_en[cls]['f1'] - pc_bl[cls]['f1']:>+10.4f}")

    print(f"\n  PER-CELL ACCURACY:")
    print(f"    {'stat':<10} {'baseline':>10} {'ensemble':>10} {'delta':>10}")
    for k in ("mean", "median", "p10", "p25"):
        print(f"    {k:<10} {summary['per_cell_accuracy_baseline'][k]:>10.4f} "
              f"{summary['per_cell_accuracy_ensemble'][k]:>10.4f} "
              f"{summary['per_cell_accuracy_ensemble'][k] - summary['per_cell_accuracy_baseline'][k]:>+10.4f}")

    print(f"\n  PER-CELL NEURITE-F1 (cell-type-aware; matches v11_final convention):")
    print(f"    {'stat':<10} {'baseline':>10} {'ensemble':>10} {'delta':>10}")
    for k in ("mean", "median", "p10", "p25"):
        print(f"    {k:<10} {summary['per_cell_neurite_f1_baseline'][k]:>10.4f} "
              f"{summary['per_cell_neurite_f1_ensemble'][k]:>10.4f} "
              f"{summary['per_cell_neurite_f1_ensemble'][k] - summary['per_cell_neurite_f1_baseline'][k]:>+10.4f}")

    if by_ct:
        print(f"\n  PER-CELL-TYPE BREAKDOWN:")
        for ct, info in by_ct.items():
            print(f"    {ct}  (n={info['n_cells']})")
            for metric in ("baseline_accuracy", "ensemble_accuracy",
                           "baseline_neurite_f1", "ensemble_neurite_f1"):
                d = info[metric]
                print(f"      {metric:<24}  mean={d['mean']:.4f}  p10={d['p10']:.4f}")

    print(f"\n  DISAGREEMENT-vs-WRONGNESS:")
    print(f"    Spearman corr: {rho.statistic:.4f}")
    print(f"    {'bin':<14} {'n_nodes':>10} {'wrongness':>10}")
    for b in bin_stats:
        if b["n_nodes"] == 0:
            continue
        lo, hi = b["range"]
        print(f"    [{lo:.2f}, {hi:.2f})   {b['n_nodes']:>10,} {b['wrongness']*100:>9.2f}%")

    print(f"\n  TRAINING CONTAMINATION:")
    for k in range(n_models + 1):
        print(f"    {k} of {n_models} models had test cell in TRAIN: "
              f"{contam_dist.get(k, 0):,} cells")
    print()
    print(f"  Summary JSON  -> {OUT_JSON}")
    print(f"  Per-cell CSV  -> {PER_CELL}")
    print(f"  Per-branch CSV-> {PER_BRANCH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
