#!/usr/bin/env python3
"""T3: score per-cell F1 on the held-out test cells of seeds 42 + 789.

For each cell in (seed42_test ∪ seed789_test) that is still QC-passed
(i.e. wasn't removed by the T2 in-place cleanup), run the v12 pipeline
using the seed-model that HELD OUT that cell — so the cell was never
seen during that model's training. This gives an honest per-cell F1.

Per-cell outputs written to CSV:
    file, cell_type_gt, n_nodes, seed_used (42, 789, or "both"),
    held_out_neurite_F1, axon_F1, basal_F1, apical_F1,
    stage1_pred, stage1_conf, stage1_correct,
    n_components, soma_z, apical_mean_z, apical_z_extent, axon_frac,
    pred_class_dist (a|b|p|s breakdown),
    elapsed_s

Watchdogs:
    - Pre-flight: 10 largest cells first; any one >30s aborts immediately
    - Per-cell warning if >5s
    - Progress every 100 cells
    - Skip cell on inference exception; don't kill the run

Usage:
    python -m paper._score_heldout_f1                   # full run
    python -m paper._score_heldout_f1 --max-cells 5     # smoke test

Output:
    paper/results/heldout_per_cell_f1.csv
    paper/results/heldout_per_cell_f1_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.features import parse_swc                                # noqa: E402
from hybrid.pipeline import run_pipeline_on_nodes                    # noqa: E402
from hybrid.evaluate import per_cell_neurite_f1                      # noqa: E402
from paper.gnn_inference import load_gnn                             # noqa: E402

QC_CSV   = ROOT / "paper" / "results" / "corpus_qc_v12_uncurated.csv"
OUT_CSV  = ROOT / "paper" / "results" / "heldout_per_cell_f1.csv"
OUT_JSON = ROOT / "paper" / "results" / "heldout_per_cell_f1_summary.json"

SEEDS    = [42, 789]
DATA_DIR = ROOT / "data" / "v12_uncurated"

# Watchdog thresholds
PREFLIGHT_N        = 10
PREFLIGHT_LIMIT_S  = 30.0
PER_CELL_WARN_S    = 5.0


def _load_split(seed: int) -> set[str]:
    p = ROOT / "paper" / "models" / f"v12_gentle_seed{seed}" / "train_test_split.json"
    sp = json.loads(p.read_text(encoding="utf-8"))
    out: set[str] = set()
    for ct in ("pyramidal", "interneuron"):
        out.update(sp["test"].get(ct, []))
    return out


def _per_class_f1(gt: list[int], pred: list[int], cell_type: str) -> dict[str, float]:
    """Return per-class F1 for axon/basal/apical present in GT.

    Class IDs: 1=soma, 2=axon, 3=basal, 4=apical. F1 reported only for
    classes present in GT (matches per_cell_neurite_f1 convention).
    """
    out = {"axon_f1": float("nan"), "basal_f1": float("nan"), "apical_f1": float("nan")}
    if not gt:
        return out
    y_t = np.asarray(gt, dtype=int)
    y_p = np.asarray(pred, dtype=int)
    for cls, name in ((2, "axon_f1"), (3, "basal_f1"), (4, "apical_f1")):
        if (y_t == cls).any():
            f1 = f1_score((y_t == cls).astype(int),
                          (y_p == cls).astype(int),
                          zero_division=0)
            out[name] = float(f1)
    return out


def _structural_features(nodes) -> dict[str, float]:
    """Compute lightweight structural fingerprints (for failure-mode analysis).

    Pre-built indices: O(N) — no O(N^2) scans.
    """
    out: dict[str, float] = {}
    n = len(nodes)
    if n == 0:
        return out

    # Connected components via parent links + children adjacency
    id_to_idx = {nd.id: i for i, nd in enumerate(nodes)}
    parent_idx = [-1] * n
    children: list[list[int]] = [[] for _ in nodes]
    for i, nd in enumerate(nodes):
        if nd.parent != -1 and nd.parent in id_to_idx:
            p = id_to_idx[nd.parent]
            parent_idx[i] = p
            children[p].append(i)

    visited = [False] * n
    n_comp = 0
    for start in range(n):
        if visited[start]:
            continue
        n_comp += 1
        stack = [start]
        while stack:
            i = stack.pop()
            if visited[i]:
                continue
            visited[i] = True
            p = parent_idx[i]
            if p >= 0 and not visited[p]:
                stack.append(p)
            for j in children[i]:
                if not visited[j]:
                    stack.append(j)
    out["n_components"] = float(n_comp)

    soma_zs   = [nd.z for nd in nodes if nd.type == 1]
    axon_zs   = [nd.z for nd in nodes if nd.type == 2]
    basal_zs  = [nd.z for nd in nodes if nd.type == 3]
    apical_zs = [nd.z for nd in nodes if nd.type == 4]
    out["soma_z"]          = float(np.mean(soma_zs)) if soma_zs else float("nan")
    out["axon_frac"]       = (len(axon_zs)  / n) if n else 0.0
    out["apical_frac"]     = (len(apical_zs)/ n) if n else 0.0
    out["basal_frac"]      = (len(basal_zs) / n) if n else 0.0
    if apical_zs:
        out["apical_mean_z"]   = float(np.mean(apical_zs))
        out["apical_z_extent"] = float(max(apical_zs) - min(apical_zs))
        if soma_zs:
            out["apical_above_soma"] = out["apical_mean_z"] - out["soma_z"]
    else:
        out["apical_mean_z"]     = float("nan")
        out["apical_z_extent"]   = 0.0
        out["apical_above_soma"] = float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-cells", type=int, default=None,
                    help="Process only N cells (smoke test)")
    args = ap.parse_args()

    if not QC_CSV.is_file():
        raise SystemExit(f"MISSING: {QC_CSV}")

    # --- Load splits + models ---
    print("Loading both seed models + splits...")
    test_by_seed: dict[int, set[str]] = {s: _load_split(s) for s in SEEDS}
    print(f"  seed=42  test: {len(test_by_seed[42])}")
    print(f"  seed=789 test: {len(test_by_seed[789])}")

    s1_path: dict[int, Path] = {}
    s2_path: dict[int, Path] = {}
    gnn_state: dict[int, object] = {}
    for s in SEEDS:
        d = ROOT / "paper" / "models" / f"v12_gentle_seed{s}"
        s1_path[s] = d / "cell_type_classifier.pkl"
        s2_path[s] = d / "branch_classifier.pkl"
        gnn_state[s] = load_gnn(d / "gnn_apical_basal.pt")
        print(f"  seed={s} models loaded")

    # --- Build per-cell file map from CURRENT QC CSV ---
    # This automatically excludes the 89 cells dropped by T2.
    qc_path_by_basename: dict[str, tuple[str, Path]] = {}   # basename -> (cell_type_gt, path)
    with QC_CSV.open("r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("qc_pass", "").strip().lower() not in ("true", "1"):
                continue
            p = Path(r["path"])
            qc_path_by_basename[p.name] = (r["cell_type"], p)
    print(f"  QC-passed cells available: {len(qc_path_by_basename)}")

    # --- Compute the eval set: cells in any test split AND still QC-passed ---
    union_test = test_by_seed[42] | test_by_seed[789]
    eval_set = union_test & set(qc_path_by_basename.keys())
    n_42_only  = len((test_by_seed[42] - test_by_seed[789]) & eval_set)
    n_789_only = len((test_by_seed[789] - test_by_seed[42]) & eval_set)
    n_both     = len((test_by_seed[42] & test_by_seed[789]) & eval_set)
    print(f"\nHeld-out cells to score: {len(eval_set)}")
    print(f"  seed=42  test only : {n_42_only}")
    print(f"  seed=789 test only : {n_789_only}")
    print(f"  in BOTH tests      : {n_both}  (will run both models)")
    n_post_t2 = len(union_test) - len(eval_set)
    if n_post_t2:
        print(f"  (excluded {n_post_t2} cells previously in test that were dropped by T2)")

    # --- Order: largest first for the pre-flight, then the rest ---
    # We parse a quick stat by reading file size as a proxy (avoid full parse twice)
    eval_list = sorted(
        eval_set,
        key=lambda b: -qc_path_by_basename[b][1].stat().st_size,
    )
    if args.max_cells:
        eval_list = eval_list[: args.max_cells]
        print(f"  --max-cells: limiting to {len(eval_list)} cells")

    # --- Inference loop ---
    rows: list[dict] = []
    n_skipped = 0
    n_failed  = 0
    t0 = time.perf_counter()
    print(f"\nStarting inference (pre-flight on first {min(PREFLIGHT_N, len(eval_list))} largest cells)...")

    for i, basename in enumerate(eval_list):
        ct_gt, path = qc_path_by_basename[basename]
        seeds_used = []
        if basename in test_by_seed[42]:  seeds_used.append(42)
        if basename in test_by_seed[789]: seeds_used.append(789)
        if not seeds_used:
            n_skipped += 1
            continue

        cell_t0 = time.perf_counter()
        try:
            nodes = parse_swc(path)
            if not nodes:
                n_skipped += 1
                continue
            gt = [n.type for n in nodes]
        except Exception as exc:
            print(f"  WARN parse {basename}: {exc}")
            n_failed += 1
            continue

        # Predict with each seed where the cell was held-out, average labels by vote
        # For simplicity, if both apply: pick label per-node by majority across the
        # two predictions; if tied, prefer the one with higher mean confidence.
        per_seed_preds: dict[int, list[int]] = {}
        per_seed_confs: dict[int, list[float]] = {}
        s1_results: list[tuple[str, float]] = []
        for s in seeds_used:
            try:
                pr = run_pipeline_on_nodes(
                    nodes, file_path="",
                    stage1_model=s1_path[s], stage2_model=s2_path[s],
                    gnn_state=gnn_state[s], use_subtree_stage2=True,
                )
            except Exception as exc:
                print(f"  WARN pipeline {basename} seed={s}: {exc}")
                continue
            per_seed_preds[s] = list(pr.node_labels)
            per_seed_confs[s] = list(pr.node_confidences)
            s1_results.append((pr.stage1.cell_type, float(pr.stage1.confidence)))

        if not per_seed_preds:
            n_failed += 1
            continue

        # Combine predictions across seeds (majority vote per node; tie -> higher conf)
        if len(per_seed_preds) == 1:
            pred = next(iter(per_seed_preds.values()))
            seed_used_str = str(next(iter(per_seed_preds.keys())))
        else:
            preds_a = per_seed_preds[seeds_used[0]]
            preds_b = per_seed_preds[seeds_used[1]]
            confs_a = per_seed_confs[seeds_used[0]]
            confs_b = per_seed_confs[seeds_used[1]]
            pred = []
            for j in range(len(preds_a)):
                if preds_a[j] == preds_b[j]:
                    pred.append(preds_a[j])
                else:
                    pred.append(preds_a[j] if confs_a[j] >= confs_b[j] else preds_b[j])
            seed_used_str = "both"

        # Metrics
        pc_f1 = per_cell_neurite_f1(gt, pred, ct_gt)
        per_class = _per_class_f1(gt, pred, ct_gt)
        n_correct = sum(1 for g, q in zip(gt, pred) if g == q)
        pc_acc = n_correct / max(1, len(gt))

        # Stage 1 ensemble: majority + max conf
        s1_pred = Counter(s for s, _ in s1_results).most_common(1)[0][0]
        s1_conf = float(np.mean([c for _, c in s1_results]))
        s1_correct = (s1_pred == ct_gt)

        structural = _structural_features(nodes)
        pred_counter = Counter(pred)

        cell_dt = time.perf_counter() - cell_t0

        # Pre-flight watchdog (first PREFLIGHT_N cells)
        if i < PREFLIGHT_N:
            print(f"  pre-flight {i+1}/{PREFLIGHT_N}: {basename} n={len(nodes)}  "
                  f"F1={pc_f1:.3f}  {cell_dt:.1f}s", flush=True)
            if cell_dt > PREFLIGHT_LIMIT_S:
                raise SystemExit(
                    f"WATCHDOG: cell {basename} (n={len(nodes)}) took {cell_dt:.1f}s "
                    f"to score (limit {PREFLIGHT_LIMIT_S}s). Aborting before this "
                    f"wastes hours."
                )
        elif cell_dt > PER_CELL_WARN_S:
            print(f"  SLOW: {basename} n={len(nodes)} took {cell_dt:.1f}s", flush=True)

        rows.append({
            "file":               basename,
            "cell_type_gt":       ct_gt,
            "n_nodes":            len(gt),
            "seed_used":          seed_used_str,
            "held_out_F1":        f"{pc_f1:.4f}",
            "acc":                f"{pc_acc:.4f}",
            "axon_F1":            f"{per_class['axon_f1']:.4f}"   if not np.isnan(per_class["axon_f1"])   else "",
            "basal_F1":           f"{per_class['basal_f1']:.4f}"  if not np.isnan(per_class["basal_f1"])  else "",
            "apical_F1":          f"{per_class['apical_f1']:.4f}" if not np.isnan(per_class["apical_f1"]) else "",
            "stage1_pred":        s1_pred,
            "stage1_conf":        f"{s1_conf:.4f}",
            "stage1_correct":     str(bool(s1_correct)),
            "n_components":       int(structural.get("n_components", 1)),
            "soma_z":             f"{structural.get('soma_z', 0):.2f}"          if not np.isnan(structural.get("soma_z", float("nan"))) else "",
            "apical_mean_z":      f"{structural.get('apical_mean_z', 0):.2f}"   if not np.isnan(structural.get("apical_mean_z", float("nan"))) else "",
            "apical_above_soma":  f"{structural.get('apical_above_soma', 0):.2f}" if not np.isnan(structural.get("apical_above_soma", float("nan"))) else "",
            "apical_z_extent":    f"{structural.get('apical_z_extent', 0):.2f}",
            "axon_frac":          f"{structural.get('axon_frac', 0):.3f}",
            "apical_frac":        f"{structural.get('apical_frac', 0):.3f}",
            "basal_frac":         f"{structural.get('basal_frac', 0):.3f}",
            "pred_axon":          pred_counter.get(2, 0),
            "pred_basal":         pred_counter.get(3, 0),
            "pred_apical":        pred_counter.get(4, 0),
            "elapsed_s":          f"{cell_dt:.2f}",
            "source":             basename.split("__", 1)[0] if "__" in basename else "",
        })

        if (i + 1) % 100 == 0:
            elapsed_min = (time.perf_counter() - t0) / 60.0
            eta_min = elapsed_min * (len(eval_list) - i - 1) / max(1, i + 1)
            print(f"  ... {i+1}/{len(eval_list)}  ({elapsed_min:.1f} min, ETA {eta_min:.0f} min)", flush=True)

    elapsed = (time.perf_counter() - t0) / 60.0
    print(f"\nDone: {len(rows)} scored, {n_skipped} skipped, {n_failed} failed  ({elapsed:.1f} min)")

    if not rows:
        print("No rows -- aborting before write.")
        return 1

    # Sort: worst F1 first (useful for failure-mode analysis)
    rows.sort(key=lambda r: float(r["held_out_F1"]))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"Wrote {OUT_CSV}")

    # Summary
    f1s = np.array([float(r["held_out_F1"]) for r in rows])
    summary = {
        "n_scored":      len(rows),
        "n_skipped":     n_skipped,
        "n_failed":      n_failed,
        "elapsed_min":   elapsed,
        "F1_mean":       float(f1s.mean()),
        "F1_median":     float(np.median(f1s)),
        "F1_p10":        float(np.percentile(f1s, 10)),
        "F1_p25":        float(np.percentile(f1s, 25)),
        "by_cell_type": {
            ct: {
                "n":      int((np.array([r["cell_type_gt"] for r in rows]) == ct).sum()),
                "F1_p10": float(np.percentile(
                    [float(r["held_out_F1"]) for r in rows if r["cell_type_gt"] == ct], 10
                )) if any(r["cell_type_gt"] == ct for r in rows) else None,
                "F1_mean": float(np.mean(
                    [float(r["held_out_F1"]) for r in rows if r["cell_type_gt"] == ct]
                )) if any(r["cell_type_gt"] == ct for r in rows) else None,
            }
            for ct in ("pyramidal", "interneuron")
        },
        "seed_used_counts": dict(Counter(r["seed_used"] for r in rows)),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")

    print()
    print(f"  F1 mean: {summary['F1_mean']:.4f}  median: {summary['F1_median']:.4f}  "
          f"P10: {summary['F1_p10']:.4f}  P25: {summary['F1_p25']:.4f}")
    for ct, s in summary["by_cell_type"].items():
        if s["F1_p10"] is not None:
            print(f"  {ct:<12}: n={s['n']:>5}  F1 mean={s['F1_mean']:.4f}  P10={s['F1_p10']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
