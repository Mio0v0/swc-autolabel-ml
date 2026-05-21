#!/usr/bin/env python3
"""Sampled evaluation of the 4-model v12_gentle ensemble.

For a random subset of v12 test cells (default 200):
  1. Run the ensemble (4 models, majority vote) and a single baseline
     model (v12_weighted_gentle) on each cell.
  2. Compare both against ground truth.
  3. Compute the per-node DISAGREEMENT signal and check how well it
     correlates with actual correctness — the key validation for using
     ensemble disagreement as a flag.

Output:
    paper/results/ensemble_eval_subset.json
    paper/results/ensemble_eval_subset_per_cell.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.features import parse_swc                            # noqa: E402
from hybrid.pipeline import run_pipeline_on_nodes                # noqa: E402
from paper.gnn_inference import load_gnn                         # noqa: E402

DATA_DIR = ROOT / "data" / "v12_uncurated"
SPLIT = ROOT / "paper" / "models" / "v12_weighted_gentle" / "train_test_split.json"
MODEL_DIRS = [
    ROOT / "paper" / "models" / "v12_weighted_gentle",
    ROOT / "paper" / "models" / "v12_gentle_seed123",
    ROOT / "paper" / "models" / "v12_gentle_seed456",
    ROOT / "paper" / "models" / "v12_gentle_seed789",
]
OUT_JSON = ROOT / "paper" / "results" / "ensemble_eval_subset.json"
OUT_CSV  = ROOT / "paper" / "results" / "ensemble_eval_subset_per_cell.csv"


def _neurite_f1(gt, pred):
    return float(f1_score(gt, pred, labels=[2, 3, 4], average="macro", zero_division=0))


def _load_models() -> list[dict]:
    out = []
    for d in MODEL_DIRS:
        print(f"  Loading {d.name}: ", end="", flush=True)
        out.append({
            "tag": d.name,
            "s1": d / "cell_type_classifier.pkl",
            "s2": d / "branch_classifier.pkl",
            "gnn_state": load_gnn(d / "gnn_apical_basal.pt"),
        })
        print("OK")
    return out


def _sample_test_files(n: int, seed: int) -> list[tuple[str, Path]]:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    files: list[tuple[str, Path]] = []
    for ct in ("pyramidal", "interneuron"):
        for name in split["test"].get(ct, []):
            p = DATA_DIR / ct / "swc" / name
            if p.is_file():
                files.append((ct, p))
    rng = random.Random(seed)
    rng.shuffle(files)
    return files[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="Number of test cells to sample")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed")
    args = ap.parse_args()

    print(f"Loading {len(MODEL_DIRS)} models...")
    models = _load_models()
    print()
    test = _sample_test_files(args.n, args.seed)
    print(f"Sampled {len(test)} test cells "
          f"({sum(1 for c,_ in test if c=='pyramidal')} pyramidal, "
          f"{sum(1 for c,_ in test if c=='interneuron')} interneuron)")
    print()

    per_cell_rows = []
    # Aggregate per-node correctness vs disagreement for the disagreement-as-flag analysis
    all_disagreement: list[float] = []
    all_correct: list[int] = []

    t0 = time.perf_counter()
    for i, (ct_gt, path) in enumerate(test):
        try:
            nodes = parse_swc(path)
            if not nodes:
                continue
            gt = [n.type for n in nodes]
        except Exception as exc:
            print(f"  WARN {path.name}: {exc}")
            continue

        per_model_labels = []
        for m in models:
            pr = run_pipeline_on_nodes(
                nodes, file_path="", stage1_model=m["s1"], stage2_model=m["s2"],
                gnn_state=m["gnn_state"], use_subtree_stage2=True,
            )
            per_model_labels.append(list(pr.node_labels))

        # Single-baseline = first model (v12_weighted_gentle)
        baseline_pred = per_model_labels[0]
        # Ensemble = per-node majority vote
        ens_pred = []
        disag = []
        for n_i in range(len(nodes)):
            votes = [per_model_labels[k][n_i] for k in range(len(models))]
            mc, count = Counter(votes).most_common(1)[0]
            ens_pred.append(int(mc))
            disag.append(1.0 - count / len(models))

        baseline_acc = sum(1 for g, p in zip(gt, baseline_pred) if g == p) / len(gt)
        ens_acc      = sum(1 for g, p in zip(gt, ens_pred) if g == p) / len(gt)
        baseline_nf1 = _neurite_f1(gt, baseline_pred)
        ens_nf1      = _neurite_f1(gt, ens_pred)

        per_cell_rows.append({
            "file": path.name, "cell_type_gt": ct_gt, "n_nodes": len(gt),
            "baseline_accuracy": baseline_acc, "ensemble_accuracy": ens_acc,
            "baseline_neurite_f1": baseline_nf1, "ensemble_neurite_f1": ens_nf1,
            "mean_disagreement": float(np.mean(disag)),
            "frac_disagreeing": float(np.mean([1 if d > 0 else 0 for d in disag])),
        })

        # Per-node correctness vs disagreement (use ENSEMBLE prediction)
        for n_i in range(len(nodes)):
            all_disagreement.append(disag[n_i])
            all_correct.append(int(ens_pred[n_i] == gt[n_i]))

        if (i + 1) % 25 == 0:
            print(f"  ... {i+1}/{len(test)}  ({(time.perf_counter()-t0)/60:.1f} min)")

    elapsed = (time.perf_counter() - t0) / 60.0
    print(f"\nInference elapsed: {elapsed:.1f} min")

    # Per-cell aggregates
    bl_acc = np.array([r["baseline_accuracy"] for r in per_cell_rows])
    en_acc = np.array([r["ensemble_accuracy"] for r in per_cell_rows])
    bl_nf1 = np.array([r["baseline_neurite_f1"] for r in per_cell_rows])
    en_nf1 = np.array([r["ensemble_neurite_f1"] for r in per_cell_rows])

    print()
    print("=" * 70)
    print(f"  SAMPLED ENSEMBLE EVAL  (n={len(per_cell_rows)} cells, 4-model ensemble)")
    print("=" * 70)
    def _show(label, b, e, fmt="{:.4f}"):
        d = e - b
        print(f"  {label:<28} baseline={fmt.format(b):<10}  ensemble={fmt.format(e):<10}  delta={d:+.4f}")
    print()
    print("  Per-cell ACCURACY:")
    _show("    mean",   bl_acc.mean(),  en_acc.mean())
    _show("    median", float(np.median(bl_acc)), float(np.median(en_acc)))
    _show("    P10",    float(np.percentile(bl_acc, 10)), float(np.percentile(en_acc, 10)))
    _show("    P25",    float(np.percentile(bl_acc, 25)), float(np.percentile(en_acc, 25)))
    print()
    print("  Per-cell NEURITE-F1:")
    _show("    mean",   bl_nf1.mean(),  en_nf1.mean())
    _show("    median", float(np.median(bl_nf1)), float(np.median(en_nf1)))
    _show("    P10",    float(np.percentile(bl_nf1, 10)), float(np.percentile(en_nf1, 10)))
    _show("    P25",    float(np.percentile(bl_nf1, 25)), float(np.percentile(en_nf1, 25)))

    # ---- Disagreement-as-flag analysis ----
    print()
    print("=" * 70)
    print("  DISAGREEMENT-as-FLAG validation (per-node)")
    print("=" * 70)
    disag_arr = np.array(all_disagreement)
    corr_arr = np.array(all_correct, dtype=np.int64)
    wrong_arr = 1 - corr_arr
    n_total = len(disag_arr)
    n_wrong = int(wrong_arr.sum())
    print(f"  Total nodes: {n_total:,}  Wrong: {n_wrong:,}  ({n_wrong/n_total*100:.2f}%)")
    print()
    print(f'  {"Disagreement":>13s} {"n_nodes":>10s} {"wrongness":>10s}')
    bins = [(0,0.001),(0.001,0.26),(0.25,0.51),(0.5,0.76),(0.75,1.01)]
    for lo, hi in bins:
        mask = (disag_arr >= lo) & (disag_arr < hi)
        if not mask.sum():
            continue
        wn = int(wrong_arr[mask].sum())
        print(f"  [{lo:.2f}, {hi:.2f})  {int(mask.sum()):>10,}  {wn/mask.sum()*100:>9.2f}%")

    # Spearman correlation
    from scipy.stats import spearmanr
    rho = spearmanr(disag_arr, wrong_arr)
    print()
    print(f"  Spearman corr (disagreement vs is_wrong): {rho.statistic:.4f}")
    print(f"  (was 0.04 for single-model softmax — anything > 0.3 is meaningful)")

    # Write outputs
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_cell_rows[0].keys()))
        w.writeheader()
        for r in per_cell_rows:
            w.writerow(r)
    summary = {
        "n_cells_sampled":      len(per_cell_rows),
        "n_models_in_ensemble": len(models),
        "inference_min":        elapsed,
        "baseline_per_cell_accuracy": {
            "mean": float(bl_acc.mean()), "median": float(np.median(bl_acc)),
            "p10":  float(np.percentile(bl_acc, 10)),
        },
        "ensemble_per_cell_accuracy": {
            "mean": float(en_acc.mean()), "median": float(np.median(en_acc)),
            "p10":  float(np.percentile(en_acc, 10)),
        },
        "baseline_per_cell_neurite_f1": {
            "mean": float(bl_nf1.mean()), "median": float(np.median(bl_nf1)),
            "p10":  float(np.percentile(bl_nf1, 10)),
        },
        "ensemble_per_cell_neurite_f1": {
            "mean": float(en_nf1.mean()), "median": float(np.median(en_nf1)),
            "p10":  float(np.percentile(en_nf1, 10)),
        },
        "disagreement_correctness_spearman": float(rho.statistic),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(f"Summary -> {OUT_JSON}")
    print(f"Per-cell -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
