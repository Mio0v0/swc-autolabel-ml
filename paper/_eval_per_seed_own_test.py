#!/usr/bin/env python3
"""Per-seed v12 evaluation on each seed's OWN held-out test set.

Emits the canonical comprehensive metric pack (see
``hybrid/comprehensive_metrics.py``) so every output dict is shape-
identical to the baseline eval (``paper/_eval_baselines_on_v12.py``).
This makes baseline-vs-v12 comparison trivial.

Output:
    paper/results/per_seed_own_test.json    list of report dicts
    paper/results/per_seed_own_test.csv     per-cell flat rows (with seed col)

Wall time: ~25 min per seed × 4 seeds ≈ 100 min on RTX 4080.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.features import parse_swc                                  # noqa: E402
from hybrid.pipeline import run_pipeline_on_nodes                      # noqa: E402
from hybrid.comprehensive_metrics import (                             # noqa: E402
    build_full_report, print_summary_table, print_per_class_per_cell,
    print_per_cell_type, CLASS_NAMES,
)
from paper.gnn_inference import load_gnn                               # noqa: E402

DATA_DIR = ROOT / "data" / "v12_uncurated"
# Default 4-seed evaluation. Override via env var:
#   SWCAL_EVAL_SEEDS=42,789
import os as _os
_seeds_env = _os.environ.get("SWCAL_EVAL_SEEDS", "").strip()
if _seeds_env:
    SEEDS = [int(s) for s in _seeds_env.split(",") if s.strip()]
else:
    SEEDS = [42, 123, 456, 789]

# SWCAL_MODEL_DIR_SUFFIX appends to model dir names (e.g. "_cleaned")
MODEL_DIR_SUFFIX = _os.environ.get("SWCAL_MODEL_DIR_SUFFIX", "")

OUT_JSON = ROOT / "paper" / "results" / "per_seed_own_test.json"
OUT_CSV  = ROOT / "paper" / "results" / "per_seed_own_test.csv"


def eval_one_seed(seed: int) -> tuple[dict, list[dict]]:
    model_dir = ROOT / "paper" / "models" / f"v12_gentle_seed{seed}{MODEL_DIR_SUFFIX}"
    split_json = model_dir / "train_test_split.json"
    if not split_json.is_file():
        raise SystemExit(f"MISSING: {split_json}")

    print(f"\n=== seed={seed} ===")
    print(f"  loading model from {model_dir.name}...")
    s1 = model_dir / "cell_type_classifier.pkl"
    s2 = model_dir / "branch_classifier.pkl"
    gnn_state = load_gnn(model_dir / "gnn_apical_basal.pt")

    sp = json.loads(split_json.read_text(encoding="utf-8"))
    test_files: list[tuple[str, Path]] = []
    for ct in ("pyramidal", "interneuron"):
        for fn in sp["test"].get(ct, []):
            p = DATA_DIR / ct / "swc" / fn
            if p.is_file():
                test_files.append((ct, p))
    print(f"  test files (own held-out split): {len(test_files)} cells")

    cell_records: list[dict] = []
    stage1_records: list[dict] = []   # Stage 1 GT vs pred (cell-type accuracy)
    gt_pool:   list[int] = []
    pred_pool: list[int] = []
    csv_rows:  list[dict] = []

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

        pr = run_pipeline_on_nodes(
            nodes, file_path="", stage1_model=s1, stage2_model=s2,
            gnn_state=gnn_state, use_subtree_stage2=True,
        )
        pred = list(pr.node_labels)

        cell_records.append({
            "cell_type": ct_gt,
            "n_nodes":   len(gt),
            "gt":        gt,
            "pred":      pred,
        })
        stage1_records.append({
            "cell_type_gt": ct_gt,
            "stage1_pred":  pr.stage1.cell_type,
            "stage1_conf":  float(pr.stage1.confidence),
        })
        gt_pool.extend(gt)
        pred_pool.extend(pred)

        # CSV flat row (kept lean)
        n_correct = sum(1 for g, p in zip(gt, pred) if g == p)
        acc = n_correct / len(gt) if len(gt) else 0.0
        csv_rows.append({
            "seed": seed, "file": path.name, "cell_type_gt": ct_gt,
            "n_nodes": len(gt), "accuracy": f"{acc:.4f}",
        })

        if (i + 1) % 200 == 0:
            print(f"    ... {i+1}/{len(test_files)} ({(time.perf_counter()-t0)/60:.1f} min)")

    elapsed = (time.perf_counter() - t0) / 60.0
    print(f"  inference: {elapsed:.1f} min")

    report = build_full_report(
        method=f"v12_gentle_seed{seed}",
        gt_pool=gt_pool, pred_pool=pred_pool,
        cell_records=cell_records,
        inference_min=elapsed,
        stage1_records=stage1_records,
    )
    pc = report["per_cell"]
    s1 = report.get("stage1", {})
    print(f"  acc={report['corpus']['accuracy']:.4f}  "
          f"neurite_F1={report['corpus']['neurite_macro_f1']:.4f}  "
          f"pc_F1_mean={pc['neurite_macro_f1']['mean']:.4f}  "
          f"pc_F1_p10={pc['neurite_macro_f1']['p10']:.4f}  "
          f"stage1_acc={s1.get('accuracy', 0.0):.4f}")
    return report, csv_rows


def main() -> int:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    all_csv_rows: list[dict] = []

    overall_t0 = time.perf_counter()
    for seed in SEEDS:
        report, rows = eval_one_seed(seed)
        reports.append(report)
        all_csv_rows.extend(rows)
        # Incremental save after each seed
        OUT_JSON.write_text(json.dumps({
            "n_seeds":     len(reports),
            "convention":  "v11 (macro F1 over GT-present classes only)",
            "note":        "Each row: model seed=X evaluated on seed=Xs OWN held-out test split. Zero contamination per row.",
            "reports":     reports,
        }, indent=2), encoding="utf-8")
        print(f"  -> incremental save: {OUT_JSON}")

    overall_min = (time.perf_counter() - overall_t0) / 60.0
    print(f"\nTotal wall clock: {overall_min:.1f} min")

    if all_csv_rows:
        with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_csv_rows[0].keys()))
            w.writeheader()
            for r in all_csv_rows:
                w.writerow(r)
        print(f"Wrote {OUT_CSV}")

    # ---- Pretty-print summaries ----
    print()
    print("=" * 140)
    print(f"  PER-SEED V12 EVAL  (each model on its OWN held-out test set; contamination-free)")
    print("=" * 140)
    print_summary_table(reports)

    print()
    print("=" * 140)
    print(f"  PER-CLASS PER-CELL F1 DISTRIBUTIONS  (only cells where that class is in GT)")
    print("=" * 140)
    print_per_class_per_cell(reports)

    print()
    print("=" * 140)
    print(f"  PER-CELL-TYPE BREAKDOWN  (pyramidal vs interneuron + 4x4 confusion)")
    print("=" * 140)
    print_per_cell_type(reports)

    # Across-seed stats
    import numpy as np
    print()
    print("=" * 80)
    print(f"  ACROSS-SEED MEAN +/- STD")
    print("=" * 80)
    keys = [
        ("stage1.accuracy",                        lambda r: r.get("stage1", {}).get("accuracy", 0.0)),
        ("stage1.pyramidal_recall",                lambda r: r.get("stage1", {}).get("per_cell_type_recall", {}).get("pyramidal", {}).get("accuracy", 0.0)),
        ("stage1.interneuron_recall",              lambda r: r.get("stage1", {}).get("per_cell_type_recall", {}).get("interneuron", {}).get("accuracy", 0.0)),
        ("corpus.accuracy",                        lambda r: r["corpus"]["accuracy"]),
        ("corpus.neurite_macro_f1",                lambda r: r["corpus"]["neurite_macro_f1"]),
        ("corpus.apical_f1",                       lambda r: r["corpus"]["per_class"]["apical"]["f1"]),
        ("per_cell.accuracy.mean",                 lambda r: r["per_cell"]["accuracy"]["mean"]),
        ("per_cell.accuracy.p10",                  lambda r: r["per_cell"]["accuracy"]["p10"]),
        ("per_cell.neurite_macro_f1.mean",         lambda r: r["per_cell"]["neurite_macro_f1"]["mean"]),
        ("per_cell.neurite_macro_f1.p10",          lambda r: r["per_cell"]["neurite_macro_f1"]["p10"]),
        ("per_cell.neurite_macro_f1.p25",          lambda r: r["per_cell"]["neurite_macro_f1"]["p25"]),
    ]
    for name, fn in keys:
        vals = np.array([fn(r) for r in reports])
        print(f"  {name:<40}  {vals.mean():.4f} +/- {vals.std():.4f}  "
              f"[min={vals.min():.4f}, max={vals.max():.4f}]")
    print()
    print(f"  JSON -> {OUT_JSON}")
    print(f"  CSV  -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
