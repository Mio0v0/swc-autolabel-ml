#!/usr/bin/env python3
"""External baselines on the v12 corpus — emits the SAME metric pack as v12.

Trains and evaluates the 4 external baselines (NeuroM-RF, L-Measure-RF,
Sholl-RF, Sholl-MLP) on the requested seed's train/test split, using the
canonical comprehensive metric pack from
``hybrid/comprehensive_metrics.py``. The output dict shape is
IDENTICAL to ``paper/_eval_per_seed_own_test.py``, enabling direct
side-by-side comparison.

Output:
    paper/results/baselines_on_v12.json     list of report dicts
    paper/results/baselines_on_v12.csv      per-cell flat rows (with method col)
    paper/results/baselines_on_v12_table.txt  pretty-print summary
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import csv as _csv

from paper.baselines import _aggregate_per_file, _build_result          # noqa: E402
from paper.external_baselines import predict_with_cache                 # noqa: E402
from hybrid.comprehensive_metrics import (                              # noqa: E402
    print_summary_table, print_per_class_per_cell, print_per_cell_type,
)
from hybrid.evaluate import _file_in_test_bucket                        # noqa: E402

DATA = ROOT / "data" / "v12_uncurated"
# SWCAL_QC_CSV overrides the QC CSV (use for cleaned-dataset experiments).
QC_CSV = Path(os.environ.get("SWCAL_QC_CSV", str(
    ROOT / "paper" / "results" / "corpus_qc_v12_uncurated.csv"
)))
# SWCAL_MODEL_DIR_SUFFIX appends a suffix to the baseline cache dir + result files
# (e.g. "_cleaned" -> baselines_cleaned/, baselines_on_v12_cleaned.json).
MODEL_DIR_SUFFIX = os.environ.get("SWCAL_MODEL_DIR_SUFFIX", "")
CACHE_DIR = ROOT / "paper" / "models" / f"baselines{MODEL_DIR_SUFFIX}"
OUT_JSON  = ROOT / "paper" / "results" / f"baselines_on_v12{MODEL_DIR_SUFFIX}.json"
OUT_CSV   = ROOT / "paper" / "results" / f"baselines_on_v12{MODEL_DIR_SUFFIX}.csv"
OUT_TABLE = ROOT / "paper" / "results" / f"baselines_on_v12{MODEL_DIR_SUFFIX}_table.txt"

METHODS = ("neurom_rf", "lmeasure_rf", "sholl_rf", "sholl_mlp")


def _load_split_from_qc(seed: int, test_size: float = 0.20):
    """Compute train/test split directly from the QC CSV — no v12 dependency.

    Same deterministic hash-bucket logic as v12 training.
    """
    train: dict[str, list[Path]] = {"pyramidal": [], "interneuron": []}
    test:  dict[str, list[Path]] = {"pyramidal": [], "interneuron": []}
    with QC_CSV.open("r", encoding="utf-8") as fh:
        for r in _csv.DictReader(fh):
            if r.get("qc_pass", "").strip().lower() not in ("true", "1"):
                continue
            ct = r["cell_type"]
            p  = Path(r["path"])
            if not p.is_file():
                continue
            (test if _file_in_test_bucket(p.name, seed, test_size) else train)[ct].append(p)
    # Sort for determinism
    for ct in ("pyramidal", "interneuron"):
        train[ct].sort()
        test[ct].sort()
    return train, test


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="all",
                    choices=("all", "neurom_rf", "sholl_rf", "sholl_mlp", "lmeasure_rf"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="Re-run baselines even if already cached in OUT_JSON.")
    ap.add_argument("--force-retrain", action="store_true",
                    help="Ignore the on-disk baseline model cache and re-train sklearn models.")
    args = ap.parse_args()

    if not QC_CSV.is_file():
        raise SystemExit(f"MISSING: {QC_CSV} — run paper._scan_corpus_qc first")
    print(f"Loading split from QC CSV (seed={args.seed}): {QC_CSV}")
    train, test = _load_split_from_qc(args.seed)
    n_tr = sum(len(v) for v in train.values())
    n_te = sum(len(v) for v in test.values())
    print(f"  train: {n_tr}  ({len(train['pyramidal'])} pyr + {len(train['interneuron'])} int)")
    print(f"  test:  {n_te}  ({len(test['pyramidal'])} pyr + {len(test['interneuron'])} int)")
    print()

    methods = list(METHODS) if args.method == "all" else [args.method]
    # Resume support: load any prior reports already on disk
    reports_by_method: dict[str, dict] = {}
    if OUT_JSON.is_file():
        try:
            prior = json.loads(OUT_JSON.read_text(encoding="utf-8"))
            for r in prior.get("reports", []):
                reports_by_method[r["method"]] = r
            print(f"  Found prior reports: {list(reports_by_method.keys())}")
        except Exception:
            pass

    all_rows: list[dict] = []
    overall_t0 = time.perf_counter()
    for method in methods:
        if method in reports_by_method and not args.force:
            print(f"\n=== {method} (CACHED, skip; use --force to re-run) ===")
            continue
        print(f"\n=== {method} ===")
        t0 = time.perf_counter()
        try:
            cache_path = CACHE_DIR / f"{method}.pkl"
            predict_fn = predict_with_cache(
                method, train, seed=args.seed,
                cache_path=cache_path, force_retrain=args.force_retrain,
            )
            print(f"  predicting on {n_te} files...")
            cell_records, all_gt, all_pred, gt_by_ct, pred_by_ct = _aggregate_per_file(test, predict_fn)
            report = _build_result(method, cell_records, all_gt, all_pred, gt_by_ct, pred_by_ct)
            report["wall_clock_min"] = (time.perf_counter() - t0) / 60.0
            report["n_train_cells"]  = n_tr
            reports_by_method[method] = report

            for r in cell_records:
                row = {
                    "method":     method,
                    "file":       Path(r.get("path", "")).name,
                    "cell_type":  r["cell_type"],
                    "n_nodes":    r["n_nodes"],
                }
                # Per-cell accuracy + neurite F1 (flatten for the CSV)
                n_correct = sum(1 for g, p in zip(r["gt"], r["pred"]) if g == p)
                acc = n_correct / r["n_nodes"] if r["n_nodes"] else 0.0
                row["accuracy"] = f"{acc:.4f}"
                from hybrid.evaluate import per_cell_neurite_f1
                row["neurite_macro_f1"] = f"{per_cell_neurite_f1(r['gt'], r['pred'], r['cell_type']):.4f}"
                all_rows.append(row)

            print(f"  acc={report['corpus']['accuracy']:.4f}  "
                  f"neurite_F1={report['corpus']['neurite_macro_f1']:.4f}  "
                  f"pc_F1_mean={report['per_cell']['neurite_macro_f1']['mean']:.4f}  "
                  f"pc_F1_p10={report['per_cell']['neurite_macro_f1']['p10']:.4f}  "
                  f"({report['wall_clock_min']:.1f} min)")
            # Incremental save
            OUT_JSON.write_text(json.dumps({
                "n_train": n_tr, "n_test": n_te, "seed": args.seed,
                "split_source": "QC CSV + hash-bucket split (no v12 dependency)",
                "qc_csv": str(QC_CSV.relative_to(ROOT)),
                "convention": "v11 (macro F1 over GT-present classes only)",
                "reports": list(reports_by_method.values()),
            }, indent=2), encoding="utf-8")
            print(f"  -> incremental save: {OUT_JSON}")
        except Exception as exc:
            import traceback
            print(f"  FAILED ({type(exc).__name__}): {exc}")
            traceback.print_exc()
            continue

    total_min = (time.perf_counter() - overall_t0) / 60.0
    print(f"\nTotal wall clock: {total_min:.1f} min")

    if all_rows:
        with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            for r in all_rows: w.writerow(r)
        print(f"Wrote {OUT_CSV}")

    reports = list(reports_by_method.values())

    # ---- Pretty print ----
    print()
    print("=" * 140)
    print(f"  EXTERNAL BASELINES on v12 corpus  (same train/test split as v12 seed={args.seed}; v11 metric convention)")
    print("=" * 140)
    print_summary_table(reports)

    print()
    print("=" * 140)
    print(f"  PER-CLASS PER-CELL F1 DISTRIBUTIONS")
    print("=" * 140)
    print_per_class_per_cell(reports)

    print()
    print("=" * 140)
    print(f"  PER-CELL-TYPE BREAKDOWN")
    print("=" * 140)
    print_per_cell_type(reports)

    OUT_TABLE.write_text(
        f"Baseline summary (v11 metric convention, seed={args.seed} split)\n"
        + json.dumps({r["method"]: {
            "accuracy": r["corpus"]["accuracy"],
            "neurite_macro_f1": r["corpus"]["neurite_macro_f1"],
            "apical_f1": r["corpus"]["per_class"]["apical"]["f1"],
            "per_cell_F1_mean": r["per_cell"]["neurite_macro_f1"]["mean"],
            "per_cell_F1_p10":  r["per_cell"]["neurite_macro_f1"]["p10"],
            "per_cell_F1_p25":  r["per_cell"]["neurite_macro_f1"]["p25"],
        } for r in reports}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {OUT_TABLE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
