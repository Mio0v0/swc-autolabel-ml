#!/usr/bin/env python3
"""Baselines on lab subset — re-run external baseline predictions restricted
to hpf_ca1 test cells, using the EXISTING cross-source-trained baseline
models. No retraining; mirrors `_eval_baselines_on_v12.py` but with a
source-prefix filter so the resulting numbers are directly comparable to
the v12 lab-subset table.

Output (per seed):
    paper/results/baselines_on_v12_seed<seed>_lab_only.json
    paper/results/baselines_on_v12_seed<seed>_lab_only.csv
    paper/results/baselines_on_v12_seed<seed>_lab_only_table.txt
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
from hybrid.evaluate import _file_in_test_bucket, per_cell_neurite_f1   # noqa: E402

DATA = ROOT / "data" / "v12_uncurated"
# Use the FULL cross-source QC CSV so the split logic matches what the
# baseline cache was trained against. We filter the *test* set to lab-only
# AFTER the split is computed, so the baselines see exactly the cells they
# were trained on, just scored on the lab slice.
QC_CSV = ROOT / "paper" / "results" / "corpus_qc_v12_uncurated.csv"
CACHE_DIR = ROOT / "paper" / "models" / "baselines"
SOURCE_PREFIX = "hpf_ca1__"

METHODS = ("neurom_rf", "lmeasure_rf", "sholl_rf", "sholl_mlp")


def _load_split_from_qc(seed: int, test_size: float = 0.20):
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
    for ct in ("pyramidal", "interneuron"):
        train[ct].sort()
        test[ct].sort()
    return train, test


def _filter_to_prefix(files: dict[str, list[Path]], prefix: str) -> dict[str, list[Path]]:
    return {ct: [p for p in plist if p.name.startswith(prefix)]
            for ct, plist in files.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="all",
                    choices=("all", "neurom_rf", "sholl_rf", "sholl_mlp", "lmeasure_rf"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--source-prefix", default=SOURCE_PREFIX)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not QC_CSV.is_file():
        raise SystemExit(f"MISSING: {QC_CSV}")

    out_json  = ROOT / "paper" / "results" / f"baselines_on_v12_seed{args.seed}_lab_only.json"
    out_csv   = ROOT / "paper" / "results" / f"baselines_on_v12_seed{args.seed}_lab_only.csv"
    out_table = ROOT / "paper" / "results" / f"baselines_on_v12_seed{args.seed}_lab_only_table.txt"

    print(f"Loading split from QC CSV (seed={args.seed}): {QC_CSV.name}")
    train, test_all = _load_split_from_qc(args.seed)
    test_lab = _filter_to_prefix(test_all, args.source_prefix)
    n_tr = sum(len(v) for v in train.values())
    n_te_all = sum(len(v) for v in test_all.values())
    n_te_lab = sum(len(v) for v in test_lab.values())
    print(f"  train (cross-source, for cache key): {n_tr}")
    print(f"  test (all): {n_te_all}")
    print(f"  test (lab-only prefix='{args.source_prefix}'): {n_te_lab}  "
          f"({len(test_lab['pyramidal'])} pyr + {len(test_lab['interneuron'])} int)")
    print()
    if n_te_lab == 0:
        raise SystemExit("No lab cells in this seed's test split.")

    methods = list(METHODS) if args.method == "all" else [args.method]
    reports_by_method: dict[str, dict] = {}
    if out_json.is_file():
        try:
            prior = json.loads(out_json.read_text(encoding="utf-8"))
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
            if not cache_path.is_file():
                print(f"  MISSING cached model {cache_path} — run _eval_baselines_on_v12.py first")
                continue
            predict_fn = predict_with_cache(
                method, train, seed=args.seed,
                cache_path=cache_path, force_retrain=False,
            )
            print(f"  predicting on {n_te_lab} lab test files...")
            cell_records, all_gt, all_pred, gt_by_ct, pred_by_ct = _aggregate_per_file(test_lab, predict_fn)
            report = _build_result(method, cell_records, all_gt, all_pred, gt_by_ct, pred_by_ct)
            report["wall_clock_min"] = (time.perf_counter() - t0) / 60.0
            report["n_train_cells"]  = n_tr
            report["n_test_cells"]   = n_te_lab
            report["source_prefix"]  = args.source_prefix
            reports_by_method[method] = report

            for r in cell_records:
                row = {
                    "method":     method,
                    "file":       Path(r.get("path", "")).name,
                    "cell_type":  r["cell_type"],
                    "n_nodes":    r["n_nodes"],
                }
                n_correct = sum(1 for g, p in zip(r["gt"], r["pred"]) if g == p)
                acc = n_correct / r["n_nodes"] if r["n_nodes"] else 0.0
                row["accuracy"] = f"{acc:.4f}"
                row["neurite_macro_f1"] = f"{per_cell_neurite_f1(r['gt'], r['pred'], r['cell_type']):.4f}"
                all_rows.append(row)

            print(f"  acc={report['corpus']['accuracy']:.4f}  "
                  f"neurite_F1={report['corpus']['neurite_macro_f1']:.4f}  "
                  f"pc_F1_mean={report['per_cell']['neurite_macro_f1']['mean']:.4f}  "
                  f"pc_F1_p10={report['per_cell']['neurite_macro_f1']['p10']:.4f}  "
                  f"({report['wall_clock_min']:.1f} min)")
            out_json.write_text(json.dumps({
                "n_train": n_tr, "n_test_lab": n_te_lab, "seed": args.seed,
                "source_prefix": args.source_prefix,
                "split_source": "cross-source QC CSV + hash-bucket split, test filtered to lab prefix",
                "qc_csv": str(QC_CSV.relative_to(ROOT)),
                "convention": "v11 (macro F1 over GT-present classes only)",
                "reports": list(reports_by_method.values()),
            }, indent=2), encoding="utf-8")
            print(f"  -> incremental save: {out_json.name}")
        except Exception as exc:
            import traceback
            print(f"  FAILED ({type(exc).__name__}): {exc}")
            traceback.print_exc()
            continue

    total_min = (time.perf_counter() - overall_t0) / 60.0
    print(f"\nTotal wall clock: {total_min:.1f} min")

    if all_rows:
        with out_csv.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            for r in all_rows: w.writerow(r)
        print(f"Wrote {out_csv.name}")

    reports = list(reports_by_method.values())
    print()
    print("=" * 140)
    print(f"  BASELINES on v12 corpus -- LAB SUBSET (prefix='{args.source_prefix}'), seed={args.seed}")
    print("=" * 140)
    print_summary_table(reports)
    print()
    print("=" * 140)
    print(f"  PER-CLASS PER-CELL F1 DISTRIBUTIONS (lab subset)")
    print("=" * 140)
    print_per_class_per_cell(reports)
    print()
    print("=" * 140)
    print(f"  PER-CELL-TYPE BREAKDOWN (lab subset)")
    print("=" * 140)
    print_per_cell_type(reports)

    out_table.write_text(
        f"Baseline summary (lab subset prefix='{args.source_prefix}', seed={args.seed})\n"
        + json.dumps({r["method"]: {
            "accuracy": r["corpus"]["accuracy"],
            "neurite_macro_f1": r["corpus"]["neurite_macro_f1"],
            "apical_f1": r["corpus"]["per_class"].get("apical", {}).get("f1"),
            "axon_f1": r["corpus"]["per_class"].get("axon", {}).get("f1"),
            "basal_f1": r["corpus"]["per_class"].get("basal/dendrite", {}).get("f1"),
            "per_cell_F1_mean": r["per_cell"]["neurite_macro_f1"]["mean"],
            "per_cell_F1_p10":  r["per_cell"]["neurite_macro_f1"]["p10"],
            "per_cell_F1_p25":  r["per_cell"]["neurite_macro_f1"]["p25"],
        } for r in reports}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {out_table.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
