#!/usr/bin/env python3
"""Aggregate per-seed baseline lab-subset evals into a multiseed table.

Mirrors `_compile_lab_subset_table.py` but reads the per-seed baseline
JSONs from `_eval_baselines_lab_subset.py` and produces one row per
method (mean +/- SD across seeds 42, 123, 789).

Reads:
    paper/results/baselines_on_v12_seed{42,123,789}_lab_only.json

Writes:
    paper/results/baselines_lab_subset_multiseed_summary.json
    paper/results/baselines_lab_subset_multiseed_summary.csv
    paper/results/baselines_lab_subset_multiseed_table.txt
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "paper" / "results"

SEEDS = [42, 123, 789]
METHODS = ("neurom_rf", "sholl_mlp", "sholl_rf", "lmeasure_rf")
METRICS = [
    ("accuracy",                "corpus.accuracy"),
    ("neurite_macro_f1",        "corpus.neurite_macro_f1"),
    ("axon_f1",                 "corpus.per_class.axon.f1"),
    ("basal_f1",                "corpus.per_class.basal/dendrite.f1"),
    ("apical_f1",               "corpus.per_class.apical.f1"),
    ("per_cell_f1_mean",        "per_cell.neurite_macro_f1.mean"),
    ("per_cell_f1_p10",         "per_cell.neurite_macro_f1.p10"),
    ("per_cell_f1_p25",         "per_cell.neurite_macro_f1.p25"),
    ("per_cell_accuracy_mean",  "per_cell.accuracy.mean"),
    ("per_cell_accuracy_p10",   "per_cell.accuracy.p10"),
]


def _dig(d, dotted):
    cur = d
    for k in dotted.split("."):
        if cur is None: return None
        cur = cur.get(k) if isinstance(cur, dict) else None
    return cur


def main() -> int:
    # method -> seed -> {metric: value}
    by_method: dict[str, dict[int, dict]] = {m: {} for m in METHODS}
    for seed in SEEDS:
        p = RESULTS / f"baselines_on_v12_seed{seed}_lab_only.json"
        if not p.is_file():
            print(f"  MISSING: {p.name}")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for report in d["reports"]:
            method = report["method"]
            row = {"n_test_cells": report.get("n_test_cells", 0)}
            for label, key in METRICS:
                v = _dig(report, key)
                row[label] = float(v) if v is not None else None
            by_method[method][seed] = row

    summary_rows: list[dict] = []
    for method in METHODS:
        per_seed = by_method[method]
        if len(per_seed) < 1:
            print(f"  SKIP {method}: no seeds")
            continue
        n_seeds = len(per_seed)
        row = {"method": method, "n_seeds": n_seeds, "seeds": sorted(per_seed.keys())}
        for label, _ in METRICS:
            vals = [per_seed[s][label] for s in sorted(per_seed) if per_seed[s][label] is not None]
            if not vals: continue
            row[f"{label}_mean"] = round(statistics.mean(vals), 6)
            row[f"{label}_sd"]   = round(statistics.pstdev(vals) if n_seeds == 1
                                          else statistics.stdev(vals), 6)
            row[f"{label}_min"]  = round(min(vals), 6)
            row[f"{label}_max"]  = round(max(vals), 6)
        summary_rows.append(row)

    out_json = RESULTS / "baselines_lab_subset_multiseed_summary.json"
    out_json.write_text(json.dumps({"summary": summary_rows}, indent=2), encoding="utf-8")

    if summary_rows:
        out_csv = RESULTS / "baselines_lab_subset_multiseed_summary.csv"
        all_keys = []
        for r in summary_rows:
            for k in r.keys():
                if k not in all_keys: all_keys.append(k)
        with out_csv.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=all_keys)
            w.writeheader()
            for r in summary_rows:
                w.writerow({k: (r.get(k, "") if not isinstance(r.get(k), list) else ";".join(map(str, r.get(k)))) for k in all_keys})

    out_txt = RESULTS / "baselines_lab_subset_multiseed_table.txt"
    lines = ["Multi-seed BASELINES on lab subset (hpf_ca1, GT cell type)",
             "=" * 64,
             "",
             f"Mean +/- SD over seeds {SEEDS}.",
             "Test cells: ~290 per seed (lab pyramidals only). Baselines",
             "trained on the full cross-source corpus; here scored on lab cells only.",
             ""]
    for row in summary_rows:
        lines.append("")
        lines.append(f"method: {row['method']}  (n={row['n_seeds']})")
        for label, _ in METRICS:
            m = row.get(f"{label}_mean"); s = row.get(f"{label}_sd")
            if m is None: continue
            lo = row[f"{label}_min"]; hi = row[f"{label}_max"]
            lines.append(f"  {label:30s} {m:6.4f} +/- {s:6.4f}  [{lo:.4f}, {hi:.4f}]")
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 64)
    print(f"  wrote {out_json.name}, {out_txt.name}")
    print("=" * 64)
    print(out_txt.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
