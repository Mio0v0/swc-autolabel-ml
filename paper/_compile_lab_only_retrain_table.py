#!/usr/bin/env python3
"""Aggregate per-seed lab-only-retrained eval JSONs into a multiseed table.

Mirrors the format of `_compile_lab_subset_table.py` for the (b) condition
in paper §5.5: pipeline trained AND evaluated on the lab corpus alone.

Reads:
    paper/results/v12_gt_celltype_seed{42,123,789}_lab_only_retrained.json

Writes:
    paper/results/v12_lab_only_retrain_multiseed_summary.json
    paper/results/v12_lab_only_retrain_multiseed_summary.csv
    paper/results/v12_lab_only_retrain_multiseed_table.txt
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "paper" / "results"

SEEDS = [42, 123, 789]
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
    ("pyramidal_neurite_f1",    "by_cell_type.pyramidal.neurite_macro_f1"),
    ("pyramidal_apical_f1",     "by_cell_type.pyramidal.per_class.apical.f1"),
]


def _dig(d: dict, dotted: str):
    cur = d
    for k in dotted.split("."):
        if cur is None:
            return None
        cur = cur.get(k) if isinstance(cur, dict) else None
    return cur


def main() -> int:
    rows = []
    for seed in SEEDS:
        p = RESULTS / f"v12_gt_celltype_seed{seed}_lab_only_retrained.json"
        if not p.is_file():
            print(f"  MISSING (will skip): {p.name}")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        report = d["report"]
        n = d.get("n_test_cells", 0)
        row = {"seed": seed, "n_test_cells": n}
        for label, key in METRICS:
            v = _dig(report, key)
            row[label] = float(v) if v is not None else None
        rows.append(row)

    if not rows:
        raise SystemExit("No lab-only-retrained JSONs found; run eval first.")

    n_seeds = len(rows)
    summary = {"method": "v12_branch3_gt_celltype_lab_only_retrained",
               "n_seeds": n_seeds, "seeds": [r["seed"] for r in rows]}
    for label, _ in METRICS:
        vals = [r[label] for r in rows if r[label] is not None]
        if not vals:
            continue
        summary[f"{label}_mean"] = round(statistics.mean(vals), 6)
        summary[f"{label}_sd"]   = round(statistics.pstdev(vals) if n_seeds == 1
                                          else statistics.stdev(vals), 6)
        summary[f"{label}_min"]  = round(min(vals), 6)
        summary[f"{label}_max"]  = round(max(vals), 6)

    out_json = RESULTS / "v12_lab_only_retrain_multiseed_summary.json"
    out_json.write_text(json.dumps({"per_seed": rows, "summary": summary},
                                    indent=2), encoding="utf-8")
    out_csv = RESULTS / "v12_lab_only_retrain_multiseed_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    out_txt = RESULTS / "v12_lab_only_retrain_multiseed_table.txt"
    lines = ["Multi-seed v12+Branch3 (GT cell type) -- LAB-ONLY RETRAIN",
             "=" * 64,
             "",
             f"Mean +/- SD over n={n_seeds} seeds ({summary['seeds']}).",
             f"Pipeline trained on hpf_ca1 cells only (1,064 train per seed),",
             f"evaluated on the matching lab-only held-out test split.",
             ""]
    for label, _ in METRICS:
        m = summary.get(f"{label}_mean"); s = summary.get(f"{label}_sd")
        if m is None: continue
        lo = summary[f"{label}_min"]; hi = summary[f"{label}_max"]
        lines.append(f"  {label:30s} {m:6.4f} +/- {s:6.4f}  [{lo:.4f}, {hi:.4f}]")
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 64)
    print(f"  wrote {out_json.name}, {out_csv.name}, {out_txt.name}")
    print("=" * 64)
    print(out_txt.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
