#!/usr/bin/env python3
"""Compare a seed-specific v12 GT-celltype eval against rerun baselines."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.comprehensive_metrics import (  # noqa: E402
    print_per_cell_type,
    print_per_class_per_cell,
    print_summary_table,
)


MODEL_DIR_SUFFIX = os.environ.get("SWCAL_MODEL_DIR_SUFFIX", "")


def _capture(fn, reports: list[dict]) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(reports)
    return buf.getvalue().rstrip()


def _headline(report: dict) -> dict:
    corpus = report["corpus"]
    per_cell = report["per_cell"]
    per_class = corpus["per_class"]
    return {
        "method": report["method"],
        "n_test_cells": report["n_test_cells"],
        "accuracy": corpus["accuracy"],
        "neurite_macro_f1": corpus["neurite_macro_f1"],
        "axon_f1": per_class["axon"]["f1"],
        "basal_f1": per_class["basal/dendrite"]["f1"],
        "apical_f1": per_class["apical"]["f1"],
        "per_cell_accuracy_mean": per_cell["accuracy"]["mean"],
        "per_cell_accuracy_p10": per_cell["accuracy"]["p10"],
        "per_cell_f1_mean": per_cell["neurite_macro_f1"]["mean"],
        "per_cell_f1_p10": per_cell["neurite_macro_f1"]["p10"],
        "per_cell_f1_p25": per_cell["neurite_macro_f1"]["p25"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--suffix", default=MODEL_DIR_SUFFIX,
                    help="Result/model suffix. Defaults to SWCAL_MODEL_DIR_SUFFIX.")
    ap.add_argument("--v12-json", type=Path, default=None,
                    help="Override v12 eval JSON path.")
    ap.add_argument("--baseline-json", type=Path, default=None,
                    help="Override baseline eval JSON path.")
    ap.add_argument("--out-suffix", default=None,
                    help="Suffix for output table filenames. Defaults to --suffix.")
    args = ap.parse_args()

    suffix = args.suffix
    out_suffix = suffix if args.out_suffix is None else args.out_suffix
    results_dir = ROOT / "paper" / "results"
    v12_json = args.v12_json or results_dir / f"v12_gt_celltype_seed{args.seed}{suffix}.json"
    baseline_json = args.baseline_json or results_dir / f"baselines_on_v12{suffix}.json"
    if not v12_json.is_absolute():
        v12_json = (ROOT / v12_json).resolve()
    if not baseline_json.is_absolute():
        baseline_json = (ROOT / baseline_json).resolve()
    out_txt = results_dir / f"v12_seed{args.seed}{out_suffix}_vs_baselines_table.txt"
    out_json = results_dir / f"v12_seed{args.seed}{out_suffix}_vs_baselines_table.json"

    for p in (v12_json, baseline_json):
        if not p.is_file():
            raise SystemExit(f"MISSING: {p}")

    v12_payload = json.loads(v12_json.read_text(encoding="utf-8"))
    baseline_payload = json.loads(baseline_json.read_text(encoding="utf-8"))
    reports = [v12_payload["report"], *baseline_payload["reports"]]

    summary = _capture(print_summary_table, reports)
    per_class = _capture(print_per_class_per_cell, reports)
    per_ct = _capture(print_per_cell_type, reports)

    lines = [
        "=" * 140,
        f"  v12 seed={args.seed}{suffix} vs rerun baselines",
        "  Stage 1 is bypassed for v12; baselines receive GT cell type; no flag model involved.",
        "=" * 140,
        "",
        "== SUMMARY ==",
        summary,
        "",
        "== PER-CLASS PER-CELL F1 DISTRIBUTIONS ==",
        per_class,
        "",
        "== PER-CELL-TYPE BREAKDOWN ==",
        per_ct,
        "",
        "Inputs:",
        f"  v12:       {v12_json.relative_to(ROOT)}",
        f"  baselines: {baseline_json.relative_to(ROOT)}",
    ]
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps({
        "seed": args.seed,
        "suffix": suffix,
        "out_suffix": out_suffix,
        "v12_eval": str(v12_json.relative_to(ROOT)),
        "baseline_eval": str(baseline_json.relative_to(ROOT)),
        "headlines": [_headline(r) for r in reports],
    }, indent=2), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote {out_txt}")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
