#!/usr/bin/env python3
"""Compile concise paper-facing tables from final flag-model results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "paper" / "results" / "final_flag_leave_one_seed_out_summary.csv",
    )
    ap.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "paper" / "results" / "final_flag_multiseed_labels.csv",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "paper" / "results" / "final_flag_feature_ablation_table.csv",
    )
    ap.add_argument(
        "--out-txt",
        type=Path,
        default=ROOT / "paper" / "results" / "final_flag_feature_ablation_table.txt",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "paper" / "results" / "final_flag_feature_ablation_table.json",
    )
    args = ap.parse_args()

    summary = pd.read_csv(args.summary)
    labels = pd.read_csv(args.labels)
    labels["bad_f060"] = pd.to_numeric(labels["held_out_F1"], errors="coerce") < 0.6
    dataset = {
        "n_rows": int(len(labels)),
        "n_unique_files": int(labels["file"].nunique()),
        "n_bad_f060": int(labels["bad_f060"].sum()),
        "bad_rate_f060": float(labels["bad_f060"].mean()),
        "by_seed": {
            str(seed): {
                "n": int(len(part)),
                "n_bad_f060": int(part["bad_f060"].sum()),
                "bad_rate_f060": float(part["bad_f060"].mean()),
            }
            for seed, part in labels.groupby("model_seed")
        },
        "by_cell_type": {
            str(ct): {
                "n": int(len(part)),
                "n_bad_f060": int(part["bad_f060"].sum()),
                "bad_rate_f060": float(part["bad_f060"].mean()),
            }
            for ct, part in labels.groupby("cell_type_gt")
        },
    }

    picks = summary[
        (
            (summary["operating_point"].eq("validation_recall") & summary["target_value"].round(6).eq(0.8))
            | (summary["operating_point"].eq("fixed_reject_rate") & summary["target_value"].round(6).eq(0.1))
        )
    ].copy()
    keep_cols = [
        "scope",
        "feature_mode",
        "operating_point",
        "target_value",
        "precision_lt_target_mean",
        "precision_lt_target_sd",
        "recall_lt_target_mean",
        "recall_lt_target_sd",
        "n_rejected_mean",
        "fp_mean",
        "fn_mean",
        "kept_f1_p10_mean",
        "test_ap_mean",
    ]
    picks = picks[keep_cols].sort_values(["scope", "operating_point", "feature_mode"])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    picks.to_csv(args.out_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    args.out_json.write_text(
        json.dumps({"dataset": dataset, "rows": picks.to_dict("records")}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "Final flag feature ablation table",
        "=" * 40,
        "Target bad label: per-cell neurite F1 < 0.60",
        f"Rows: {dataset['n_rows']} row-seed pairs, {dataset['n_unique_files']} unique files, "
        f"{dataset['n_bad_f060']} bad ({dataset['bad_rate_f060']:.1%})",
        "",
        "Mean +/- SD over leave-one-seed-out folds.",
        "",
        f"{'scope':<12} {'features':<13} {'op':<17} {'target':>6} {'P':>13} {'R':>13} {'flag':>8} {'FP':>8} {'FN':>8} {'keptP10':>8} {'AP':>8}",
        "-" * 116,
    ]
    for _, r in picks.iterrows():
        lines.append(
            f"{r['scope']:<12} {r['feature_mode']:<13} {r['operating_point']:<17} "
            f"{float(r['target_value']):>6.2f} "
            f"{_fmt(r['precision_lt_target_mean'])}+/-{_fmt(r['precision_lt_target_sd']):<5} "
            f"{_fmt(r['recall_lt_target_mean'])}+/-{_fmt(r['recall_lt_target_sd']):<5} "
            f"{float(r['n_rejected_mean']):>8.1f} "
            f"{float(r['fp_mean']):>8.1f} "
            f"{float(r['fn_mean']):>8.1f} "
            f"{float(r['kept_f1_p10_mean']):>8.4f} "
            f"{float(r['test_ap_mean']):>8.3f}"
        )
    args.out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_txt}")
    print(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
