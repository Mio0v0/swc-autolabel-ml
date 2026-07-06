#!/usr/bin/env python3
"""Compile canonical paper-facing outputs from already-computed artifacts.

This script does not retrain or reevaluate models. It creates stable final
filenames for tables that already exist under experiment-specific names and
builds a dataset/QC summary table from the corpus QC catalog.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "paper" / "results"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _copy(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise SystemExit(f"MISSING: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _dataset_tables() -> dict:
    qc_path = RESULTS / "corpus_qc_v12_uncurated.csv"
    if not qc_path.is_file():
        raise SystemExit(f"MISSING: {qc_path}")
    qc = pd.read_csv(qc_path)
    qc["qc_pass"] = qc["qc_pass"].astype(str).str.lower().eq("true")
    qc["n_nodes"] = pd.to_numeric(qc["n_nodes"], errors="coerce")

    rows: list[dict] = []

    def add_row(group: str, label: str, part: pd.DataFrame) -> None:
        total = int(len(part))
        passed = int(part["qc_pass"].sum())
        failed = total - passed
        pass_nodes = part.loc[part["qc_pass"], "n_nodes"].dropna()
        rows.append(
            {
                "group": group,
                "label": label,
                "total_cells": total,
                "qc_pass_cells": passed,
                "qc_fail_cells": failed,
                "qc_pass_rate": passed / max(1, total),
                "median_nodes_qc_pass": float(pass_nodes.median()) if len(pass_nodes) else "",
                "mean_nodes_qc_pass": float(pass_nodes.mean()) if len(pass_nodes) else "",
            }
        )

    add_row("overall", "all", qc)
    for cell_type, part in qc.groupby("cell_type", dropna=False):
        add_row("cell_type", str(cell_type), part)
    for source, part in qc.groupby("source", dropna=False):
        add_row("source", str(source), part)
    for (source, cell_type), part in qc.groupby(["source", "cell_type"], dropna=False):
        add_row("source_cell_type", f"{source}:{cell_type}", part)

    out_csv = RESULTS / "final_dataset_corpus_table.csv"
    out_md = RESULTS / "final_dataset_corpus_table.md"
    out_json = RESULTS / "final_dataset_corpus_table.json"
    _write_csv(out_csv, rows)

    fail_reason = (
        qc.loc[~qc["qc_pass"]]
        .groupby("qc_reason", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    fail_reason_csv = RESULTS / "final_dataset_qc_fail_reasons.csv"
    fail_reason.to_csv(fail_reason_csv, index=False)

    lines = [
        "# Final dataset / QC table",
        "",
        "| Group | Label | Total | QC pass | QC fail | Pass rate | Median nodes pass |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        median = row["median_nodes_qc_pass"]
        median_text = f"{float(median):.0f}" if median != "" else ""
        lines.append(
            f"| {row['group']} | {row['label']} | {row['total_cells']} | "
            f"{row['qc_pass_cells']} | {row['qc_fail_cells']} | "
            f"{_fmt_pct(float(row['qc_pass_rate']))} | {median_text} |"
        )
    lines.extend(
        [
            "",
            "## QC fail reasons",
            "",
            "| Reason | Count |",
            "|---|---:|",
        ]
    )
    for _, r in fail_reason.iterrows():
        lines.append(f"| {r['qc_reason']} | {int(r['count'])} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(
        json.dumps(
            {
                "source": str(qc_path.relative_to(ROOT)),
                "rows": rows,
                "fail_reasons": fail_reason.to_dict("records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "dataset_csv": str(out_csv.relative_to(ROOT)),
        "dataset_md": str(out_md.relative_to(ROOT)),
        "dataset_json": str(out_json.relative_to(ROOT)),
        "fail_reasons_csv": str(fail_reason_csv.relative_to(ROOT)),
    }


def _canonical_aliases() -> dict:
    # Full v12-vs-baseline table.
    _copy(RESULTS / "v12_multiseed_vs_baselines_summary.csv", RESULTS / "final_multiseed_vs_baselines.csv")
    _copy(RESULTS / "v12_multiseed_vs_baselines_summary.json", RESULTS / "final_multiseed_vs_baselines.json")
    _copy(RESULTS / "v12_multiseed_vs_baselines_summary.txt", RESULTS / "final_multiseed_vs_baselines_table.txt")

    # V12-only summary row.
    summary = pd.read_csv(RESULTS / "v12_multiseed_vs_baselines_summary.csv")
    v12_summary = summary[summary["method"].astype(str).eq("v12_branch3_gt_celltype")].copy()
    v12_summary.to_csv(RESULTS / "final_multiseed_v12_branch3_summary.csv", index=False)

    payload = json.loads((RESULTS / "v12_multiseed_vs_baselines_summary.json").read_text(encoding="utf-8"))
    v12_rows = [r for r in payload.get("rows", []) if r.get("method") == "v12_branch3_gt_celltype"]
    v12_sum = [r for r in payload.get("summary", []) if r.get("method") == "v12_branch3_gt_celltype"]
    (RESULTS / "final_multiseed_v12_branch3_summary.json").write_text(
        json.dumps(
            {
                "source": "paper/results/v12_multiseed_vs_baselines_summary.json",
                "seeds": payload.get("seeds", []),
                "rows": v12_rows,
                "summary": v12_sum,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = ["Final multi-seed v12+Branch3 summary", "=" * 40, ""]
    if not v12_summary.empty:
        row = v12_summary.iloc[0]
        metrics = [
            "accuracy",
            "neurite_macro_f1",
            "axon_f1",
            "basal_f1",
            "apical_f1",
            "per_cell_f1_mean",
            "per_cell_f1_p10",
            "pyramidal_neurite_f1",
            "pyramidal_apical_f1",
            "interneuron_neurite_f1",
        ]
        lines.append("method: v12_branch3_gt_celltype  (n=3)")
        for metric in metrics:
            lines.append(
                f"  {metric:<28} {float(row[f'{metric}_mean']):.4f} +/- "
                f"{float(row[f'{metric}_sd']):.4f} "
                f"[{float(row[f'{metric}_min']):.4f}, {float(row[f'{metric}_max']):.4f}]"
            )
    (RESULTS / "final_multiseed_v12_branch3_table.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    # Final flag table aliases.
    _copy(RESULTS / "final_flag_feature_ablation_table.csv", RESULTS / "final_flag_feature_ablation.csv")
    _copy(RESULTS / "final_flag_feature_ablation_table.txt", RESULTS / "final_flag_feature_ablation.txt")
    _copy(RESULTS / "final_flag_feature_ablation_table.json", RESULTS / "final_flag_feature_ablation.json")
    _copy(RESULTS / "final_flag_leave_one_seed_out_summary.txt", RESULTS / "final_flag_leave_one_seed_out_table.txt")
    _copy(RESULTS / "final_flag_multiseed_labels.csv", RESULTS / "final_flag_multiseed_cells.csv")

    return {
        "final_multiseed_vs_baselines": "paper/results/final_multiseed_vs_baselines.{csv,json,table.txt}",
        "final_multiseed_v12_branch3": "paper/results/final_multiseed_v12_branch3_{summary.csv,summary.json,table.txt}",
        "final_flag_feature_ablation": "paper/results/final_flag_feature_ablation.{csv,json,txt}",
        "final_flag_leave_one_seed_out_table": "paper/results/final_flag_leave_one_seed_out_table.txt",
        "final_flag_multiseed_cells": "paper/results/final_flag_multiseed_cells.csv",
    }


def main() -> int:
    dataset = _dataset_tables()
    aliases = _canonical_aliases()
    manifest = {
        "dataset_outputs": dataset,
        "canonical_aliases": aliases,
    }
    out_manifest = RESULTS / "final_canonical_outputs_manifest.json"
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out_manifest}")
    for section in manifest.values():
        for value in section.values():
            print(f"  {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
