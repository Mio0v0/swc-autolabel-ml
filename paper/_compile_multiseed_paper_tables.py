#!/usr/bin/env python3
"""Compile seed-level paper tables for v12+Branch3, baselines, and ablations."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"MISSING: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_row(seed: int, report: dict, family: str) -> dict:
    corpus = report["corpus"]
    per_class = corpus["per_class"]
    pc = report["per_cell"]
    by_ct = report.get("by_cell_type", {})
    pyr = by_ct.get("pyramidal", {})
    intr = by_ct.get("interneuron", {})

    def ct_metric(block: dict, key: str, default: float = 0.0) -> float:
        try:
            if key == "neurite_f1":
                return float(block["corpus"]["neurite_macro_f1"])
            if key == "accuracy":
                return float(block["corpus"]["accuracy"])
            return float(block["corpus"]["per_class"][key]["f1"])
        except Exception:
            return default

    return {
        "seed": seed,
        "family": family,
        "method": report["method"],
        "n_test_cells": report["n_test_cells"],
        "accuracy": corpus["accuracy"],
        "neurite_macro_f1": corpus["neurite_macro_f1"],
        "axon_f1": per_class["axon"]["f1"],
        "basal_f1": per_class["basal/dendrite"]["f1"],
        "apical_f1": per_class["apical"]["f1"],
        "per_cell_accuracy_mean": pc["accuracy"]["mean"],
        "per_cell_accuracy_p10": pc["accuracy"]["p10"],
        "per_cell_f1_mean": pc["neurite_macro_f1"]["mean"],
        "per_cell_f1_p10": pc["neurite_macro_f1"]["p10"],
        "per_cell_f1_p25": pc["neurite_macro_f1"]["p25"],
        "pyramidal_accuracy": ct_metric(pyr, "accuracy"),
        "pyramidal_neurite_f1": ct_metric(pyr, "neurite_f1"),
        "pyramidal_apical_f1": ct_metric(pyr, "apical"),
        "interneuron_accuracy": ct_metric(intr, "accuracy"),
        "interneuron_neurite_f1": ct_metric(intr, "neurite_f1"),
    }


def _mean_sd(rows: list[dict], group_key: str, metric_keys: list[str]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row[group_key]), []).append(row)
    out: list[dict] = []
    for group, items in sorted(groups.items()):
        rec = {group_key: group, "n_seeds": len(items)}
        for key in metric_keys:
            vals = [float(r[key]) for r in items]
            rec[f"{key}_mean"] = statistics.fmean(vals)
            rec[f"{key}_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
            rec[f"{key}_min"] = min(vals)
            rec[f"{key}_max"] = max(vals)
        out.append(rec)
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def _summary_text(title: str, summary_rows: list[dict], metric_keys: list[str], group_key: str) -> str:
    lines = [title, "=" * len(title), ""]
    for row in summary_rows:
        lines.append(f"{group_key}: {row[group_key]}  (n={row['n_seeds']})")
        for key in metric_keys:
            lines.append(
                f"  {key:<28} {_fmt(row[f'{key}_mean'])} +/- {_fmt(row[f'{key}_sd'])} "
                f"[{_fmt(row[f'{key}_min'])}, {_fmt(row[f'{key}_max'])}]"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="123,42,789")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    results = ROOT / "paper" / "results"
    metric_keys = [
        "accuracy",
        "neurite_macro_f1",
        "axon_f1",
        "basal_f1",
        "apical_f1",
        "per_cell_f1_mean",
        "per_cell_f1_p10",
        "per_cell_f1_p25",
        "pyramidal_neurite_f1",
        "pyramidal_apical_f1",
        "interneuron_neurite_f1",
    ]

    comparison_rows: list[dict] = []
    for seed in seeds:
        v12_payload = _load_json(results / f"v12_gt_celltype_seed{seed}.json")
        v12_row = _metric_row(seed, v12_payload["report"], "v12")
        v12_row["method"] = "v12_branch3_gt_celltype"
        comparison_rows.append(v12_row)

        baseline_path = results / f"baselines_on_v12_seed{seed}.json"
        baseline_payload = _load_json(baseline_path)
        for report in baseline_payload["reports"]:
            comparison_rows.append(_metric_row(seed, report, "baseline"))

    comparison_summary = _mean_sd(comparison_rows, "method", metric_keys)
    _write_csv(results / "v12_multiseed_vs_baselines_rows.csv", comparison_rows)
    _write_csv(results / "v12_multiseed_vs_baselines_summary.csv", comparison_summary)
    (results / "v12_multiseed_vs_baselines_summary.json").write_text(
        json.dumps({"seeds": seeds, "rows": comparison_rows, "summary": comparison_summary}, indent=2),
        encoding="utf-8",
    )
    (results / "v12_multiseed_vs_baselines_summary.txt").write_text(
        _summary_text("Multi-seed v12+Branch3 vs baselines", comparison_summary, metric_keys, "method"),
        encoding="utf-8",
    )

    ablation_rows: list[dict] = []
    for seed in seeds:
        for label, suffix in (("branch3_on", ""), ("branch3_off", "_no_branch3")):
            payload = _load_json(results / f"v12_gt_celltype_seed{seed}{suffix}.json")
            row = _metric_row(seed, payload["report"], "ablation")
            row["method"] = label
            row["branch3_enabled"] = payload.get("branch3_enabled")
            ablation_rows.append(row)

    ablation_summary = _mean_sd(ablation_rows, "method", metric_keys)
    by_seed = {seed: {} for seed in seeds}
    for row in ablation_rows:
        by_seed[int(row["seed"])][row["method"]] = row
    delta_rows: list[dict] = []
    for seed, block in by_seed.items():
        on = block["branch3_on"]
        off = block["branch3_off"]
        delta = {"seed": seed, "method": "branch3_on_minus_off"}
        for key in metric_keys:
            delta[key] = float(on[key]) - float(off[key])
        delta_rows.append(delta)
    delta_summary = _mean_sd(delta_rows, "method", metric_keys)

    _write_csv(results / "v12_branch3_ablation_rows.csv", ablation_rows)
    _write_csv(results / "v12_branch3_ablation_deltas.csv", delta_rows)
    _write_csv(results / "v12_branch3_ablation_summary.csv", ablation_summary + delta_summary)
    (results / "v12_branch3_ablation_summary.json").write_text(
        json.dumps({
            "seeds": seeds,
            "rows": ablation_rows,
            "deltas": delta_rows,
            "summary": ablation_summary,
            "delta_summary": delta_summary,
        }, indent=2),
        encoding="utf-8",
    )
    text = _summary_text("Branch3 ablation", ablation_summary, metric_keys, "method")
    text += "\n" + _summary_text("Branch3 deltas", delta_summary, metric_keys, "method")
    (results / "v12_branch3_ablation_summary.txt").write_text(text, encoding="utf-8")

    print(f"Wrote {results / 'v12_multiseed_vs_baselines_summary.txt'}")
    print(f"Wrote {results / 'v12_branch3_ablation_summary.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
