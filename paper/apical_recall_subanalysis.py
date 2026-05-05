#!/usr/bin/env python3
"""Apical-recall sub-analysis from existing per-file evaluation CSVs.

Cross-references the v9 per-file F1 distribution with per-source and
per-cell-type splits to surface where the apical class is hardest.
Reads files we already have, no engine reruns needed.

Inputs:
    paper/results/snapshots/v9_final_subtree_gnn.csv
    paper/results/snapshots/v9_per_source_breakdown.json

Outputs:
    paper/results/snapshots/apical_recall_subanalysis.json
    paper/results/snapshots/apical_recall_subanalysis.txt

Usage::

    python -m paper.apical_recall_subanalysis
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = ROOT / "paper" / "results" / "snapshots"


def _load_per_file(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                row["neurite_macro_f1_stage23"] = float(row.get("neurite_macro_f1_stage23") or 0.0)
                row["n_nodes"] = int(row.get("n_nodes") or 0)
            except (TypeError, ValueError):
                continue
            rows.append(row)
    return rows


def _classify_source(filename: str) -> str:
    name = filename.lower()
    if name.startswith("hpf_ca1__"):
        return "hpf_ca1"
    if name.startswith("allen__"):
        return "allen"
    if name.startswith("lab__"):
        return "lab"
    if name.startswith("neuromorpho__"):
        return "neuromorpho"
    return "other"


def _summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    arr = np.array(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "min": float(arr.min()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path,
        default=SNAPSHOT_DIR / "v9_final_subtree_gnn.csv",
    )
    parser.add_argument(
        "--out-json", type=Path,
        default=SNAPSHOT_DIR / "apical_recall_subanalysis.json",
    )
    parser.add_argument(
        "--out-text", type=Path,
        default=SNAPSHOT_DIR / "apical_recall_subanalysis.txt",
    )
    args = parser.parse_args()

    rows = _load_per_file(args.csv)
    print(f"Loaded {len(rows)} files from {args.csv}")

    # Group by (cell_type, source)
    bucket_f1: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        ct = (r.get("cell_type") or "").lower()
        src = _classify_source(Path(r.get("path", "")).name)
        bucket_f1[(ct, src)].append(r["neurite_macro_f1_stage23"])

    summaries: dict[str, dict] = {}
    for (ct, src), vals in sorted(bucket_f1.items()):
        summaries[f"{ct}__{src}"] = {
            "cell_type": ct, "source": src, **_summarize(vals),
        }

    # Worst-5 files (lowest F1) — these are the hard cases the paper
    # discussion section should address.
    worst5 = sorted(rows, key=lambda r: r["neurite_macro_f1_stage23"])[:10]

    # Tail-vs-head F1 cohorts: bottom 10% vs the rest. Useful for
    # framing 'most cells are easy; the long tail is concentrated in N
    # files'.
    f1s = sorted(r["neurite_macro_f1_stage23"] for r in rows)
    tail_thresh = float(np.percentile(f1s, 10))
    tail_files = [r for r in rows if r["neurite_macro_f1_stage23"] <= tail_thresh]
    head_files = [r for r in rows if r["neurite_macro_f1_stage23"] > tail_thresh]

    payload = {
        "n_files": len(rows),
        "by_celltype_and_source": summaries,
        "worst_files": [
            {
                "path": r["path"],
                "cell_type": r["cell_type"],
                "source": _classify_source(Path(r["path"]).name),
                "n_nodes": r["n_nodes"],
                "neurite_macro_f1": r["neurite_macro_f1_stage23"],
            }
            for r in worst5
        ],
        "tail_vs_head": {
            "tail_threshold_p10": tail_thresh,
            "n_tail": len(tail_files),
            "n_head": len(head_files),
            "tail_f1_mean": float(np.mean([r["neurite_macro_f1_stage23"] for r in tail_files])) if tail_files else None,
            "head_f1_mean": float(np.mean([r["neurite_macro_f1_stage23"] for r in head_files])) if head_files else None,
            "tail_source_counts": dict(_count_sources(tail_files)),
            "tail_celltype_counts": dict(_count_celltypes(tail_files)),
        },
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Plain-text rollup
    lines = [
        "Apical-recall sub-analysis (v9 final, held-out test split)",
        "=" * 60,
        "",
        f"Total files in test split: {len(rows)}",
        "",
        "Per-file F1 by cell type and source:",
        f"  {'group':<26s}  {'n':>4s}  {'mean':>7s}  {'median':>7s}  {'p10':>7s}  {'p25':>7s}",
    ]
    for key, s in sorted(summaries.items(), key=lambda kv: -kv[1].get("mean", 0.0)):
        lines.append(
            f"  {key:<26s}  {s['n']:>4d}  {s['mean']:>7.4f}  "
            f"{s['median']:>7.4f}  {s['p10']:>7.4f}  {s['p25']:>7.4f}"
        )

    lines.append("")
    lines.append("Tail (bottom 10% by F1):")
    tv = payload["tail_vs_head"]
    lines.append(
        f"  threshold = {tv['tail_threshold_p10']:.4f}  |  "
        f"tail = {tv['n_tail']} files (mean F1 = {tv['tail_f1_mean']:.4f})  |  "
        f"head = {tv['n_head']} files (mean F1 = {tv['head_f1_mean']:.4f})"
    )
    if tv.get("tail_source_counts"):
        lines.append(
            f"  tail source breakdown: " +
            ", ".join(f"{k}={v}" for k, v in sorted(tv["tail_source_counts"].items(), key=lambda kv: -kv[1]))
        )
    if tv.get("tail_celltype_counts"):
        lines.append(
            f"  tail cell-type breakdown: " +
            ", ".join(f"{k}={v}" for k, v in sorted(tv["tail_celltype_counts"].items(), key=lambda kv: -kv[1]))
        )

    lines.append("")
    lines.append("Worst 10 files:")
    for w in payload["worst_files"]:
        lines.append(
            f"  F1={w['neurite_macro_f1']:.4f}  n_nodes={w['n_nodes']:>6d}  "
            f"{w['source']:<12s} {w['cell_type']:<12s}  {Path(w['path']).name}"
        )

    text = "\n".join(lines)
    args.out_text.write_text(text, encoding="utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print()
    print(text)
    print(f"\nWrote {args.out_json}")
    print(f"Wrote {args.out_text}")
    return 0


def _count_sources(files: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in files:
        counts[_classify_source(Path(r["path"]).name)] += 1
    return dict(counts)


def _count_celltypes(files: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in files:
        counts[(r.get("cell_type") or "").lower()] += 1
    return dict(counts)


if __name__ == "__main__":
    sys.exit(main())
