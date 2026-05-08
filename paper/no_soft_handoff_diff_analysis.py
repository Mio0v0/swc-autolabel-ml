#!/usr/bin/env python3
"""Where does removing soft-handoff actually help?

The no_soft_handoff ablation showed +0.0089 mean F1 and a much larger
+0.0377 P10 jump vs v9_final. This script computes the per-file delta
(no_soft_handoff F1 − v9_final F1) and breaks it down by cell type and
source, to confirm the P10 win comes from rescuing the hard cells we
care about (apical-bearing pyramidals) rather than incidentally fixing
a few easy interneurons.

Inputs:
    paper/results/snapshots/v9_final_subtree_gnn.csv
    paper/results/snapshots/eval_no_soft_handoff_per_file.csv

Outputs:
    paper/results/snapshots/no_soft_handoff_diff_analysis.txt

Usage::

    python -m paper.no_soft_handoff_diff_analysis
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = ROOT / "paper" / "results" / "snapshots"


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


def _load_per_file(csv_path: Path, f1_col: str) -> dict[str, dict]:
    """Return {basename: {f1, cell_type, n_nodes, source}}."""
    out: dict[str, dict] = {}
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fname = Path(str(row["path"])).name
            try:
                f1 = float(row.get(f1_col) or 0.0)
            except (TypeError, ValueError):
                continue
            try:
                n_nodes = int(row.get("n_nodes") or 0)
            except (TypeError, ValueError):
                n_nodes = 0
            out[fname] = {
                "path": row["path"],
                "f1": f1,
                "cell_type": (row.get("cell_type") or "").lower(),
                "n_nodes": n_nodes,
                "source": _classify_source(fname),
            }
    return out


def _summarize_diffs(diffs: list[float]) -> dict:
    if not diffs:
        return {"n": 0}
    arr = np.array(diffs, dtype=np.float64)
    helped = int((arr > 1e-6).sum())
    hurt = int((arr < -1e-6).sum())
    ties = int(((arr <= 1e-6) & (arr >= -1e-6)).sum())
    return {
        "n": int(arr.size),
        "mean_diff": float(arr.mean()),
        "median_diff": float(np.median(arr)),
        "p10_diff": float(np.percentile(arr, 10)),
        "p90_diff": float(np.percentile(arr, 90)),
        "max_diff": float(arr.max()),
        "min_diff": float(arr.min()),
        "helped": helped,
        "hurt": hurt,
        "ties": ties,
    }


def _fmt_summary(s: dict) -> str:
    if s.get("n", 0) == 0:
        return "n=0"
    return (
        f"n={s['n']:>4d}  "
        f"mean Δ={s['mean_diff']:>+.4f}  "
        f"median Δ={s['median_diff']:>+.4f}  "
        f"P10 Δ={s['p10_diff']:>+.4f}  "
        f"helped/hurt/tied={s['helped']:>3d}/{s['hurt']:>3d}/{s['ties']:>3d}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v9-csv", type=Path,
        default=SNAPSHOT_DIR / "v9_final_subtree_gnn.csv",
    )
    parser.add_argument(
        "--no-handoff-csv", type=Path,
        default=SNAPSHOT_DIR / "eval_no_soft_handoff_per_file.csv",
    )
    parser.add_argument(
        "--out-text", type=Path,
        default=SNAPSHOT_DIR / "no_soft_handoff_diff_analysis.txt",
    )
    args = parser.parse_args()

    v9 = _load_per_file(args.v9_csv, "neurite_macro_f1_stage23")
    nh = _load_per_file(args.no_handoff_csv, "neurite_macro_f1")
    common = sorted(set(v9) & set(nh))
    print(f"Loaded v9_final: {len(v9)} files; no_soft_handoff: {len(nh)} files")
    print(f"Common (joined on basename): {len(common)} files")

    # Compute per-file deltas.
    rows = []
    for fname in common:
        a = v9[fname]
        b = nh[fname]
        rows.append({
            "path": a["path"],
            "fname": fname,
            "cell_type": a["cell_type"],
            "source": a["source"],
            "n_nodes": a["n_nodes"],
            "v9_f1": a["f1"],
            "nh_f1": b["f1"],
            "diff": b["f1"] - a["f1"],
        })

    overall = _summarize_diffs([r["diff"] for r in rows])

    by_celltype: dict[str, list[float]] = defaultdict(list)
    by_source: dict[str, list[float]] = defaultdict(list)
    by_bucket: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        by_celltype[r["cell_type"]].append(r["diff"])
        by_source[r["source"]].append(r["diff"])
        by_bucket[(r["cell_type"], r["source"])].append(r["diff"])

    # Cells where soft_handoff was hurting the most (biggest positive diff
    # = no_soft_handoff rescued them).
    rescued = sorted(rows, key=lambda r: -r["diff"])[:15]
    # Cells where soft_handoff was helping (negative diff).
    regressed = sorted(rows, key=lambda r: r["diff"])[:15]

    lines = []
    lines.append("Where does removing soft-handoff actually help?")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Compares per-file neurite-macro-F1 of:")
    lines.append("  v9_final         = trunk + soft_handoff (current paper headline)")
    lines.append("  no_soft_handoff  = trunk, soft_handoff DISABLED")
    lines.append("Δ = no_soft_handoff F1 − v9_final F1.  Positive = rescued.")
    lines.append("")
    lines.append(f"OVERALL  ({len(rows)} files)")
    lines.append(f"  {_fmt_summary(overall)}")
    lines.append("")

    lines.append("By cell type:")
    for ct in sorted(by_celltype):
        s = _summarize_diffs(by_celltype[ct])
        lines.append(f"  {ct:<15s}  {_fmt_summary(s)}")
    lines.append("")

    lines.append("By source:")
    for src in sorted(by_source):
        s = _summarize_diffs(by_source[src])
        lines.append(f"  {src:<15s}  {_fmt_summary(s)}")
    lines.append("")

    lines.append("By cell type × source (sorted by mean Δ, biggest rescue first):")
    bucket_summaries = [(k, _summarize_diffs(v)) for k, v in by_bucket.items()]
    bucket_summaries.sort(key=lambda x: -x[1].get("mean_diff", 0.0))
    for (ct, src), s in bucket_summaries:
        bucket_label = f"{ct}/{src}"
        lines.append(f"  {bucket_label:<26s}  {_fmt_summary(s)}")
    lines.append("")

    lines.append("Top 15 cells RESCUED by removing soft_handoff (biggest +Δ):")
    lines.append(f"  {'cell_type':<13s}  {'source':<13s}  {'n_nodes':>8s}  "
                 f"{'v9_F1':>7s}  {'no_h_F1':>7s}  {'Δ':>8s}  fname")
    for r in rescued:
        if r["diff"] <= 1e-6:
            break
        lines.append(
            f"  {r['cell_type']:<13s}  {r['source']:<13s}  {r['n_nodes']:>8d}  "
            f"{r['v9_f1']:>7.4f}  {r['nh_f1']:>7.4f}  {r['diff']:>+8.4f}  {r['fname']}"
        )
    lines.append("")

    lines.append("Top 15 cells where soft_handoff WAS HELPING (biggest −Δ; "
                 "removing it hurt these):")
    lines.append(f"  {'cell_type':<13s}  {'source':<13s}  {'n_nodes':>8s}  "
                 f"{'v9_F1':>7s}  {'no_h_F1':>7s}  {'Δ':>8s}  fname")
    for r in regressed:
        if r["diff"] >= -1e-6:
            break
        lines.append(
            f"  {r['cell_type']:<13s}  {r['source']:<13s}  {r['n_nodes']:>8d}  "
            f"{r['v9_f1']:>7.4f}  {r['nh_f1']:>7.4f}  {r['diff']:>+8.4f}  {r['fname']}"
        )

    text = "\n".join(lines)
    args.out_text.write_text(text, encoding="utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print()
    print(text)
    print()
    print(f"Wrote {args.out_text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
