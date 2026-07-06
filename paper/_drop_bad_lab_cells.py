#!/usr/bin/env python3
"""Drop neuromorpho cells from high-failure-rate labs IN-PLACE.

Uses paper/results/heldout_per_cell_f1.csv to identify lab prefixes
where the held-out failure rate is high (>=THRESH_RATE) on a meaningful
sample (>=MIN_SAMPLE). All cells from those labs are dropped from
paper/results/corpus_qc_v12_uncurated.csv (in place; SWC files are
NOT deleted on disk).

Runs in seconds. No model inference. No new file outputs beyond the
in-place CSV update + a summary JSON of which labs were dropped and
why.

Lab prefix extraction:
    Strip "neuromorpho__" prefix, take the leading alphabetic chunk
    (with light digit-then-alpha continuation), e.g.:
        neuromorpho__ofc_2017_Cell_03.swc       -> "ofc"
        neuromorpho__roc-3-N4-cell1.swc         -> "roc"
        neuromorpho__C1042-whole-2.swc          -> "C"  (1-char, expanded -> "C1042")

Output:
    paper/results/corpus_qc_v12_uncurated.csv         (overwritten)
    paper/results/dropped_labs_summary.json           summary

Usage:
    python -m paper._drop_bad_lab_cells
        [--rate 0.40]       # drop labs with >= this failure rate
        [--min-sample 3]    # require >= this many held-out cells per lab
        [--dry-run]         # print which labs would be dropped, don't write
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QC_CSV     = ROOT / "paper" / "results" / "corpus_qc_v12_uncurated.csv"
HELDOUT_CSV = ROOT / "paper" / "results" / "heldout_per_cell_f1.csv"
SUMMARY_JSON = ROOT / "paper" / "results" / "dropped_labs_summary.json"


def lab_prefix(filename: str) -> str:
    name = filename.replace(".swc", "")
    if "__" in name:
        _, name = name.split("__", 1)
    m = re.match(r"^([A-Za-z]+)", name)
    if m:
        prefix = m.group(1)
        if len(prefix) <= 2:
            m2 = re.match(r"^[A-Za-z]+[0-9]*[A-Za-z]*", name)
            if m2: prefix = m2.group(0)
        return prefix
    return name[:8]


def source(filename: str) -> str:
    return filename.split("__", 1)[0] if "__" in filename else "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rate",       type=float, default=0.40,
                    help="Drop labs whose held-out failure rate >= this (default 0.40)")
    ap.add_argument("--min-sample", type=int,   default=3,
                    help="Require this many held-out cells in the lab (default 3)")
    ap.add_argument("--bottom-pct", type=float, default=10.0,
                    help="Failure = in bottom this percent by F1 (default 10)")
    ap.add_argument("--dry-run",    action="store_true",
                    help="Don't actually overwrite the QC CSV")
    args = ap.parse_args()

    if not QC_CSV.is_file():
        raise SystemExit(f"MISSING: {QC_CSV}")
    if not HELDOUT_CSV.is_file():
        raise SystemExit(f"MISSING: {HELDOUT_CSV}")

    # 1) Compute held-out failure rate per lab
    rows = []
    with HELDOUT_CSV.open("r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    rows.sort(key=lambda r: float(r["held_out_F1"]))
    n_bottom = max(1, int(len(rows) * args.bottom_pct / 100))
    bottom_files = {r["file"] for r in rows[:n_bottom]}

    prefix_holdout_n   = Counter()
    prefix_holdout_bad = Counter()
    for r in rows:
        if source(r["file"]) != "neuromorpho":
            continue
        p = lab_prefix(r["file"])
        prefix_holdout_n[p] += 1
        if r["file"] in bottom_files:
            prefix_holdout_bad[p] += 1

    # 2) Build the bad-lab set (>=rate AND >=min_sample)
    bad_labs: dict[str, dict] = {}
    for prefix, n in prefix_holdout_n.items():
        if n < args.min_sample:
            continue
        bad = prefix_holdout_bad.get(prefix, 0)
        rate = bad / n
        if rate >= args.rate:
            bad_labs[prefix] = {
                "holdout_n":    n,
                "holdout_bad":  bad,
                "failure_rate": round(rate, 4),
            }
    print(f"Bad labs identified (rate >= {args.rate*100:.0f}%, sample >= {args.min_sample}):")
    print(f"  {len(bad_labs)} labs")
    for p, info in sorted(bad_labs.items(), key=lambda kv: -kv[1]["failure_rate"]):
        print(f"    {p:<25}  holdout {info['holdout_bad']}/{info['holdout_n']} "
              f"= {info['failure_rate']*100:.1f}%")

    # 3) Scan current QC CSV; mark rows from bad labs for drop
    rows_in_total = 0
    rows_in_qcpass = 0
    rows_out: list[dict] = []
    dropped_by_lab: Counter = Counter()
    dropped_by_celltype: Counter = Counter()
    with QC_CSV.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        for r in reader:
            rows_in_total += 1
            qc_pass = r.get("qc_pass", "").strip().lower() in ("true", "1")
            if qc_pass: rows_in_qcpass += 1
            fname = Path(r["path"]).name
            if qc_pass and source(fname) == "neuromorpho":
                p = lab_prefix(fname)
                if p in bad_labs:
                    dropped_by_lab[p] += 1
                    dropped_by_celltype[r.get("cell_type", "?")] += 1
                    continue
            rows_out.append(r)

    n_dropped = sum(dropped_by_lab.values())
    print(f"\nWill drop {n_dropped} cells:")
    print(f"  by cell_type: {dict(dropped_by_celltype)}")
    print(f"  top 10 by lab:")
    for p, n in dropped_by_lab.most_common(10):
        print(f"    {p:<25}  {n} cells")

    if args.dry_run:
        print("\n[dry-run] skipping write.")
        return 0

    # 4) Overwrite in place
    with QC_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows_out: w.writerow(r)

    # 5) Summary JSON
    SUMMARY_JSON.write_text(json.dumps({
        "criteria": {
            "min_failure_rate": args.rate,
            "min_holdout_sample": args.min_sample,
            "bottom_pct_for_failure_def": args.bottom_pct,
        },
        "n_labs_dropped":           len(bad_labs),
        "n_cells_dropped":          n_dropped,
        "dropped_by_celltype":      dict(dropped_by_celltype),
        "dropped_by_lab":           {p: dropped_by_lab[p] for p in sorted(bad_labs.keys())},
        "bad_labs_with_rates":      bad_labs,
        "rows_in_total_before":     rows_in_total,
        "rows_in_qcpass_before":    rows_in_qcpass,
        "rows_in_total_after":      len(rows_out),
        "rows_in_qcpass_after":     rows_in_qcpass - n_dropped,
    }, indent=2), encoding="utf-8")
    print(f"\nOverwrote {QC_CSV}")
    print(f"Summary -> {SUMMARY_JSON}")
    print(f"  QC-passed before: {rows_in_qcpass}")
    print(f"  QC-passed after : {rows_in_qcpass - n_dropped}  (-{n_dropped})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
