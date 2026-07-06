#!/usr/bin/env python3
"""Remove stale result rows whose files are no longer QC-passed.

Several older analysis CSVs predate the current bad-GT / bad-lab cleanup. This
utility makes those stale cells disappear from downstream flag-model inputs by
filtering every result CSV with a ``file`` column against the current
``corpus_qc_v12_uncurated.csv`` QC-pass set.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
QC_CSV = ROOT / "paper" / "results" / "corpus_qc_v12_uncurated.csv"
RESULTS_DIR = ROOT / "paper" / "results"
OUT_JSON = RESULTS_DIR / "purged_non_qc_rows_summary.json"


def _load_qc_pass_basenames(path: Path) -> set[str]:
    qc = pd.read_csv(path)
    if "path" not in qc.columns or "qc_pass" not in qc.columns:
        raise SystemExit(f"{path} must contain path and qc_pass columns")
    mask = qc["qc_pass"].astype(str).str.lower().isin({"true", "1"})
    return {Path(str(p)).name for p in qc.loc[mask, "path"]}


def _candidate_csvs(results_dir: Path) -> list[Path]:
    return sorted(p for p in results_dir.glob("*.csv") if p.name != QC_CSV.name)


def _filter_csv(path: Path, keep_files: set[str], dry_run: bool) -> dict | None:
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception as exc:
        return {"file": str(path), "error": str(exc)}
    if "file" not in header.columns:
        return None

    df = pd.read_csv(path)
    before = len(df)
    keep_mask = df["file"].astype(str).map(lambda x: Path(x).name in keep_files)
    stale_rows = int((~keep_mask).sum())
    if stale_rows == 0:
        return {
            "file": str(path.relative_to(ROOT)),
            "rows_before": before,
            "rows_after": before,
            "rows_removed": 0,
            "unique_files_removed": 0,
        }

    stale_unique = int(df.loc[~keep_mask, "file"].astype(str).nunique())
    if not dry_run:
        df.loc[keep_mask].to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    return {
        "file": str(path.relative_to(ROOT)),
        "rows_before": before,
        "rows_after": int(keep_mask.sum()),
        "rows_removed": stale_rows,
        "unique_files_removed": stale_unique,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qc-csv", type=Path, default=QC_CSV)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    keep_files = _load_qc_pass_basenames(args.qc_csv)
    rows: list[dict] = []
    for path in _candidate_csvs(args.results_dir):
        res = _filter_csv(path, keep_files, args.dry_run)
        if res is not None:
            rows.append(res)

    removed = [r for r in rows if r.get("rows_removed", 0) > 0]
    summary = {
        "qc_csv": str(args.qc_csv.resolve().relative_to(ROOT)),
        "n_qc_pass_files": len(keep_files),
        "dry_run": bool(args.dry_run),
        "n_csvs_with_file_column": len(rows),
        "n_csvs_changed": len(removed),
        "total_rows_removed": int(sum(r.get("rows_removed", 0) for r in rows)),
        "files": rows,
    }
    if not args.dry_run:
        OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"QC-pass files: {len(keep_files)}")
    print(f"CSV files with file column: {len(rows)}")
    print(f"Changed files: {len(removed)}")
    print(f"Rows removed: {summary['total_rows_removed']}")
    for r in removed:
        print(
            f"  {r['file']}: -{r['rows_removed']} rows "
            f"({r['unique_files_removed']} unique files)"
        )
    if not args.dry_run:
        print(f"Wrote {OUT_JSON.resolve().relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
