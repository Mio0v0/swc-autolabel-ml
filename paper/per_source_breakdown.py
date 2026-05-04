"""Per-source breakdown of per-file F1 scores.

Reviewers always ask: "does it actually generalize across data sources,
or is one source carrying the headline number?" The answer needs a
table that splits the held-out test set by source (Allen, lab,
neuromorpho, hpf_ca1) and reports per-source mean / median / P10 /
worst-cell-F1.

Source is inferred from filename prefix:

    allen__*       Allen Institute morphologies
    lab__*         Internal lab tracings
    neuromorpho__* NeuroMorpho.org submissions
    hpf_ca1__*     CA1 hippocampal pyramidals (v9 only)

Reads the per-file CSV produced by ``hybrid.evaluate`` (or any
``cross_dataset_eval`` output) and writes a JSON + plain-text table.

Usage::

    python -m paper.per_source_breakdown
    python -m paper.per_source_breakdown --csv path/to/per_file_scores.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "paper" / "results" / "snapshots" / "v9_final_subtree_gnn.csv"
DEFAULT_JSON_OUT = ROOT / "paper" / "results" / "per_source_breakdown.json"
DEFAULT_TXT_OUT = ROOT / "paper" / "results" / "per_source_breakdown.txt"


KNOWN_PREFIXES = ("allen__", "lab__", "neuromorpho__", "hpf_ca1__")


def _classify_source(name: str) -> str:
    base = name.lower()
    for pref in KNOWN_PREFIXES:
        if base.startswith(pref):
            return pref.rstrip("_")
    return "other"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = int(round((pct / 100.0) * (len(arr) - 1)))
    return arr[idx]


def _summarize(scores: list[float]) -> dict:
    if not scores:
        return {
            "n": 0, "mean": 0.0, "median": 0.0,
            "p10": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0,
            "min": 0.0, "max": 0.0,
        }
    s = sorted(scores)
    return {
        "n": len(s),
        "mean": round(statistics.fmean(s), 4),
        "median": round(s[len(s) // 2], 4),
        "p10": round(_percentile(s, 10), 4),
        "p25": round(_percentile(s, 25), 4),
        "p75": round(_percentile(s, 75), 4),
        "p90": round(_percentile(s, 90), 4),
        "min": round(s[0], 4),
        "max": round(s[-1], 4),
    }


def breakdown(csv_path: Path) -> dict:
    """Return ``{source: {"by_cell_type": {ct: stats}}, "overall": stats}``."""
    by_cell_type_and_source: dict[tuple[str, str], list[float]] = {}
    by_source: dict[str, list[float]] = {}
    by_cell_type: dict[str, list[float]] = {}
    by_source_per_class: dict[str, dict[str, list[float]]] = {}
    overall: list[float] = []

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        # Auto-detect column name for the headline F1 we want to summarize.
        f1_col = (
            "neurite_macro_f1_stage23"
            if "neurite_macro_f1_stage23" in reader.fieldnames
            else ("neurite_macro_f1" if "neurite_macro_f1" in reader.fieldnames
                  else None)
        )
        if f1_col is None:
            raise KeyError(
                f"Could not find a per-file F1 column in {csv_path}. "
                f"Available: {reader.fieldnames}"
            )
        ct_col = "cell_type" if "cell_type" in reader.fieldnames else None
        path_col = (
            "path" if "path" in reader.fieldnames
            else ("file" if "file" in reader.fieldnames else None)
        )
        if path_col is None:
            raise KeyError(
                f"Could not find a path/file column in {csv_path}. "
                f"Available: {reader.fieldnames}"
            )

        for row in reader:
            full_path = row[path_col]
            name = Path(full_path).name
            source = _classify_source(name)
            try:
                f1 = float(row[f1_col])
            except (ValueError, TypeError):
                continue
            ct = row[ct_col] if ct_col else "unknown"
            overall.append(f1)
            by_source.setdefault(source, []).append(f1)
            by_cell_type.setdefault(ct, []).append(f1)
            by_cell_type_and_source.setdefault((ct, source), []).append(f1)

    out = {
        "csv_path": str(csv_path),
        "f1_column": f1_col,
        "overall": _summarize(overall),
        "by_source": {s: _summarize(v) for s, v in sorted(by_source.items())},
        "by_cell_type": {ct: _summarize(v) for ct, v in sorted(by_cell_type.items())},
        "by_cell_type_and_source": {
            f"{ct}/{s}": _summarize(v)
            for (ct, s), v in sorted(by_cell_type_and_source.items())
        },
    }
    return out


def format_table(data: dict) -> str:
    lines: list[str] = []
    lines.append(f"Per-source breakdown of per-file F1")
    lines.append(f"Source CSV: {data['csv_path']}")
    lines.append(f"F1 column:  {data['f1_column']}")
    lines.append("")
    fmt = "  {:<24s} {:>5s}  {:>7s}  {:>7s}  {:>7s}  {:>7s}  {:>7s}"
    lines.append(fmt.format("group", "n", "mean", "median", "p10", "p25", "p90"))
    lines.append("  " + "-" * 70)

    def _row(label: str, s: dict) -> str:
        return fmt.format(
            label,
            str(s["n"]),
            f"{s['mean']:.4f}", f"{s['median']:.4f}",
            f"{s['p10']:.4f}", f"{s['p25']:.4f}", f"{s['p90']:.4f}",
        )

    lines.append("Overall:")
    lines.append(_row("all", data["overall"]))
    lines.append("")
    lines.append("By data source:")
    for s, stats in data["by_source"].items():
        lines.append(_row(s, stats))
    lines.append("")
    lines.append("By cell type:")
    for ct, stats in data["by_cell_type"].items():
        lines.append(_row(ct, stats))
    lines.append("")
    lines.append("By cell type + source:")
    for label, stats in data["by_cell_type_and_source"].items():
        lines.append(_row(label, stats))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_JSON_OUT)
    ap.add_argument("--out-txt", type=Path, default=DEFAULT_TXT_OUT)
    args = ap.parse_args()

    if not args.csv.is_file():
        raise FileNotFoundError(f"per-file CSV not found: {args.csv}")

    data = breakdown(args.csv)
    table = format_table(data)
    print(table)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    args.out_txt.write_text(table, encoding="utf-8")
    print(f"\nSaved {args.out_json}")
    print(f"Saved {args.out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
