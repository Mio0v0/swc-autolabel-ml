#!/usr/bin/env python3
"""T4: failure-mode pattern analysis on T3's held-out F1 results.

Reads paper/results/heldout_per_cell_f1.csv and categorizes each cell
(especially the bottom percentile) into deterministic failure modes
defined by structural + prediction signals.

Categories are NOT mutually exclusive -- one cell can match multiple.
Each category has a clear detection rule that can be re-applied
elsewhere (e.g. to flag GT errors in cells we never evaluated honestly).

Usage:
    python -m paper._analyze_failure_modes
        [--bottom-pct 10]     # default: analyze bottom 10% by F1
        [--in PATH]           # input CSV (default: heldout_per_cell_f1.csv)

Output:
    paper/results/gt_failure_modes.md       human-readable taxonomy + examples
    paper/results/gt_failure_modes.csv      per-cell category labels (multi-category)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IN_CSV  = ROOT / "paper" / "results" / "heldout_per_cell_f1.csv"
OUT_MD  = ROOT / "paper" / "results" / "gt_failure_modes.md"
OUT_CSV = ROOT / "paper" / "results" / "gt_failure_modes.csv"


# ---------------------- Category detection rules ----------------------

def _f(x: str) -> float:
    if x == "" or x is None: return float("nan")
    try: return float(x)
    except ValueError: return float("nan")


def _i(x: str) -> int:
    try: return int(x)
    except (ValueError, TypeError): return 0


CATEGORIES = [
    # (name, description, rule_fn(row_dict) -> bool)

    ("WHOLE_CELL_MISLABEL",
     "Stage 1 confidently disagrees with GT cell type. Whole-cell label looks wrong.",
     lambda r: r["stage1_correct"] == "False" and _f(r["stage1_conf"]) > 0.85),

    ("APICAL_MISSED",
     "GT has apical (>1% of nodes) but model predicted ZERO apical nodes.",
     lambda r: _f(r["apical_frac"]) > 0.01 and _i(r["pred_apical"]) == 0),

    ("APICAL_UNDERPREDICTED",
     "Model predicted <50% as many apical nodes as GT has (and apical_F1 low).",
     lambda r: (_f(r["apical_frac"]) > 0.01 and
                _i(r["pred_apical"]) < 0.5 * _f(r["apical_frac"]) * _i(r["n_nodes"]) and
                _f(r["apical_F1"]) < 0.5)),

    ("APICAL_OVERPREDICTED",
     "Model predicted >1.5x as many apical nodes as GT has (and apical_F1 low).",
     lambda r: (_i(r["pred_apical"]) > 1.5 * max(1, _f(r["apical_frac"]) * _i(r["n_nodes"])) and
                _f(r["apical_F1"]) < 0.5 and
                _i(r["pred_apical"]) > 100)),

    ("APICAL_INVERTED",
     "GT apical nodes are on average BELOW the soma (typically reversed orientation).",
     lambda r: not np.isnan(_f(r["apical_above_soma"])) and _f(r["apical_above_soma"]) < 0),

    ("AXON_DENDRITE_CONFUSION",
     "Axon F1 < 0.95 (very rare; axon usually trivially separable from dendrite).",
     lambda r: not np.isnan(_f(r["axon_F1"])) and _f(r["axon_F1"]) < 0.95),

    ("BASAL_LOST",
     "Basal F1 < 0.5 despite GT having basal nodes -- model labeled basal as something else.",
     lambda r: not np.isnan(_f(r["basal_F1"])) and _f(r["basal_F1"]) < 0.5 and _f(r["basal_frac"]) > 0.01),

    ("DISCONNECTED",
     "Reconstruction has >1 connected component (broken trace).",
     lambda r: _i(r["n_components"]) > 1),

    ("TRUNCATED_APICAL",
     "GT has apical but very short z-extent (<50 microns).",
     lambda r: 0.0 < _f(r["apical_z_extent"]) < 50.0 and _f(r["apical_frac"]) > 0.01),

    ("STAGE1_PROPAGATION",
     "Stage 1 was wrong AND per-cell F1 is low -- error propagated downstream.",
     lambda r: r["stage1_correct"] == "False" and _f(r["held_out_F1"]) < 0.7),
]


def _matching_categories(row: dict) -> list[str]:
    return [name for name, _, rule in CATEGORIES if rule(row)]


# ---------------------- Reporting ----------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bottom-pct", type=float, default=10.0,
                    help="Analyze bottom X percent of cells by F1 (default 10)")
    ap.add_argument("--in", dest="in_path", type=Path, default=IN_CSV)
    ap.add_argument("--top-examples", type=int, default=15,
                    help="Number of example file paths to show per category")
    args = ap.parse_args()

    if not args.in_path.is_file():
        raise SystemExit(f"MISSING: {args.in_path}")

    rows: list[dict] = []
    with args.in_path.open("r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    print(f"Loaded {len(rows)} per-cell rows from {args.in_path}")

    # Sort by held_out_F1 ascending; bottom N
    rows.sort(key=lambda r: _f(r["held_out_F1"]))
    threshold_idx = max(1, int(len(rows) * args.bottom_pct / 100))
    bottom = rows[:threshold_idx]
    bottom_threshold_f1 = _f(bottom[-1]["held_out_F1"])
    print(f"Bottom {args.bottom_pct:.0f}% = {len(bottom)} cells "
          f"(F1 < {bottom_threshold_f1:.4f})")

    # Categorize: every cell gets a list of matching categories
    bottom_cats_by_file: dict[str, list[str]] = {}
    cat_counts_bottom = Counter()
    cat_counts_pyr_bottom = Counter()
    cat_counts_int_bottom = Counter()
    for r in bottom:
        cats = _matching_categories(r)
        bottom_cats_by_file[r["file"]] = cats
        for c in cats:
            cat_counts_bottom[c] += 1
            if r["cell_type_gt"] == "pyramidal":   cat_counts_pyr_bottom[c] += 1
            elif r["cell_type_gt"] == "interneuron": cat_counts_int_bottom[c] += 1

    # Also categorize the FULL corpus (so we can compare "did the rule
    # also fire on many top-90% cells?" -- a category that fires
    # equally on top and bottom isn't a discriminative failure-mode).
    cat_counts_all = Counter()
    all_cats_by_file: dict[str, list[str]] = {}
    for r in rows:
        cats = _matching_categories(r)
        all_cats_by_file[r["file"]] = cats
        for c in cats: cat_counts_all[c] += 1

    # ---- CSV: per-cell category labels ----
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "cell_type_gt", "held_out_F1", "in_bottom",
                    "n_categories", "categories"])
        bottom_files = {r["file"] for r in bottom}
        for r in rows:
            cats = all_cats_by_file[r["file"]]
            w.writerow([
                r["file"], r["cell_type_gt"], r["held_out_F1"],
                "yes" if r["file"] in bottom_files else "no",
                len(cats),
                "|".join(cats),
            ])
    print(f"Wrote {OUT_CSV}")

    # ---- Markdown report ----
    lines: list[str] = []
    P = lines.append
    P(f"# GT failure-mode taxonomy")
    P(f"")
    P(f"**Input**: `{args.in_path.relative_to(ROOT)}` ({len(rows)} held-out cells)")
    P(f"**Bottom slice**: lowest {args.bottom_pct:.0f}% by per-cell F1 = "
      f"**{len(bottom)} cells** (F1 < {bottom_threshold_f1:.4f})")
    P(f"")
    # Cell-type split of bottom
    n_pyr_bottom = sum(1 for r in bottom if r["cell_type_gt"] == "pyramidal")
    n_int_bottom = sum(1 for r in bottom if r["cell_type_gt"] == "interneuron")
    P(f"**By cell type in bottom slice**: {n_pyr_bottom} pyramidals, {n_int_bottom} interneurons")
    P(f"")

    P(f"## Category summary")
    P(f"")
    P(f"| Category | Bottom-{args.bottom_pct:.0f}% count (pyr / int) | Full-corpus count | Bottom enrichment |")
    P(f"|---|---|---|---|")
    enriched: list[tuple[str, float, int, int, int]] = []
    for name, desc, _rule in CATEGORIES:
        b = cat_counts_bottom.get(name, 0)
        a = cat_counts_all.get(name, 0)
        bp = cat_counts_pyr_bottom.get(name, 0)
        bi = cat_counts_int_bottom.get(name, 0)
        # Enrichment = (bottom %) / (corpus %)
        bottom_rate = b / max(1, len(bottom))
        corpus_rate = a / max(1, len(rows))
        enrichment = bottom_rate / corpus_rate if corpus_rate > 0 else float("inf")
        enriched.append((name, enrichment, b, a, bp))
        P(f"| **{name}** | {b}  ({bp} pyr / {bi} int) | {a} | "
          f"{enrichment:.1f}x" + (" :rocket:" if enrichment >= 2.0 else "") + " |")
    P(f"")
    P(f"_Enrichment = (% of bottom slice in this category) / (% of all cells in this category). "
      f"Values > ~2x are reliable failure-mode signals; 1x means the rule fires equally on bottom and top._")
    P(f"")

    # Top categories by enrichment
    enriched.sort(key=lambda x: -x[1])
    P(f"## Categories ranked by enrichment in bottom slice")
    P(f"")
    for name, enr, b_n, _a_n, b_pyr in enriched:
        if b_n == 0: continue
        P(f"### {name}  (enrichment = {enr:.1f}x, n = {b_n} bottom cells, "
          f"{b_pyr} pyramidals)")
        desc = next(d for n, d, _ in CATEGORIES if n == name)
        P(f"")
        P(f"**Rule**: {desc}")
        P(f"")
        # Sample cells matching this category, sorted by lowest F1
        matching = [r for r in bottom if name in bottom_cats_by_file[r["file"]]]
        if matching:
            P(f"**Examples** (worst {min(args.top_examples, len(matching))} of {len(matching)}):")
            P(f"")
            P(f"| file | cell_type | F1 | apical_F1 | basal_F1 | n_nodes | pred_a/b/p | other categories |")
            P(f"|---|---|---|---|---|---|---|---|")
            for r in matching[: args.top_examples]:
                other = [c for c in bottom_cats_by_file[r["file"]] if c != name]
                P(f"| `{r['file']}` | {r['cell_type_gt']} | "
                  f"{r['held_out_F1']} | {r['apical_F1'] or '-'} | {r['basal_F1'] or '-'} | "
                  f"{r['n_nodes']} | "
                  f"{r['pred_axon']}/{r['pred_basal']}/{r['pred_apical']} | "
                  f"{', '.join(other) if other else '-'} |")
            P(f"")

    # Cells matching NO category
    n_unmatched = sum(1 for r in bottom if not bottom_cats_by_file[r["file"]])
    P(f"## Unmatched bottom-slice cells")
    P(f"")
    P(f"**{n_unmatched} bottom-slice cells matched NO category** -- low F1 but no obvious failure pattern. "
      f"These may need new categories or are genuinely hard cells.")
    P(f"")
    if n_unmatched > 0:
        unmatched = [r for r in bottom if not bottom_cats_by_file[r["file"]]]
        P(f"Worst {min(10, len(unmatched))} unmatched:")
        P(f"")
        P(f"| file | cell_type | F1 | apical_F1 | basal_F1 | apical_frac | apical_above_soma |")
        P(f"|---|---|---|---|---|---|---|")
        for r in unmatched[:10]:
            P(f"| `{r['file']}` | {r['cell_type_gt']} | {r['held_out_F1']} | "
              f"{r['apical_F1'] or '-'} | {r['basal_F1'] or '-'} | "
              f"{r['apical_frac']} | {r['apical_above_soma'] or '-'} |")
        P(f"")

    # Suggested cleaning vs modeling
    P(f"## Suggested actions per category")
    P(f"")
    P(f"| Category | Likely action |")
    P(f"|---|---|")
    P(f"| WHOLE_CELL_MISLABEL | **CLEAN** -- drop or relabel. Already caught by T2 mostly. |")
    P(f"| APICAL_INVERTED | **CLEAN** -- almost certainly a coordinate/orientation GT error. |")
    P(f"| DISCONNECTED | **CLEAN** -- reconstruction is broken; drop. |")
    P(f"| TRUNCATED_APICAL | **INSPECT** -- legitimate partial reconstructions; don't drop. Maybe down-weight. |")
    P(f"| APICAL_MISSED | **MODEL** -- need stronger apical detector (partition asymmetry, contraction). |")
    P(f"| APICAL_UNDERPREDICTED | **MODEL** -- same as above. |")
    P(f"| APICAL_OVERPREDICTED | **MODEL** -- decision boundary is permissive; needs counter-features. |")
    P(f"| BASAL_LOST | **MODEL** -- usually means basal got mistaken for apical. Same model fix. |")
    P(f"| AXON_DENDRITE_CONFUSION | **MODEL** -- rare; usually small/atypical cells. Inspect. |")
    P(f"| STAGE1_PROPAGATION | **MODEL** (Stage 1) -- improve Stage 1 OR enable soft-handoff threshold tuning. |")
    P(f"")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")

    # Console summary
    print()
    print("=== Category counts in bottom slice (sorted by enrichment) ===")
    for name, enr, b_n, a_n, b_pyr in enriched:
        if b_n == 0: continue
        print(f"  {name:<26}  bottom={b_n:>4}  full={a_n:>5}  enrichment={enr:>5.1f}x  ({b_pyr} pyr)")
    print()
    print(f"Unmatched bottom cells: {n_unmatched}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
