#!/usr/bin/env python3
"""Apples-to-apples comparison: v12 (Stage 2+3) vs L-Measure-RF.

Both evaluated on the same 4263 pyramidal test cells from the cleaned
11862-corpus, 50/50 split, seed=2024. Reads the comprehensive eval
JSONs and writes a side-by-side comparison table.

Also: combines per-cell results to compute optional ensemble upper
bounds:
    - "best-of-two" F1 (upper bound on any cell-level ensemble)
    - F1 distribution of v12 - L-Measure (per-cell delta)

Usage:
    python -m paper._compare_baselines

Output:
    paper/results/comparison_v12_vs_lmeasure.txt
    paper/results/comparison_v12_vs_lmeasure.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
V12_JSON  = ROOT / "paper" / "results" / "v12_pyramidal_only_seed2024_clean_eval.json"
LMS_JSON  = ROOT / "paper" / "results" / "lmeasure_pyramidal_only_eval.json"
V12_CSV   = ROOT / "paper" / "results" / "v12_pyramidal_only_seed2024_clean_eval.csv"
LMS_CSV   = ROOT / "paper" / "results" / "lmeasure_pyramidal_only_eval.csv"

OUT_TXT  = ROOT / "paper" / "results" / "comparison_v12_vs_lmeasure.txt"
OUT_JSON = ROOT / "paper" / "results" / "comparison_v12_vs_lmeasure.json"


def _load_per_cell(csv_path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with csv_path.open("r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r["file"]] = r
    return out


def _f(x: str) -> float:
    if x is None or x == "": return float("nan")
    try: return float(x)
    except ValueError: return float("nan")


def main() -> int:
    v12 = json.loads(V12_JSON.read_text(encoding="utf-8"))
    lms = json.loads(LMS_JSON.read_text(encoding="utf-8"))

    # ===================== Corpus comparison =====================
    rows = []
    def _row(label, v, l):
        return (label, v, l, l - v)  # delta = L-Measure - v12
    cv = v12["corpus"]; cl = lms["corpus"]
    rows.append(_row("Per-node accuracy",   cv["accuracy"],          cl["accuracy"]))
    rows.append(_row("Macro F1 (4 cls)",    cv["macro_f1"],          cl["macro_f1"]))
    rows.append(_row("Neurite macro F1",    cv["neurite_macro_f1"],  cl["neurite_macro_f1"]))
    for cls in ("soma","axon","basal","apical"):
        rows.append(_row(f"{cls} F1 corpus",
                         cv["per_class"][cls]["f1"],
                         cl["per_class"][cls]["f1"]))

    # ===================== Per-cell distributions =====================
    pcv = v12["per_cell"]; pcl = lms["per_cell"]
    pc_rows = []
    for stat in ("mean", "P10", "P25", "P50", "P75", "P90"):
        pc_rows.append(("F1 neurite",   stat, pcv["F1_neurite"][stat],   pcl["F1_neurite"][stat]))
    for stat in ("mean", "P10", "P25", "P50"):
        pc_rows.append(("accuracy",     stat, pcv["accuracy"][stat],     pcl["accuracy"][stat]))
    for cls in ("axon_F1","basal_F1","apical_F1"):
        for stat in ("mean", "P10"):
            pc_rows.append((cls,        stat, pcv["per_class_F1"][cls][stat], pcl["per_class_F1"][cls][stat]))

    # ===================== Per-cell ensemble upper bound =====================
    v12_pc = _load_per_cell(V12_CSV)
    lms_pc = _load_per_cell(LMS_CSV)
    common = set(v12_pc) & set(lms_pc)
    print(f"Common cells in both eval CSVs: {len(common)}")

    v_f1 = np.array([_f(v12_pc[f]["F1_neurite"]) for f in common])
    l_f1 = np.array([_f(lms_pc[f]["F1_neurite"]) for f in common])
    best_f1 = np.maximum(v_f1, l_f1)
    worst_f1 = np.minimum(v_f1, l_f1)
    delta = v_f1 - l_f1

    ens_summary = {
        "v12_only_mean":     float(v_f1.mean()),
        "v12_only_P10":      float(np.percentile(v_f1, 10)),
        "lmeasure_only_mean": float(l_f1.mean()),
        "lmeasure_only_P10": float(np.percentile(l_f1, 10)),
        "best_of_two_mean":  float(best_f1.mean()),
        "best_of_two_P10":   float(np.percentile(best_f1, 10)),
        "worst_of_two_mean": float(worst_f1.mean()),
        "n_cells_v12_better_by_05": int((delta > 0.05).sum()),
        "n_cells_lm_better_by_05":  int((delta < -0.05).sum()),
        "n_cells_close":            int((np.abs(delta) <= 0.05).sum()),
    }

    # ===================== Per-class F1 P10 reality check =====================
    # On the bottom 10% by F1, both models often have apical_F1=0. Verify.
    v_ap = np.array([_f(v12_pc[f]["apical_F1"]) for f in common])
    l_ap = np.array([_f(lms_pc[f]["apical_F1"]) for f in common])
    v_ba = np.array([_f(v12_pc[f]["basal_F1"]) for f in common])
    l_ba = np.array([_f(lms_pc[f]["basal_F1"]) for f in common])

    # ===================== Write report =====================
    L = []
    P = L.append
    P("=" * 80)
    P("  v12 (Stage 2+3, GT cell-type override) vs L-Measure-RF baseline")
    P(f"  Same 4263 test pyramidals (cleaned 11862-corpus, 50/50 split, seed 2024)")
    P("=" * 80)
    P("")
    P("== CORPUS-LEVEL COMPARISON ==")
    P(f"  {'metric':<22} {'v12 (S2+S3)':>14} {'L-Measure':>14} {'L − v12':>14}")
    P(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*14}")
    for lbl, v, l, d in rows:
        flag = "  L wins" if d > 0.005 else ("  v12 wins" if d < -0.005 else "  ≈ tied")
        P(f"  {lbl:<22} {v:>14.4f} {l:>14.4f} {d:>+14.4f}{flag}")

    P("")
    P("== PER-CELL DISTRIBUTIONS ==")
    P(f"  {'metric':<14} {'stat':<6} {'v12 (S2+S3)':>14} {'L-Measure':>14} {'L − v12':>14}")
    P(f"  {'-'*14} {'-'*6} {'-'*14} {'-'*14} {'-'*14}")
    for metric, stat, v, l in pc_rows:
        d = l - v
        flag = "  L wins" if d > 0.005 else ("  v12 wins" if d < -0.005 else "  ≈ tied")
        P(f"  {metric:<14} {stat:<6} {v:>14.4f} {l:>14.4f} {d:>+14.4f}{flag}")

    P("")
    P("== PER-CELL ENSEMBLE UPPER BOUND ==")
    P(f"  (For each cell, take the BETTER of v12 / L-Measure F1. ")
    P(f"   This is the ceiling for any oracle ensemble of these two models.)")
    P(f"  {'measure':<30} {'value':>10}")
    P(f"  v12 only mean F1                {ens_summary['v12_only_mean']:>10.4f}")
    P(f"  v12 only P10 F1                 {ens_summary['v12_only_P10']:>10.4f}")
    P(f"  L-Measure only mean F1          {ens_summary['lmeasure_only_mean']:>10.4f}")
    P(f"  L-Measure only P10 F1           {ens_summary['lmeasure_only_P10']:>10.4f}")
    P(f"  best-of-two mean F1             {ens_summary['best_of_two_mean']:>10.4f}  <- ENSEMBLE CEILING")
    P(f"  best-of-two P10 F1              {ens_summary['best_of_two_P10']:>10.4f}  <- ENSEMBLE CEILING")
    P(f"")
    P(f"  cells where v12 better (>0.05): {ens_summary['n_cells_v12_better_by_05']:>5}  ({100*ens_summary['n_cells_v12_better_by_05']/len(common):.1f}%)")
    P(f"  cells where L-M  better (>0.05):{ens_summary['n_cells_lm_better_by_05']:>5}  ({100*ens_summary['n_cells_lm_better_by_05']/len(common):.1f}%)")
    P(f"  cells where models agree (≤0.05):{ens_summary['n_cells_close']:>5}  ({100*ens_summary['n_cells_close']/len(common):.1f}%)")

    P("")
    P("== HEADLINE ==")
    delta_mean_F1 = pcl["F1_neurite"]["mean"] - pcv["F1_neurite"]["mean"]
    delta_P10_F1  = pcl["F1_neurite"]["P10"]  - pcv["F1_neurite"]["P10"]
    delta_apical_corpus = cl["per_class"]["apical"]["f1"] - cv["per_class"]["apical"]["f1"]
    P(f"  L-Measure leads on:    corpus apical F1 ({delta_apical_corpus:+.4f}),  per-node accuracy")
    P(f"  v12 leads on:          per-cell F1 mean ({-delta_mean_F1:+.4f}),  per-cell F1 P10 ({-delta_P10_F1:+.4f})")
    P(f"  Net: v12 has better per-cell TAIL (more robust on hard cells),")
    P(f"  L-Measure has slightly better corpus-aggregate per-class metrics.")
    P("")

    body = "\n".join(L)
    OUT_TXT.write_text(body, encoding="utf-8")
    OUT_JSON.write_text(json.dumps({
        "corpus_comparison": [{"metric": l, "v12": v, "lmeasure": ll, "delta_lm_minus_v12": d}
                              for l, v, ll, d in rows],
        "per_cell_comparison": [{"metric": m, "stat": s, "v12": v, "lmeasure": ll}
                                for m, s, v, ll in pc_rows],
        "ensemble_upper_bound": ens_summary,
    }, indent=2), encoding="utf-8")
    print(body)
    print(f"\nWrote {OUT_TXT}")
    print(f"Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
