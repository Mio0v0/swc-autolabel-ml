#!/usr/bin/env python3
"""Wave A1: Rejection-based quality flagging sweep.

For the v12 baseline eval CSV, sweep two rejection signals and
report what happens to metrics when we exclude the flagged cells:

    ORACLE rejection (upper bound on any rejection scheme):
        sort cells by GT F1 ascending; reject the bottom X% by F1.
        Lower bound on best-case improvement from any rejection method.

    REAL proxy (no GT used):
        Per-cell suspicion score from prediction-side signals only:
            - n_nodes (small cells flagged)
            - pred apical fraction (pyramidal with 0% pred apical -> flag)
            - pred class diversity (single-class predictions -> flag)
            - source/lab from known-bad list
        Sweep cumulative thresholds; pick the operating point closest to
        the ORACLE ceiling.

Output:
    paper/results/rejection_sweep.txt
    paper/results/rejection_sweep.json
"""
from __future__ import annotations

import csv
import json
import sys
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
V12_CSV = ROOT / "paper" / "results" / "v12_pyramidal_only_seed2024_clean_eval.csv"
DROPPED_LABS_JSON = ROOT / "paper" / "results" / "dropped_labs_summary.json"
OUT_TXT  = ROOT / "paper" / "results" / "rejection_sweep.txt"
OUT_JSON = ROOT / "paper" / "results" / "rejection_sweep.json"


def _lab_prefix(filename: str) -> str:
    name = filename.replace(".swc", "")
    if "__" in name: _, name = name.split("__", 1)
    m = re.match(r"^([A-Za-z]+)", name)
    if m:
        prefix = m.group(1)
        if len(prefix) <= 2:
            m2 = re.match(r"^[A-Za-z]+[0-9]*[A-Za-z]*", name)
            if m2: prefix = m2.group(0)
        return prefix
    return name[:8]


def _summary(F1: np.ndarray) -> dict:
    return {
        "n":        int(F1.size),
        "mean":     float(F1.mean()),
        "P10":      float(np.percentile(F1, 10)),
        "P25":      float(np.percentile(F1, 25)),
        "P50":      float(np.percentile(F1, 50)),
    }


def main() -> int:
    rows = []
    with V12_CSV.open("r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["F1"] = float(r["F1_neurite"])
            r["n_nodes"] = int(r["n_nodes"])
            r["n_apical_pred"] = int(r["n_apical_pred"])
            r["n_basal_pred"]  = int(r["n_basal_pred"])
            r["n_axon_pred"]   = int(r["n_axon_pred"])
            n_neur = r["n_apical_pred"] + r["n_basal_pred"] + r["n_axon_pred"]
            r["pred_apical_frac"] = r["n_apical_pred"] / max(1, n_neur)
            n_present = sum(1 for c in (r["n_apical_pred"], r["n_basal_pred"], r["n_axon_pred"]) if c > 0)
            r["n_classes_predicted"] = n_present
            r["source"] = r["file"].split("__", 1)[0] if "__" in r["file"] else "unknown"
            r["lab_prefix"] = _lab_prefix(r["file"]) if r["source"] == "neuromorpho" else ""
            rows.append(r)
    n_total = len(rows)
    print(f"Loaded {n_total} cells from {V12_CSV.name}")

    # Baseline (no rejection)
    F1_all = np.array([r["F1"] for r in rows])
    baseline = _summary(F1_all)

    # ===================== ORACLE rejection sweep =====================
    rows_by_f1 = sorted(rows, key=lambda r: r["F1"])
    oracle_curves = []
    for pct in (0, 1, 2, 5, 7.5, 10, 12.5, 15, 20):
        n_drop = int(n_total * pct / 100)
        kept = rows_by_f1[n_drop:]
        F1_kept = np.array([r["F1"] for r in kept])
        s = _summary(F1_kept)
        s["pct_rejected"] = pct
        s["n_rejected"]   = n_drop
        oracle_curves.append(s)

    # ===================== REAL proxy rejection sweep =====================
    # Bad-lab list (already cleaned but kept for reference)
    bad_labs: set[str] = set()
    if DROPPED_LABS_JSON.is_file():
        bd = json.loads(DROPPED_LABS_JSON.read_text(encoding="utf-8"))
        bad_labs = set(bd.get("bad_labs_with_rates", {}).keys())

    # Per-cell suspicion score (prediction + structural only; NO GT)
    for r in rows:
        s = 0.0
        # 1) Tiny cells -> high suspicion
        if   r["n_nodes"] < 100:  s += 3
        elif r["n_nodes"] < 200:  s += 2
        elif r["n_nodes"] < 500:  s += 1
        # 2) Pyramidal predicted with 0 apical -> strong red flag
        if r["n_apical_pred"] == 0:                s += 3
        elif r["pred_apical_frac"] < 0.01:         s += 2
        elif r["pred_apical_frac"] < 0.03:         s += 1
        # 3) Single-class predictions
        if r["n_classes_predicted"] <= 1:          s += 3
        elif r["n_classes_predicted"] == 2:        s += 1
        # 4) Lab-prefix risk (from previous failure-rate analysis)
        if r["lab_prefix"] in bad_labs:            s += 2
        r["suspicion_score"] = s

    # Sort by suspicion DESC; sweep "reject top X% by suspicion"
    rows_by_susp = sorted(rows, key=lambda r: -r["suspicion_score"])
    proxy_curves = []
    for pct in (0, 1, 2, 5, 7.5, 10, 12.5, 15, 20):
        n_drop = int(n_total * pct / 100)
        kept = rows_by_susp[n_drop:]
        F1_kept = np.array([r["F1"] for r in kept])
        s = _summary(F1_kept)
        s["pct_rejected"] = pct
        s["n_rejected"]   = n_drop
        # What was the average F1 of the rejected cells? (precision-like signal)
        rejected = rows_by_susp[:n_drop]
        if rejected:
            rej_f1 = np.array([r["F1"] for r in rejected])
            s["rejected_F1_mean"] = float(rej_f1.mean())
            s["rejected_F1_lt_0.5_frac"] = float((rej_f1 < 0.5).mean())
        else:
            s["rejected_F1_mean"] = None
            s["rejected_F1_lt_0.5_frac"] = None
        proxy_curves.append(s)

    # ===================== Report =====================
    L = []
    P = L.append
    P("=" * 80)
    P("  WAVE A1: REJECTION-BASED QUALITY FLAGGING SWEEP")
    P(f"  v12 baseline, n={n_total} pyramidals, eval_with_GT_celltype_override")
    P("=" * 80)
    P("")
    P(f"  baseline (no rejection): F1 mean={baseline['mean']:.4f}  P10={baseline['P10']:.4f}  P25={baseline['P25']:.4f}")
    P("")
    P("== ORACLE rejection (drops bottom-X by GT F1; upper bound) ==")
    P(f"  {'reject %':>10} {'n drop':>8} {'F1 mean':>10} {'F1 P10':>10} {'F1 P25':>10}")
    for c in oracle_curves:
        P(f"  {c['pct_rejected']:>9.1f}% {c['n_rejected']:>8} {c['mean']:>10.4f} {c['P10']:>10.4f} {c['P25']:>10.4f}")
    P("")
    P("== REAL proxy rejection (no GT used) ==")
    P("  Suspicion score = sum of:")
    P("    n_nodes<100=+3, <200=+2, <500=+1")
    P("    pred_apical_frac=0=+3, <0.01=+2, <0.03=+1")
    P("    classes_predicted=1=+3, =2=+1")
    P("    bad_lab=+2")
    P("")
    P(f"  {'reject %':>10} {'n drop':>8} {'F1 mean':>10} {'F1 P10':>10} {'F1 P25':>10}   {'rej F1':>9} {'rej<0.5':>9}")
    for c in proxy_curves:
        rfm = f"{c['rejected_F1_mean']:.4f}" if c['rejected_F1_mean'] is not None else "  -"
        rfl = f"{c['rejected_F1_lt_0.5_frac']:.2f}" if c['rejected_F1_lt_0.5_frac'] is not None else "  -"
        P(f"  {c['pct_rejected']:>9.1f}% {c['n_rejected']:>8} {c['mean']:>10.4f} {c['P10']:>10.4f} {c['P25']:>10.4f}   {rfm:>9} {rfl:>9}")
    P("")
    P("  rej F1 column = mean F1 of REJECTED cells (lower = better precision)")
    P("  rej<0.5 column = fraction of rejected cells with F1<0.5 (higher = better precision)")
    P("")

    # Comparison: oracle vs proxy at 10%
    P("== HEADLINE @ 10% rejection ==")
    oracle_10 = next(c for c in oracle_curves if c["pct_rejected"] == 10)
    proxy_10  = next(c for c in proxy_curves  if c["pct_rejected"] == 10)
    P(f"  oracle: F1 mean={oracle_10['mean']:.4f}  P10={oracle_10['P10']:.4f}")
    P(f"  proxy:  F1 mean={proxy_10['mean']:.4f}  P10={proxy_10['P10']:.4f}")
    P(f"  proxy precision: {proxy_10['rejected_F1_lt_0.5_frac']*100:.1f}% of rejected cells were actually bad (F1<0.5)")
    P(f"  proxy lift vs baseline: P10 {baseline['P10']:.4f} -> {proxy_10['P10']:.4f}  (+{(proxy_10['P10']-baseline['P10'])*100:.2f} pp)")
    P("")

    OUT_TXT.write_text("\n".join(L), encoding="utf-8")
    OUT_JSON.write_text(json.dumps({
        "baseline":    baseline,
        "oracle":      oracle_curves,
        "proxy":       proxy_curves,
    }, indent=2), encoding="utf-8")
    print("\n".join(L))
    print(f"Wrote {OUT_TXT}")
    print(f"Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
