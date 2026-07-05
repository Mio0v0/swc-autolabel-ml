#!/usr/bin/env python3
"""Aggregate per-seed active-learning curves into a multiseed summary.

Reads active_learning_curve_seed{42,123,789}.json and produces per-round
mean +/- SD for each arm, plus an efficiency summary (curator corrections
needed to reach a target macro-F1, flag-guided vs random).

Writes:
    paper/results/active_learning_curve_multiseed.json
    paper/results/active_learning_curve_multiseed.txt
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "paper" / "results"
SEEDS = [42, 123, 789]
METRICS = ["neurite_macro_f1", "apical_f1", "basal_f1", "axon_f1",
           "per_cell_f1_mean", "per_cell_f1_p10", "node_accuracy"]


def _agg_arm(per_seed_curves: list[list[dict]]) -> list[dict]:
    """per_seed_curves: list over seeds of that seed's arm curve (list of rounds)."""
    n_rounds = min(len(c) for c in per_seed_curves)
    out = []
    for r in range(n_rounds):
        row = {
            "round": per_seed_curves[0][r]["round"],
            "n_curator_corrections": per_seed_curves[0][r]["n_curator_corrections"],
        }
        for m in METRICS:
            vals = [c[r][m] for c in per_seed_curves]
            row[m] = round(statistics.mean(vals), 4)
            row[m + "_sd"] = round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4)
        out.append(row)
    return out


def main() -> int:
    flag_curves, rand_curves, seeds_found = [], [], []
    for s in SEEDS:
        p = RESULTS / f"active_learning_curve_seed{s}.json"
        if not p.is_file():
            print(f"  MISSING: {p.name}")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        flag_curves.append(d["flag_arm"])
        rand_curves.append(d["random_arm"])
        seeds_found.append(s)

    if not flag_curves:
        raise SystemExit("No per-seed AL curves found.")

    flag_agg = _agg_arm(flag_curves)
    rand_agg = _agg_arm(rand_curves)

    # Efficiency: corrections to first reach macro-F1 targets, per arm (mean curve).
    def _first_reach(curve, target):
        for row in curve:
            if row["neurite_macro_f1"] >= target:
                return row["n_curator_corrections"]
        return None

    targets = [0.74, 0.75, 0.76, 0.77]
    efficiency = []
    for t in targets:
        efficiency.append({
            "target_macro_f1": t,
            "flag_corrections": _first_reach(flag_agg, t),
            "random_corrections": _first_reach(rand_agg, t),
        })

    out = {
        "seeds": seeds_found,
        "n_seeds": len(seeds_found),
        "metrics": METRICS,
        "flag_mean": flag_agg,
        "random_mean": rand_agg,
        "efficiency": efficiency,
    }
    out_json = RESULTS / "active_learning_curve_multiseed.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = ["Active-learning curve (multiseed): flag-guided vs random",
             "=" * 68, "",
             f"Mean +/- SD over seeds {seeds_found}.",
             "Simulated curator; Stage-2 retrained each round on seed + revealed cells.",
             "",
             f"{'corrections':>11} "
             f"{'macroF1 flag':>16} {'macroF1 rand':>16} {'d':>7}   "
             f"{'apical flag':>16} {'apical rand':>16} {'d':>7}"]
    for ff, rr in zip(flag_agg, rand_agg):
        c = ff["n_curator_corrections"]
        dm = ff["neurite_macro_f1"] - rr["neurite_macro_f1"]
        da = ff["apical_f1"] - rr["apical_f1"]
        lines.append(
            f"{c:>11} "
            f"{ff['neurite_macro_f1']:>7.4f}+/-{ff['neurite_macro_f1_sd']:<6.4f} "
            f"{rr['neurite_macro_f1']:>7.4f}+/-{rr['neurite_macro_f1_sd']:<6.4f} {dm:>+7.4f}   "
            f"{ff['apical_f1']:>7.4f}+/-{ff['apical_f1_sd']:<6.4f} "
            f"{rr['apical_f1']:>7.4f}+/-{rr['apical_f1_sd']:<6.4f} {da:>+7.4f}")
    lines += ["", "Label efficiency (curator corrections to reach macro-F1 target):",
              f"{'target':>8} {'flag':>8} {'random':>8} {'savings':>9}"]
    for e in efficiency:
        f = e["flag_corrections"]; r = e["random_corrections"]
        sv = (f"{100*(1 - f/r):.0f}%" if (f and r and r > 0) else "-")
        lines.append(f"{e['target_macro_f1']:>8.2f} {str(f):>8} {str(r):>8} {sv:>9}")
    out_txt = RESULTS / "active_learning_curve_multiseed.txt"
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\n  wrote {out_json.name}, {out_txt.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
