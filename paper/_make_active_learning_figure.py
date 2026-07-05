#!/usr/bin/env python3
"""Figure: active-learning curve — flag-guided vs random curator selection.

Reads paper/results/active_learning_curve.json (single seed) or the
multiseed aggregate, and renders a two-panel figure:
  (left)  neurite macro-F1 vs number of curator corrections
  (right) apical F1 vs number of curator corrections
with flag-guided and random arms overlaid.

Usage:
    python -m paper._make_active_learning_figure
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "paper" / "results"
FIGDIR = RESULTS / "figures"


def _load_multiseed():
    """Prefer the multiseed aggregate if present, else the single-seed run."""
    agg = RESULTS / "active_learning_curve_multiseed.json"
    if agg.is_file():
        return json.loads(agg.read_text(encoding="utf-8")), True
    single = RESULTS / "active_learning_curve.json"
    return json.loads(single.read_text(encoding="utf-8")), False


def _series(curve, xkey, ykey):
    xs = [r[xkey] for r in curve]
    ys = [r[ykey] for r in curve]
    return xs, ys


def main() -> int:
    data, is_multi = _load_multiseed()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    panels = [("neurite_macro_f1", "Neurite macro-F1"),
              ("apical_f1", "Apical F1")]

    if is_multi:
        # Aggregate: each arm has per-round mean + sd arrays.
        for ax, (ykey, ylabel) in zip(axes, panels):
            for arm, color, label in (("flag", "#c0392b", "Flag-guided"),
                                      ("random", "#7f8c8d", "Random")):
                rounds = data[f"{arm}_mean"]
                xs = [r["n_curator_corrections"] for r in rounds]
                ys = [r[ykey] for r in rounds]
                sd = [r.get(ykey + "_sd", 0.0) for r in rounds]
                ax.plot(xs, ys, "-o", color=color, label=label, markersize=4)
                lo = [y - s for y, s in zip(ys, sd)]
                hi = [y + s for y, s in zip(ys, sd)]
                ax.fill_between(xs, lo, hi, color=color, alpha=0.15)
            ax.set_xlabel("Curator corrections")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel + " vs. curator effort")
            ax.grid(True, alpha=0.3)
            ax.legend(frameon=False)
    else:
        cflag = data["flag_arm"]; crand = data["random_arm"]
        for ax, (ykey, ylabel) in zip(axes, panels):
            xf, yf = _series(cflag, "n_curator_corrections", ykey)
            xr, yr = _series(crand, "n_curator_corrections", ykey)
            ax.plot(xf, yf, "-o", color="#c0392b", label="Flag-guided", markersize=4)
            ax.plot(xr, yr, "-o", color="#7f8c8d", label="Random", markersize=4)
            ax.set_xlabel("Curator corrections")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel + " vs. curator effort")
            ax.grid(True, alpha=0.3)
            ax.legend(frameon=False)

    fig.suptitle("Simulated-curator active learning: flag-guided selection "
                 "improves the labeler faster than random", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = FIGDIR / "final_figure_active_learning.png"
    svg = FIGDIR / "final_figure_active_learning.svg"
    fig.savefig(png, dpi=150)
    fig.savefig(svg)
    print(f"  wrote {png.relative_to(ROOT)}")
    print(f"  wrote {svg.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
