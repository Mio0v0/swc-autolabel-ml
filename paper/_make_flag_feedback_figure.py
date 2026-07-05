#!/usr/bin/env python3
"""Figure: flag-model precision/recall vs curator verdicts (§5.8)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "paper" / "results"
FIGDIR = RESULTS / "figures"


def main() -> int:
    d = json.loads((RESULTS / "flag_feedback_curve.json").read_text(encoding="utf-8"))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # Panel 1: precision, 3 arms
    ax = axes[0]
    fb, rd, fac = d["feedback"], d["random"], d["factory"]
    xs = [r["n_verdicts"] for r in fb]
    ax.axhline(fac[0]["precision"], color="#2c3e50", ls="--", label="Factory (no feedback)")
    for arm, color, lab in ((fb, "#c0392b", "Feedback (review flagged)"),
                            (rd, "#7f8c8d", "Random review")):
        ys = [r["precision"] for r in arm]
        sd = [r["precision_sd"] for r in arm]
        ax.plot(xs, ys, "-o", color=color, label=lab, markersize=4)
        ax.fill_between(xs, [y-s for y,s in zip(ys,sd)], [y+s for y,s in zip(ys,sd)],
                        color=color, alpha=0.15)
    ax.set_xlabel("Curator verdicts"); ax.set_ylabel("Flag precision @10% reject")
    ax.set_title("Flag precision vs. curator verdicts"); ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    # Panel 2: recall + AP for feedback arm
    ax = axes[1]
    for m, color, lab in (("recall", "#c0392b", "Recall @10% reject"),
                          ("ap", "#2980b9", "Average precision")):
        ys = [r[m] for r in fb]; sd = [r[m + "_sd"] for r in fb]
        ax.plot(xs, ys, "-o", color=color, label=lab, markersize=4)
        ax.fill_between(xs, [y-s for y,s in zip(ys,sd)], [y+s for y,s in zip(ys,sd)],
                        color=color, alpha=0.15)
    ax.set_xlabel("Curator verdicts"); ax.set_ylabel("Score")
    ax.set_title("Feedback arm: recall & AP"); ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Curator feedback retrains the flag model: precision and recall "
                 "rise, faster than random review", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = FIGDIR / "final_figure_flag_feedback.png"
    fig.savefig(png, dpi=150); fig.savefig(FIGDIR / "final_figure_flag_feedback.svg")
    print(f"  wrote {png.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
