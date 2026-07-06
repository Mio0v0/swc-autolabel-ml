#!/usr/bin/env python3
"""Figure 1 — pipeline schematic (QC → cell type → Stage 2 → GNN →
Stage 3 + Branch3 → flag → curator feedback loop)."""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "paper" / "results" / "figures"

# palette
C_INPUT = "#d5dbdb"
C_MODEL = "#aed6f1"
C_RULE  = "#a9dfbf"
C_FLAG  = "#f5b7b1"
C_LOOP  = "#f9e79f"
EDGE = "#34495e"


def box(ax, x, y, w, h, text, color, fontsize=9, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                       linewidth=1.3, edgecolor=EDGE, facecolor=color, zorder=2)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold" if bold else "normal", zorder=3)
    return (x, y, w, h)


def arrow(ax, x0, y0, x1, y1, style="-|>", color=EDGE, ls="-", lw=1.5, rad=0.0):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14,
                        linewidth=lw, color=color, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=1)
    ax.add_patch(a)


def main() -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.2); ax.axis("off")

    # Main flow (left → right), y = 3.4
    y = 3.4; h = 0.9; gap = 0.35
    xs = 0.2
    b_in  = box(ax, xs, y, 1.15, h, "Raw\nSWC", C_INPUT, bold=True); xs += 1.15 + gap
    b_qc  = box(ax, xs, y, 1.35, h, "Stage 0\nQC gate", C_RULE); xs += 1.35 + gap
    b_ct  = box(ax, xs, y, 1.7, h, "Stage 1\nCell type\n(user or predict)", C_MODEL); xs += 1.7 + gap
    b_s2  = box(ax, xs, y, 1.6, h, "Stage 2\nsubtree\nclassifier", C_MODEL); xs += 1.6 + gap
    b_gnn = box(ax, xs, y, 1.7, h, "Stage 2.5\napical/basal\nGNN + Branch3", C_MODEL); xs += 1.7 + gap
    b_s3  = box(ax, xs, y, 1.5, h, "Stage 3\ntopology\nrefinement", C_RULE); xs += 1.5 + gap

    for a, b in ((b_in, b_qc), (b_qc, b_ct), (b_ct, b_s2), (b_s2, b_gnn), (b_gnn, b_s3)):
        arrow(ax, a[0]+a[2], a[1]+a[3]/2, b[0], b[1]+b[3]/2)

    # Labeled SWC (below Stage 3)
    b_lab = box(ax, b_s3[0], 2.0, b_s3[2], 0.7, "Labeled SWC", C_INPUT, bold=True)
    arrow(ax, b_s3[0]+b_s3[2]/2, b_s3[1], b_lab[0]+b_lab[2]/2, b_lab[1]+b_lab[3])

    # Flag (below labeled), y=0.9
    b_flag = box(ax, 8.15, 0.85, 1.7, 0.8, "Stage 4\nFlag model", C_FLAG)
    arrow(ax, b_lab[0]+b_lab[2]/2, b_lab[1], b_flag[0]+b_flag[2]/2, b_flag[1]+b_flag[3])

    # accept (right of flag)
    b_acc = box(ax, 10.2, 0.9, 1.55, 0.7, "accept\nlabels", C_INPUT, fontsize=8)
    arrow(ax, b_flag[0]+b_flag[2], b_flag[1]+b_flag[3]/2, b_acc[0], b_acc[1]+b_acc[3]/2)
    ax.text((b_flag[0]+b_flag[2]+b_acc[0])/2, b_acc[1]+b_acc[3]+0.12, "not flagged",
            ha="center", fontsize=7, style="italic", color="#555")

    # Curator review (left of flag), y=0.9
    b_cur = box(ax, 5.35, 0.7, 2.35, 1.1,
                "Stage 5\nCurator review\n(supplies corrected labels)", C_LOOP, fontsize=8)
    arrow(ax, b_flag[0], b_flag[1]+b_flag[3]/2, b_cur[0]+b_cur[2], b_cur[1]+b_cur[3]/2)
    ax.text((b_flag[0]+b_cur[0]+b_cur[2])/2, b_cur[1]+b_cur[3]+0.12, "flagged",
            ha="center", fontsize=7, style="italic", color="#a33")

    # Verified pool (left)
    b_pool = box(ax, 2.6, 0.8, 2.1, 0.9,
                 "Curator-verified\npool", C_LOOP, fontsize=8)
    arrow(ax, b_cur[0], b_cur[1]+b_cur[3]/2, b_pool[0]+b_pool[2], b_pool[1]+b_pool[3]/2)
    ax.text((b_cur[0]+b_pool[0]+b_pool[2])/2, b_pool[1]+b_pool[3]+0.1,
            "labels derive\nflag truth", ha="center", fontsize=6.5, color="#666")

    # Feedback arrow: pool → up into Stage 2 / GNN / flag (dashed, versioned retrain)
    arrow(ax, b_pool[0]+b_pool[2]/2, b_pool[1]+b_pool[3],
          b_s2[0]+b_s2[2]/2, b_s2[1], ls="--", color="#b9770e", lw=1.8, rad=-0.25)
    ax.text(3.3, 2.55, "versioned retrain\n(factory preserved)", ha="center",
            fontsize=7.5, color="#b9770e", style="italic")

    # Legend
    from matplotlib.patches import Patch
    leg = [Patch(facecolor=C_MODEL, edgecolor=EDGE, label="learned (XGBoost / GNN)"),
           Patch(facecolor=C_RULE, edgecolor=EDGE, label="rule-based"),
           Patch(facecolor=C_FLAG, edgecolor=EDGE, label="quality flag"),
           Patch(facecolor=C_LOOP, edgecolor=EDGE, label="human-in-the-loop")]
    ax.legend(handles=leg, loc="upper center", ncol=4, frameon=False,
              fontsize=8.5, bbox_to_anchor=(0.5, 1.06))

    fig.tight_layout()
    png = FIGDIR / "final_figure_pipeline_schematic.png"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    fig.savefig(FIGDIR / "final_figure_pipeline_schematic.svg", bbox_inches="tight")
    print(f"  wrote {png.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
