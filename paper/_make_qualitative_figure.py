#!/usr/bin/env python3
"""Qualitative figure — rendered SWC morphologies, ground truth vs predicted,
colored by structure type. Shows a correctly-labeled cell and a mislabeled
(flag-caught) cell side by side."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.features import parse_swc                      # noqa: E402
from hybrid.pipeline import run_pipeline_on_nodes           # noqa: E402
from hybrid.evaluate import per_cell_neurite_f1             # noqa: E402
from paper.gnn_inference import load_gnn                    # noqa: E402
from paper.gnn_branch3_inference import load_branch3        # noqa: E402

DATA = ROOT / "data" / "v12_uncurated"
MODEL = ROOT / "paper" / "models" / "v12_gentle_seed123"
FIGDIR = ROOT / "paper" / "results" / "figures"

TYPE_COLOR = {1: "#111111", 2: "#2e86c1", 3: "#27ae60", 4: "#c0392b"}  # soma/axon/basal/apical
TYPE_NAME = {1: "soma", 2: "axon", 3: "basal", 4: "apical"}

CELLS = [
    ("neuromorpho__mPFC_w2_L5_06.swc", "pyramidal", "Correct"),
    ("neuromorpho__april11s1cell-1.swc", "pyramidal", "Mislabeled (apical lost)"),
]


def _render(ax, nodes, labels, title):
    id2idx = {n.id: i for i, n in enumerate(nodes)}
    xs = np.array([n.x for n in nodes]); ys = np.array([n.y for n in nodes])
    for i, n in enumerate(nodes):
        pj = id2idx.get(n.parent)
        if pj is None:
            continue
        t = labels[i]
        ax.plot([n.x, nodes[pj].x], [n.y, nodes[pj].y],
                color=TYPE_COLOR.get(int(t), "#999999"), linewidth=0.6, zorder=1)
    # soma marker
    soma = [i for i, n in enumerate(nodes) if int(labels[i]) == 1]
    if soma:
        ax.scatter(xs[soma], ys[soma], s=28, color=TYPE_COLOR[1], zorder=3)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal"); ax.axis("off")


def main() -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    gnn = load_gnn(MODEL / "gnn_apical_basal.pt")
    b3 = load_branch3(MODEL / "gnn_branch3_rescue.pt", gate_path=None)
    s1 = MODEL / "cell_type_classifier.pkl"; s2 = MODEL / "branch_classifier.pkl"

    n_rows = len(CELLS)
    fig, axes = plt.subplots(n_rows, 2, figsize=(7.5, 3.6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 2)

    for r, (fname, ct, tag) in enumerate(CELLS):
        p = DATA / ct / "swc" / fname
        nodes = parse_swc(p)
        gt = [n.type for n in nodes]
        pr = run_pipeline_on_nodes(nodes, file_path="", stage1_model=s1, stage2_model=s2,
                                   gnn_state=gnn, branch3_state=b3, use_subtree_stage2=True,
                                   override_cell_type=ct)
        pred = list(pr.node_labels)
        f1 = per_cell_neurite_f1(gt, pred, ct)
        short = fname.replace("neuromorpho__", "").replace(".swc", "")
        _render(axes[r, 0], nodes, gt, f"{tag}: {short}\nground truth")
        _render(axes[r, 1], nodes, pred, f"predicted  (per-cell F1 = {f1:.2f})")

    handles = [Line2D([0], [0], color=TYPE_COLOR[t], lw=2, label=TYPE_NAME[t]) for t in (1, 2, 3, 4)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Ground-truth vs. predicted labels: a correctly-labeled cell and a "
                 "flag-caught mislabel", fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    png = FIGDIR / "final_figure_qualitative.png"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    fig.savefig(FIGDIR / "final_figure_qualitative.svg", bbox_inches="tight")
    print(f"  wrote {png.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
