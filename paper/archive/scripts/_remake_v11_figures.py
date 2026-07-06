#!/usr/bin/env python3
"""Remake Figure 1 + Figure 2 with the honest seed-averaged P10 (0.74).

Only the P10 number is changed; all other displayed values (per-class
P/R/F1, per-node accuracy, neurite F1, mean, median) are unchanged.
Dot positions in Figure 2 come from the recovered seed=42 per-cell CSV
(eval_v11_final_per_file.csv) and the Sholl-RF rows of the recovered
external-baseline per-cell CSV. The +13.4 pt delta annotation reflects
0.74 - 0.606.

Output:
    paper/figures/figure1_detailed_performance.png
    paper/figures/figure2_per_file_distribution.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
REC  = ROOT / "paper" / "_recovered_v11"
OUT  = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- 3-seed averaged numbers (seeds 42, 123, 456) ---
# Per-class P/R/F1: simple mean across the 3 per-seed JSONs (overall_stage23).
# Supports: sum across 3 seeds.
PER_CLASS = [
    # name             P_mean   R_mean   F1_mean   support_sum
    ("Soma",           1.0000,  0.9994,  0.9997,         1563),
    ("Axon",           0.9978,  0.9990,  0.9984,   34_568_310),
    ("Basal dendrite", 0.9482,  0.9646,  0.9563,    2_552_626),
    ("Apical dendrite",0.9701,  0.9284,  0.9488,    1_919_477),
]
# Corpus-level: simple mean of per-seed overall_stage23 stats.
SUMMARY = [
    ("Per-node accuracy",         "0.9933"),  # avg(0.9934, 0.9927, 0.9937)
    ("Macro-F1 (4 classes)",      "0.9758"),  # avg(0.9755, 0.9740, 0.9779)
    ("Neurite-only macro-F1",     "0.9678"),  # avg(0.9673, 0.9653, 0.9709)
    ("Balanced accuracy",         "0.9728"),  # avg(0.9720, 0.9737, 0.9728)
    ("Per-file F1 — mean",        "0.9462"),  # 3-seed mean of per-seed means
    ("Per-file F1 — median",      "1.0000"),
    ("Per-file F1 — 10th pctile", "0.7400"),  # 3-seed mean of per-seed P10s rounded to 0.74
]

P10_NEW          = 0.7400   # 3-seed mean of per-seed P10s, rounded to 0.74
P10_BASELINE     = 0.606    # Sholl-RF P10 (seed=42 only -- only seed available for baselines)
MEAN_SWC         = 0.9462   # 3-seed mean
MEAN_SHOLL       = 0.920    # seed=42, unchanged


# ---------------- Figure 1 ----------------
def make_figure1() -> Path:
    fig, ax = plt.subplots(figsize=(10, 9.5))
    ax.axis("off")

    # ---- A. Per-class P/R/F1 ----
    A_top    = 0.98
    A_label  = 0.96
    A_height = 0.32
    A_bbox   = [0.05, A_top - A_label - A_height + 0.94, 0.92, A_height]
    # Simpler: hard-code positions
    ax.text(0.05, 0.97, "A. Per-class performance",
            transform=ax.transAxes, fontsize=14, fontweight="bold",
            color="black", ha="left", va="top")

    headers_top = ["Class", "Precision", "Recall", "F1", "Support (nodes)"]
    rows_top = [
        [name,
         f"{p:.4f}", f"{r:.4f}", f"{f:.4f}",
         f"{s:,}"] for (name, p, r, f, s) in PER_CLASS
    ]
    tbl = ax.table(
        cellText=rows_top, colLabels=headers_top,
        cellLoc="center", loc="upper center",
        bbox=[0.05, 0.58, 0.92, 0.32],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    for j in range(len(headers_top)):
        c = tbl[(0, j)]
        c.set_facecolor("#1f3a64")
        c.set_text_props(color="white", weight="bold")
        c.set_height(0.10)
    for i in range(1, len(rows_top) + 1):
        tbl[(i, 0)].set_text_props(ha="left")
        tbl[(i, 0)]._loc = "left"
        for j in range(len(headers_top)):
            tbl[(i, j)].set_height(0.075)

    # ---- B. Overall / per-file summary ----
    ax.text(0.05, 0.50, "B. Overall / per-file summary",
            transform=ax.transAxes, fontsize=14, fontweight="bold",
            color="black", ha="left", va="top")

    headers_bot = ["Metric", "Value"]
    rows_bot = [[k, v] for (k, v) in SUMMARY]
    tbl2 = ax.table(
        cellText=rows_bot, colLabels=headers_bot,
        cellLoc="center",
        bbox=[0.20, 0.02, 0.60, 0.40],
    )
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(11)
    for j in range(len(headers_bot)):
        c = tbl2[(0, j)]
        c.set_facecolor("#1f3a64")
        c.set_text_props(color="white", weight="bold")
        c.set_height(0.05)
    for i in range(1, len(rows_bot) + 1):
        tbl2[(i, 0)].set_text_props(ha="left")
        tbl2[(i, 0)]._loc = "left"
        for j in range(len(headers_bot)):
            tbl2[(i, j)].set_height(0.045)

    out = OUT / "figure1_detailed_performance.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------------- Figure 2 ----------------
def _load_swc_f1() -> np.ndarray:
    """Pool per-cell F1s across all 3 v11-era seeds (42, 123, 456)."""
    f1s: list[float] = []
    for fn in ("eval_v11_final_per_file.csv",
               "eval_multi_seed_123_per_file.csv",
               "eval_multi_seed_456_per_file.csv"):
        with (REC / fn).open("r", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                f1s.append(float(r["neurite_macro_f1_stage23"]))
    return np.asarray(f1s)


def _load_sholl_rf_f1() -> np.ndarray:
    f1s = []
    with (REC / "external_baselines_per_file.csv").open("r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["method"] == "sholl_rf":
                f1s.append(float(r["neurite_macro_f1"]))
    return np.asarray(f1s)


def make_figure2() -> Path:
    swc   = _load_swc_f1()
    sholl = _load_sholl_rf_f1()

    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5),
                              gridspec_kw={"width_ratios": [1.0, 1.05]})

    # ---- Left: strip plot ----
    ax = axes[0]
    ax.text(0.0, 1.18, "A. Per-file F1 distribution",
            transform=ax.transAxes, fontsize=14, fontweight="bold",
            color="black", ha="left", va="bottom")
    ax.text(0.0, 1.10,
            "SWC-Studio: n=1563 pooled across 3 seeds (42, 123, 456).",
            transform=ax.transAxes, fontsize=9, color="black", ha="left", va="bottom")
    ax.text(0.0, 1.04,
            "Sholl-RF: n=461 (seed=42 only).  "
            "Red bar = P10  •  Black bar = median  •  White dot = mean",
            transform=ax.transAxes, fontsize=9, color="black", ha="left", va="bottom")

    rng = np.random.default_rng(0)
    x_swc   = 0 + rng.uniform(-0.18, 0.18, size=swc.size)
    x_sholl = 1 + rng.uniform(-0.18, 0.18, size=sholl.size)
    ax.scatter(x_swc, swc, s=22, color="#3c5a8a", alpha=0.55, edgecolor="none")
    ax.scatter(x_sholl, sholl, s=22, color="#888", alpha=0.55, edgecolor="none")

    # Median bars
    for x, vals in [(0, swc), (1, sholl)]:
        m = float(np.median(vals))
        ax.hlines(m, x - 0.32, x + 0.32, colors="black", linewidth=2.5, zorder=4)
    # P10 bars  (SWC uses the seed-averaged 0.74; Sholl uses real)
    ax.hlines(P10_NEW, -0.32, 0.32, colors="#c0392b", linewidth=3.0, zorder=5)
    ax.hlines(P10_BASELINE, 1 - 0.32, 1 + 0.32, colors="#c0392b", linewidth=3.0, zorder=5)
    # Mean dots (open circle)
    ax.scatter([0], [MEAN_SWC],   s=80, facecolor="white", edgecolor="black", linewidth=1.5, zorder=6)
    ax.scatter([1], [MEAN_SHOLL], s=80, facecolor="white", edgecolor="black", linewidth=1.5, zorder=6)

    # Annotations  (display P10 as 0.74; underlying value 0.7382)
    ax.annotate("P10 = 0.74",                     xy=(0.34, P10_NEW),
                xytext=(0.40, P10_NEW), color="#c0392b", fontsize=11, va="center", fontweight="bold")
    ax.annotate(f"Mean = {MEAN_SWC:.3f}",         xy=(0.20, MEAN_SWC + 0.025),
                xytext=(0.30, MEAN_SWC + 0.025), color="black", fontsize=11, va="center")
    ax.annotate(f"P10 = {P10_BASELINE:.3f}",      xy=(1.34, P10_BASELINE),
                xytext=(1.40, P10_BASELINE), color="#c0392b", fontsize=11, va="center", fontweight="bold")
    ax.annotate(f"Mean = {MEAN_SHOLL:.3f}",       xy=(1.20, MEAN_SHOLL + 0.025),
                xytext=(1.30, MEAN_SHOLL + 0.025), color="black", fontsize=11, va="center")

    # Delta arrow (0.74 - 0.606 = +13.4 pts)
    delta_pp = (P10_NEW - P10_BASELINE) * 100.0
    ax.annotate(
        "",
        xy=(0.70, P10_BASELINE), xytext=(0.70, P10_NEW),
        arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=2),
    )
    ax.text(0.78, (P10_NEW + P10_BASELINE) / 2,
            f"+{delta_pp:.1f}\npts",
            color="#c0392b", fontsize=12, fontweight="bold", va="center")

    # "N cells below F1=0.7"
    n_below_swc = int((swc < 0.70).sum())
    n_below_sho = int((sholl < 0.70).sum())
    ax.text(0, -0.06, f"{n_below_swc} cells\nbelow F1 = 0.7",
            ha="center", va="top", color="#1f3a64", fontsize=11, fontweight="bold",
            transform=ax.get_xaxis_transform())
    ax.text(1, -0.06, f"{n_below_sho} cells\nbelow F1 = 0.7",
            ha="center", va="top", color="#666", fontsize=11,
            transform=ax.get_xaxis_transform())

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["SWC-Studio", "Sholl-RF"], fontsize=12)
    ax.set_xlim(-0.6, 1.85)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Per-file neurite-macro-F1 (Stage 2+3)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # ---- Right: survival curve ----
    ax = axes[1]
    ax.text(0.0, 1.18, "B. Survival curve",
            transform=ax.transAxes, fontsize=14, fontweight="bold",
            color="black", ha="left", va="bottom")
    ax.text(0.0, 1.10,
            "Higher = better (more cells above threshold).",
            transform=ax.transAxes, fontsize=9, color="black", ha="left", va="bottom")
    ax.text(0.0, 1.04,
            "SWC-Studio curve pooled across 3 seeds; Sholl-RF curve from seed=42.",
            transform=ax.transAxes, fontsize=9, color="black", ha="left", va="bottom")

    thresholds = np.linspace(0.0, 1.0, 401)
    surv_swc   = np.array([(swc   >= t).mean() for t in thresholds])
    surv_sholl = np.array([(sholl >= t).mean() for t in thresholds])

    ax.plot(thresholds, surv_swc,   color="#1f3a64", lw=2.5, label="SWC-Studio")
    ax.plot(thresholds, surv_sholl, color="#888",    lw=2.0, linestyle="--", label="Sholl-RF")
    ax.axvline(0.9, color="#999", linestyle=":", lw=1)
    ax.text(0.905, 0.02, "F1 = 0.9", color="#666", fontsize=10)

    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.02)
    ax.set_xlabel("F1 threshold", fontsize=11)
    ax.set_ylabel("Fraction of cells with F1 ≥ threshold", fontsize=11)
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=11)

    out = OUT / "figure2_per_file_distribution.png"
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> int:
    p1 = make_figure1()
    p2 = make_figure2()
    print(f"Wrote {p1}")
    print(f"Wrote {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
