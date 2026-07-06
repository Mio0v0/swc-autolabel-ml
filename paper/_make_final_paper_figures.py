#!/usr/bin/env python3
"""Generate final paper figures from canonical result artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, average_precision_score  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "paper" / "results"
FIG_DIR = ROOT / "paper" / "results" / "figures"
CLASS_ORDER = ["soma", "axon", "basal/dendrite", "apical"]
CLASS_LABELS = ["soma", "axon", "basal", "apical"]


def _save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _aggregate_confusion(path: Path, mode: str) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mat = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=float)
    for seed, reports in payload.get("reports", {}).items():
        report = reports.get(mode)
        if not report:
            continue
        conf = report["corpus"]["confusion"]
        for i, gt in enumerate(CLASS_ORDER):
            for j, pred in enumerate(CLASS_ORDER):
                mat[i, j] += float(conf.get(gt, {}).get(pred, 0))
    return mat


def make_confusion_matrix() -> dict:
    path = RESULTS / "final_ablation_ladder_summary.json"
    mat = _aggregate_confusion(path, "full_branch3_gt_celltype")
    row_sum = mat.sum(axis=1, keepdims=True)
    norm = np.divide(mat, row_sum, out=np.zeros_like(mat), where=row_sum > 0)

    csv_path = FIG_DIR / "final_figure_confusion_matrix.csv"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gt", *CLASS_LABELS])
        for label, row in zip(CLASS_LABELS, mat):
            writer.writerow([label, *[int(x) for x in row]])

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_LABELS)), CLASS_LABELS, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASS_LABELS)), CLASS_LABELS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Ground truth label")
    ax.set_title("v12+Branch3 confusion matrix")
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            text = f"{norm[i, j]:.2f}\n{int(mat[i, j]):,}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized fraction")
    _save(fig, "final_figure_confusion_matrix")
    return {"matrix_csv": str(csv_path.relative_to(ROOT))}


def make_f1_distribution() -> dict:
    path = RESULTS / "final_failure_modes.csv"
    df = pd.read_csv(path)
    f1 = pd.to_numeric(df["held_out_F1"], errors="coerce")
    flagged = df["flagged"].astype(str).str.lower().eq("true")
    bins = np.linspace(0.0, 1.0, 41)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(f1.dropna(), bins=bins, density=True, alpha=0.35, label="all rows", color="#4c78a8")
    ax.hist(f1[~flagged].dropna(), bins=bins, density=True, histtype="step", linewidth=2.0,
            label="kept after flag", color="#2f855a")
    ax.hist(f1[flagged].dropna(), bins=bins, density=True, histtype="step", linewidth=2.0,
            label="flagged", color="#c2410c")
    ax.axvline(0.6, color="black", linestyle="--", linewidth=1.2, label="bad threshold")
    ax.set_xlabel("Per-cell neurite macro F1")
    ax.set_ylabel("Density")
    ax.set_title("Per-cell F1 before and after flag rejection")
    ax.legend(frameon=False)
    ax.set_xlim(0, 1.01)
    _save(fig, "final_figure_f1_distribution")

    summary = {
        "n_rows": int(len(df)),
        "n_flagged": int(flagged.sum()),
        "all_p10": float(np.nanpercentile(f1, 10)),
        "kept_p10": float(np.nanpercentile(f1[~flagged], 10)),
        "flagged_bad_rate": float((f1[flagged] < 0.6).mean()) if flagged.any() else 0.0,
    }
    (FIG_DIR / "final_figure_f1_distribution_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def make_flag_pr_curve() -> dict:
    path = RESULTS / "final_failure_mode_flag_scores.csv"
    df = pd.read_csv(path)
    y = (pd.to_numeric(df["held_out_F1"], errors="coerce") < 0.6).astype(int).to_numpy()
    score = pd.to_numeric(df["flag_score"], errors="coerce").fillna(0.0).to_numpy()
    precision, recall, thresholds = precision_recall_curve(y, score)
    ap = float(average_precision_score(y, score))

    csv_path = FIG_DIR / "final_figure_flag_pr_curve.csv"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["precision", "recall", "threshold"])
        for i in range(len(precision)):
            thr = thresholds[i] if i < len(thresholds) else ""
            writer.writerow([float(precision[i]), float(recall[i]), thr])

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(recall, precision, color="#4c78a8", linewidth=2.0)
    ax.set_xlabel("Recall for F1 < 0.6")
    ax.set_ylabel("Precision among flagged rows")
    ax.set_title(f"Flag precision-recall curve (AP={ap:.3f})")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.grid(True, alpha=0.25)
    _save(fig, "final_figure_flag_pr_curve")
    return {"average_precision": ap, "curve_csv": str(csv_path.relative_to(ROOT))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    manifest = {
        "confusion_matrix": make_confusion_matrix(),
        "f1_distribution": make_f1_distribution(),
        "flag_pr_curve": make_flag_pr_curve(),
    }
    out = FIG_DIR / "final_figures_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
