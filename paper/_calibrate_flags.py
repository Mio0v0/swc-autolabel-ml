#!/usr/bin/env python3
"""Calibrate the two-level confidence flag thresholds on the v12 test split.

For each test cell:
  - Run the v12 pipeline (Stage 1+2, optionally Stage 3 GNN).
  - Compute per-branch + per-cell confidence summaries.
  - Compare predictions to ground truth (already normalized by parse_swc).

For each candidate threshold:
  - Per-branch flag precision = P(branch wrong | branch flagged)
  - Per-cell  flag precision  = P(cell bad   | cell flagged)
    where "bad cell" := neurite-macro-F1 < 0.85 (configurable).

Pick the thresholds that hit a target precision (default 0.90). Save
the calibrated ConfidenceConfig to paper/models/v12/flag_config.json.

Usage:
    python -m paper._calibrate_flags
    python -m paper._calibrate_flags --target-precision 0.95
    python -m paper._calibrate_flags --bad-cell-f1-threshold 0.80
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.features import parse_swc                                       # noqa: E402
from hybrid.pipeline import run_pipeline_on_nodes                           # noqa: E402
from hybrid.confidence import (                                             # noqa: E402
    ConfidenceConfig, summarize_confidence,
)

# ---------------------------------------------------------------------------
# Pinned inputs
# ---------------------------------------------------------------------------
DATA_DIR    = ROOT / "data" / "v12_uncurated"
# Default paths — overridable via CLI args.
DEFAULT_MODEL_DIR    = ROOT / "paper" / "models" / "v12"
DEFAULT_GNN_FALLBACK = ROOT / "paper" / "models" / "gnn_apical_basal_v11_final.pt"


def _neurite_macro_f1(gt: list[int], pred: list[int]) -> float:
    """Macro-F1 over neurite classes (2, 3, 4), ignoring soma and undefined."""
    if not gt:
        return 0.0
    return float(f1_score(gt, pred, labels=[2, 3, 4], average="macro", zero_division=0))


def _load_test_files(split_json: Path) -> list[tuple[str, Path]]:
    split = json.loads(split_json.read_text(encoding="utf-8"))
    out: list[tuple[str, Path]] = []
    for ct in ("pyramidal", "interneuron"):
        for name in split["test"].get(ct, []):
            p = DATA_DIR / ct / "swc" / name
            if p.is_file():
                out.append((ct, p))
    return out


# ---------------------------------------------------------------------------
# Run inference and collect calibration records
# ---------------------------------------------------------------------------
def run_inference_on_test(
    test_files: list[tuple[str, Path]],
    gnn_path: Path | None,
    s1_model: Path,
    s2_model: Path,
    sample: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Return (per_branch_records, per_cell_records).

    per_branch_records: one row per branch:
        {
            'file': str, 'cell_type_gt': str, 'cell_type_pred': str,
            'n_nodes': int,
            'pred_label': int, 'gt_label': int,
            'is_wrong': bool,                # majority pred != majority gt
            'frac_correct_nodes': float,
            'mean_conf': float, 'min_conf': float,
        }
    per_cell_records: one row per cell:
        {
            'file': str, 'cell_type_gt': str, 'cell_type_pred': str,
            'n_nodes': int,
            'neurite_f1': float,
            'is_bad': bool,                  # will be filled per-call by caller
            'mean_node_conf': float, 'frac_low_conf': float,
            'stage1_conf': float,
        }
    """
    gnn_state = None
    if gnn_path is not None and gnn_path.is_file():
        from paper.gnn_inference import load_gnn
        print(f"  Loading GNN: {gnn_path.name}")
        gnn_state = load_gnn(gnn_path)

    if sample:
        test_files = test_files[:sample]

    branch_rows: list[dict] = []
    cell_rows: list[dict] = []
    t0 = time.perf_counter()
    n_files = len(test_files)
    print(f"  Running inference on {n_files} held-out test cells...")

    for i, (ct_gt, path) in enumerate(test_files):
        try:
            nodes = parse_swc(path)   # normalized
            if not nodes:
                continue
            pr = run_pipeline_on_nodes(
                nodes, file_path=str(path),
                stage1_model=s1_model, stage2_model=s2_model,
                gnn_state=gnn_state, use_subtree_stage2=True,
            )
        except Exception as exc:
            print(f"  WARN: {path.name}: {exc}")
            continue

        gt_labels  = [n.type for n in nodes]
        pred_labels = pr.node_labels
        confs       = pr.node_confidences

        # Per-cell stats
        neurite_f1 = _neurite_macro_f1(gt_labels, pred_labels)
        # Use ConfidenceConfig.default() for the segmentation thresholds we
        # only use for thresholding low-conf nodes; the actual flag is set later.
        cfg_default = ConfidenceConfig.default()
        branches, cell = summarize_confidence(
            nodes, pred_labels, confs,
            stage1_cell_type=pr.stage1.cell_type,
            stage1_confidence=float(pr.stage1.confidence),
            cfg=cfg_default,
        )

        cell_rows.append({
            "file":            path.name,
            "cell_type_gt":    ct_gt,
            "cell_type_pred":  pr.stage1.cell_type,
            "n_nodes":         len(nodes),
            "neurite_f1":      neurite_f1,
            "mean_node_conf":  cell.mean_node_confidence,
            "frac_low_conf":   cell.fraction_low_confidence,
            "stage1_conf":     cell.stage1_confidence,
        })

        # Per-branch stats. For each branch, the predicted majority label is
        # already in BranchConfidence; the ground-truth majority comes from
        # the same node indices.
        for b in branches:
            gt_labs = [gt_labels[idx] for idx in b.node_indices]
            gt_majority = Counter(gt_labs).most_common(1)[0][0]
            n_correct = sum(1 for idx in b.node_indices if pred_labels[idx] == gt_labels[idx])
            branch_rows.append({
                "file":           path.name,
                "cell_type_gt":   ct_gt,
                "cell_type_pred": pr.stage1.cell_type,
                "n_nodes":        b.n_nodes,
                "pred_label":     b.predicted_label,
                "gt_label":       int(gt_majority),
                "is_wrong":       int(gt_majority != b.predicted_label),
                "frac_correct_nodes": n_correct / b.n_nodes,
                "mean_conf":      b.mean_confidence,
                "min_conf":       b.min_confidence,
            })

        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{n_files}  (elapsed {(time.perf_counter()-t0)/60:.1f} min)")

    print(f"  inference elapsed: {(time.perf_counter()-t0)/60:.1f} min")
    return branch_rows, cell_rows


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------
def sweep_branch_threshold(
    branch_rows: list[dict],
    target_precision: float,
) -> tuple[float, dict]:
    """Find largest threshold τ such that
       precision(branch wrong | mean_conf < τ) >= target_precision.

    Returns (chosen_threshold, summary_dict).
    """
    df = sorted(branch_rows, key=lambda r: r["mean_conf"])
    # As we widen the threshold from low->high, the flagged set grows.
    # For each split-point τ = mean_conf[i], the flagged set is everything
    # to the left (< τ); precision = wrong_count / flag_count over that set.
    n = len(df)
    wrong = np.array([r["is_wrong"] for r in df], dtype=np.int64)
    confs = np.array([r["mean_conf"] for r in df], dtype=np.float64)

    cum_wrong = np.cumsum(wrong)
    cum_flag  = np.arange(1, n + 1)

    # For each candidate τ-cutoff index k (flag the first k entries),
    # precision = cum_wrong[k-1] / k, coverage = (n - k) / n
    precision = cum_wrong / cum_flag
    recall    = cum_wrong / max(1, int(wrong.sum()))

    # We want the LARGEST k (loosest flag → most coverage) with precision >= target.
    eligible = np.where(precision >= target_precision)[0]
    if eligible.size:
        k_best = int(eligible.max())
        chosen_threshold = float(confs[k_best])
        chosen_precision = float(precision[k_best])
        chosen_recall    = float(recall[k_best])
        chosen_n_flagged = int(k_best + 1)
    else:
        # No threshold achieves target. Fall back to most-conservative.
        k_best = 0
        chosen_threshold = float(confs[0]) if n else 0.5
        chosen_precision = float(precision[0]) if n else 0.0
        chosen_recall    = float(recall[0])    if n else 0.0
        chosen_n_flagged = 1 if n else 0

    return chosen_threshold, {
        "target_precision":    target_precision,
        "achieved_precision":  chosen_precision,
        "achieved_recall":     chosen_recall,
        "n_branches_total":    n,
        "n_branches_flagged":  chosen_n_flagged,
        "n_branches_wrong":    int(wrong.sum()),
        "coverage":            1.0 - chosen_n_flagged / max(1, n),
    }


def sweep_cell_threshold(
    cell_rows: list[dict],
    bad_cell_f1_threshold: float,
    target_precision: float,
) -> tuple[float, float, dict]:
    """Joint sweep on two cell-level signals:
       - mean_node_conf  threshold (flag fires if BELOW)
       - frac_low_conf   threshold (flag fires if ABOVE)
    A cell is "bad" iff neurite_f1 < bad_cell_f1_threshold.

    Returns (cell_mean_threshold, cell_frac_low_threshold, summary_dict).

    For simplicity we sweep mean_node_conf only and set frac_low_conf
    threshold to a fixed value (0.20). A future version could grid-search.
    """
    is_bad = np.array(
        [int(r["neurite_f1"] < bad_cell_f1_threshold) for r in cell_rows],
        dtype=np.int64,
    )
    mean_conf = np.array([r["mean_node_conf"] for r in cell_rows], dtype=np.float64)

    # Sort by mean_conf ascending (smallest first = most likely to be flagged)
    order = np.argsort(mean_conf)
    sorted_bad  = is_bad[order]
    sorted_conf = mean_conf[order]
    n = len(cell_rows)

    cum_bad  = np.cumsum(sorted_bad)
    cum_flag = np.arange(1, n + 1)
    precision = cum_bad / cum_flag
    recall    = cum_bad / max(1, int(is_bad.sum()))

    eligible = np.where(precision >= target_precision)[0]
    if eligible.size:
        k_best = int(eligible.max())
        chosen_threshold = float(sorted_conf[k_best])
        chosen_precision = float(precision[k_best])
        chosen_recall    = float(recall[k_best])
        chosen_n_flagged = int(k_best + 1)
    else:
        k_best = 0
        chosen_threshold = float(sorted_conf[0]) if n else 0.5
        chosen_precision = float(precision[0]) if n else 0.0
        chosen_recall    = float(recall[0])    if n else 0.0
        chosen_n_flagged = 1 if n else 0

    # Fixed frac_low threshold — could be jointly optimized later.
    cell_frac_low_threshold = 0.20

    return chosen_threshold, cell_frac_low_threshold, {
        "target_precision":     target_precision,
        "bad_cell_f1_threshold": bad_cell_f1_threshold,
        "achieved_precision":   chosen_precision,
        "achieved_recall":      chosen_recall,
        "n_cells_total":        n,
        "n_cells_flagged":      chosen_n_flagged,
        "n_cells_bad":          int(is_bad.sum()),
        "coverage":             1.0 - chosen_n_flagged / max(1, n),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-precision", type=float, default=0.90,
                    help="Target precision for the flag thresholds (default 0.90).")
    ap.add_argument("--bad-cell-f1-threshold", type=float, default=0.85,
                    help="Cells with neurite-F1 below this are considered 'bad' (default 0.85).")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                    help="Dir containing cell_type_classifier.pkl, branch_classifier.pkl, "
                         "gnn_apical_basal.pt, and train_test_split.json (default: v12 baseline).")
    ap.add_argument("--gnn-model", type=Path, default=None,
                    help="Override path to GNN .pt; default uses <model-dir>/gnn_apical_basal.pt.")
    ap.add_argument("--tag", type=str, default=None,
                    help="Suffix tag for output files (default: derived from --model-dir name).")
    ap.add_argument("--sample", type=int, default=None,
                    help="Use only the first N test cells (debug option).")
    args = ap.parse_args()

    # Resolve paths from model dir
    s1_model    = args.model_dir / "cell_type_classifier.pkl"
    s2_model    = args.model_dir / "branch_classifier.pkl"
    gnn_default = args.model_dir / "gnn_apical_basal.pt"
    split_json  = args.model_dir / "train_test_split.json"
    tag         = args.tag or args.model_dir.name

    flag_config_out = args.model_dir / "flag_config.json"
    curves_csv      = ROOT / "paper" / "results" / f"flag_calibration_curves_{tag}.csv"
    summary_json    = ROOT / "paper" / "results" / f"flag_calibration_summary_{tag}.json"

    print(f"Model dir: {args.model_dir}")
    print(f"Tag      : {tag}")

    # GNN selection
    if args.gnn_model:
        gnn_path = args.gnn_model
    elif gnn_default.is_file():
        gnn_path = gnn_default
        print(f"Using GNN at {gnn_path}")
    elif DEFAULT_GNN_FALLBACK.is_file():
        gnn_path = DEFAULT_GNN_FALLBACK
        print(f"GNN not found in model dir; falling back to {gnn_path}")
    else:
        gnn_path = None
        print("WARN: no GNN checkpoint found; Stage 3 will be skipped.")

    # Load test split
    test_files = _load_test_files(split_json)
    print(f"Test split: {len(test_files)} cells "
          f"({sum(1 for ct,_ in test_files if ct=='pyramidal')} pyramidal, "
          f"{sum(1 for ct,_ in test_files if ct=='interneuron')} interneuron)")

    # Inference
    print()
    print("[1] Inference on test set")
    branch_rows, cell_rows = run_inference_on_test(
        test_files, gnn_path, s1_model, s2_model, args.sample,
    )
    print(f"  collected {len(branch_rows)} branch rows, {len(cell_rows)} cell rows")

    # Threshold sweeps
    print()
    print("[2] Per-branch flag calibration")
    branch_threshold, branch_summary = sweep_branch_threshold(
        branch_rows, args.target_precision,
    )
    print(f"  branch_flag_threshold = {branch_threshold:.4f}")
    print(f"  achieved precision    = {branch_summary['achieved_precision']:.4f}  "
          f"(target {branch_summary['target_precision']})")
    print(f"  achieved recall       = {branch_summary['achieved_recall']:.4f}")
    print(f"  flagged {branch_summary['n_branches_flagged']:,}/{branch_summary['n_branches_total']:,} branches "
          f"(coverage {branch_summary['coverage']*100:.1f}% auto-accepted)")

    print()
    print("[3] Per-cell flag calibration")
    cell_mean_th, cell_frac_th, cell_summary = sweep_cell_threshold(
        cell_rows, args.bad_cell_f1_threshold, args.target_precision,
    )
    print(f"  cell_mean_threshold       = {cell_mean_th:.4f}")
    print(f"  cell_low_fraction_thresh  = {cell_frac_th:.4f} (fixed)")
    print(f"  bad-cell definition       = neurite_f1 < {args.bad_cell_f1_threshold}")
    print(f"  achieved precision        = {cell_summary['achieved_precision']:.4f}  "
          f"(target {cell_summary['target_precision']})")
    print(f"  achieved recall           = {cell_summary['achieved_recall']:.4f}")
    print(f"  flagged {cell_summary['n_cells_flagged']}/{cell_summary['n_cells_total']} cells "
          f"(coverage {cell_summary['coverage']*100:.1f}% auto-accepted)")

    # Build + save ConfidenceConfig
    cfg = ConfidenceConfig(
        node_low_threshold=0.70,
        branch_flag_threshold=branch_threshold,
        cell_mean_threshold=cell_mean_th,
        cell_low_fraction_threshold=cell_frac_th,
        stage1_low_threshold=0.60,
        calibration_id=time.strftime("v12_%Y%m%d_%H%M%S"),
        target_precision=args.target_precision,
        note=f"Calibrated on v12 test split. bad_cell_f1_threshold={args.bad_cell_f1_threshold}.",
    )
    cfg.save(flag_config_out)
    print(f"\nSaved flag config -> {flag_config_out}")

    # Save full curves for later inspection
    summary_json.write_text(json.dumps({
        "target_precision":      args.target_precision,
        "bad_cell_f1_threshold": args.bad_cell_f1_threshold,
        "model_dir":             str(args.model_dir),
        "tag":                   tag,
        "gnn_model":             str(gnn_path) if gnn_path else None,
        "n_test_cells":          len(cell_rows),
        "n_test_branches":       len(branch_rows),
        "config":                asdict(cfg),
        "branch_summary":        branch_summary,
        "cell_summary":          cell_summary,
    }, indent=2), encoding="utf-8")
    print(f"Summary  -> {summary_json}")

    # Optional: dump per-branch/per-cell rows for offline analysis
    curves_csv.parent.mkdir(parents=True, exist_ok=True)
    with curves_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(branch_rows[0].keys()) if branch_rows else [])
        w.writeheader()
        for r in branch_rows:
            w.writerow(r)
    print(f"Per-branch rows -> {curves_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
