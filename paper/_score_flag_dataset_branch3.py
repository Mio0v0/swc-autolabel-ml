#!/usr/bin/env python3
"""Build flag-model labels/features with the accepted conservative Branch3 labeler.

This scores the current flag-model cell list with a single production-style
model directory, instead of the older seed42/seed789 held-out models. It writes
two deployment-safe inputs for ``paper._train_flag_model``:

- per-cell F1 labels and basic predicted counts
- confidence/geometry/disagreement features

The extra disagreement features compare the accepted Branch3-enabled output
against the same model with Branch3 disabled. They are deployment-safe because
they use only model predictions and raw SWC geometry.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.evaluate import per_cell_neurite_f1  # noqa: E402
from hybrid.features import parse_swc  # noqa: E402
from hybrid.pipeline import run_pipeline_on_nodes  # noqa: E402
from paper._build_flag_confidence_features import (  # noqa: E402
    _branch_summary,
    _confidence_summary,
    _predicted_geometry,
    _round_row,
)
from paper._score_heldout_f1 import _per_class_f1, _structural_features  # noqa: E402
from paper.gnn_branch3_inference import load_branch3  # noqa: E402
from paper.gnn_inference import load_gnn  # noqa: E402


QC_CSV = ROOT / "paper" / "results" / "corpus_qc_v12_uncurated.csv"
INPUT_CSV = ROOT / "paper" / "results" / "heldout_per_cell_f1.csv"
MODEL_DIR = ROOT / "paper" / "models" / "v12_pyramidal_only_seed2024_clean"
OUT_LABELS = ROOT / "paper" / "results" / "heldout_per_cell_f1_branch3.csv"
OUT_FEATURES = ROOT / "paper" / "results" / "flag_confidence_features_branch3.csv"
OUT_JSON = ROOT / "paper" / "results" / "flag_branch3_dataset_summary.json"


def _load_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _load_qc_paths() -> dict[str, tuple[str, Path]]:
    out: dict[str, tuple[str, Path]] = {}
    with QC_CSV.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("qc_pass", "").strip().lower() not in {"true", "1"}:
                continue
            p = Path(row["path"])
            if p.is_file():
                out[p.name] = (row["cell_type"], p)
    return out


def _load_model_dir(model_dir: Path) -> dict:
    s1 = model_dir / "cell_type_classifier.pkl"
    s2 = model_dir / "branch_classifier.pkl"
    gnn = model_dir / "gnn_apical_basal.pt"
    branch3 = model_dir / "gnn_branch3_rescue.pt"
    for p in (s1, s2, gnn, branch3):
        if not p.is_file():
            raise SystemExit(f"MISSING: {p}")
    return {
        "stage1_model": s1,
        "stage2_model": s2,
        "gnn_state": load_gnn(gnn),
        "branch3_state": load_branch3(branch3),
    }


def _branch3_disagreement_features(
    final_labels: list[int],
    final_confs: list[float],
    base_labels: list[int] | None,
    base_confs: list[float] | None,
) -> dict[str, float]:
    n = max(1, len(final_labels))
    if base_labels is None or base_confs is None:
        return {
            "branch3_changed_frac": 0.0,
            "branch3_label_disagree_frac": 0.0,
            "branch3_conf_abs_delta_mean": 0.0,
            "branch3_axon_delta_frac": 0.0,
            "branch3_basal_delta_frac": 0.0,
            "branch3_apical_delta_frac": 0.0,
            "branch3_to_apical_frac": 0.0,
            "branch3_from_apical_frac": 0.0,
        }

    final = np.asarray(final_labels, dtype=int)
    base = np.asarray(base_labels, dtype=int)
    final_c = np.asarray(final_confs, dtype=float)
    base_c = np.asarray(base_confs, dtype=float)
    changed = final != base
    return {
        "branch3_changed_frac": float(changed.mean()) if changed.size else 0.0,
        "branch3_label_disagree_frac": float(changed.mean()) if changed.size else 0.0,
        "branch3_conf_abs_delta_mean": float(np.mean(np.abs(final_c - base_c))) if final_c.size else 0.0,
        "branch3_axon_delta_frac": float(((final == 2).sum() - (base == 2).sum()) / n),
        "branch3_basal_delta_frac": float(((final == 3).sum() - (base == 3).sum()) / n),
        "branch3_apical_delta_frac": float(((final == 4).sum() - (base == 4).sum()) / n),
        "branch3_to_apical_frac": float(((base != 4) & (final == 4)).sum() / n),
        "branch3_from_apical_frac": float(((base == 4) & (final != 4)).sum() / n),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=INPUT_CSV)
    ap.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    ap.add_argument("--out-labels", type=Path, default=OUT_LABELS)
    ap.add_argument("--out-features", type=Path, default=OUT_FEATURES)
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--progress-every", type=int, default=100)
    args = ap.parse_args()

    rows = _load_csv_rows(args.input)
    if args.max_cells is not None:
        rows = rows[: args.max_cells]
    path_by_file = _load_qc_paths()
    model = _load_model_dir(args.model_dir)
    print(f"Loaded {len(rows)} input rows and {len(path_by_file)} QC paths")
    print(f"Model: {args.model_dir}")

    label_rows: list[dict] = []
    feature_rows: list[dict] = []
    n_failed = 0
    t0 = time.perf_counter()

    for i, row in enumerate(rows, 1):
        basename = row["file"]
        item = path_by_file.get(basename)
        if item is None:
            n_failed += 1
            continue
        ct_gt, path = item
        cell_t0 = time.perf_counter()
        try:
            nodes = parse_swc(path)
            gt = [n.type for n in nodes]
            pr = run_pipeline_on_nodes(
                nodes,
                file_path="",
                stage1_model=model["stage1_model"],
                stage2_model=model["stage2_model"],
                gnn_state=model["gnn_state"],
                branch3_state=model["branch3_state"],
                use_subtree_stage2=True,
            )
            labels = list(pr.node_labels)
            confs = [float(c) for c in pr.node_confidences]

            base_labels: list[int] | None = None
            base_confs: list[float] | None = None
            if pr.stage1.cell_type == "pyramidal":
                base_pr = run_pipeline_on_nodes(
                    nodes,
                    file_path="",
                    stage1_model=model["stage1_model"],
                    stage2_model=model["stage2_model"],
                    gnn_state=model["gnn_state"],
                    branch3_state=None,
                    use_subtree_stage2=True,
                )
                base_labels = list(base_pr.node_labels)
                base_confs = [float(c) for c in base_pr.node_confidences]

            pc_f1 = per_cell_neurite_f1(gt, labels, ct_gt)
            per_class = _per_class_f1(gt, labels, ct_gt)
            n_correct = sum(1 for g, q in zip(gt, labels) if g == q)
            pc_acc = n_correct / max(1, len(gt))
            structural = _structural_features(nodes)
            pred_counter = Counter(labels)
            elapsed_s = time.perf_counter() - cell_t0

            label_rows.append(
                {
                    "file": basename,
                    "cell_type_gt": ct_gt,
                    "n_nodes": len(gt),
                    "seed_used": "branch3",
                    "held_out_F1": f"{pc_f1:.4f}",
                    "acc": f"{pc_acc:.4f}",
                    "axon_F1": f"{per_class['axon_f1']:.4f}" if not np.isnan(per_class["axon_f1"]) else "",
                    "basal_F1": f"{per_class['basal_f1']:.4f}" if not np.isnan(per_class["basal_f1"]) else "",
                    "apical_F1": f"{per_class['apical_f1']:.4f}" if not np.isnan(per_class["apical_f1"]) else "",
                    "stage1_pred": pr.stage1.cell_type,
                    "stage1_conf": f"{float(pr.stage1.confidence):.4f}",
                    "stage1_correct": str(pr.stage1.cell_type == ct_gt),
                    "n_components": int(structural.get("n_components", 1)),
                    "soma_z": f"{structural.get('soma_z', 0):.2f}" if not np.isnan(structural.get("soma_z", float("nan"))) else "",
                    "apical_mean_z": f"{structural.get('apical_mean_z', 0):.2f}" if not np.isnan(structural.get("apical_mean_z", float("nan"))) else "",
                    "apical_above_soma": f"{structural.get('apical_above_soma', 0):.2f}" if not np.isnan(structural.get("apical_above_soma", float("nan"))) else "",
                    "apical_z_extent": f"{structural.get('apical_z_extent', 0):.2f}",
                    "axon_frac": f"{structural.get('axon_frac', 0):.3f}",
                    "apical_frac": f"{structural.get('apical_frac', 0):.3f}",
                    "basal_frac": f"{structural.get('basal_frac', 0):.3f}",
                    "pred_axon": pred_counter.get(2, 0),
                    "pred_basal": pred_counter.get(3, 0),
                    "pred_apical": pred_counter.get(4, 0),
                    "elapsed_s": f"{elapsed_s:.2f}",
                    "source": basename.split("__", 1)[0] if "__" in basename else "",
                }
            )

            feature = {
                "file": basename,
                "n_nodes_conf": len(nodes),
                "stage1_pred_conf_run": pr.stage1.cell_type,
                "stage1_conf_conf_run": float(pr.stage1.confidence),
                "pred_axon_conf_run": pred_counter.get(2, 0),
                "pred_basal_conf_run": pred_counter.get(3, 0),
                "pred_apical_conf_run": pred_counter.get(4, 0),
                "node_label_disagree_frac": 0.0,
                "node_conf_abs_delta_mean": 0.0,
                **_branch3_disagreement_features(labels, confs, base_labels, base_confs),
                **_confidence_summary(labels, confs),
                **_branch_summary(nodes, labels, confs, pr.stage1.cell_type, float(pr.stage1.confidence)),
                **_predicted_geometry(nodes, labels),
            }
            feature_rows.append(_round_row(feature))
        except Exception as exc:
            print(f"  WARN {basename}: {exc}", flush=True)
            n_failed += 1
            continue

        if i % args.progress_every == 0:
            elapsed = (time.perf_counter() - t0) / 60.0
            eta = elapsed * (len(rows) - i) / max(1, i)
            print(f"  ... {i}/{len(rows)} ({elapsed:.1f} min, ETA {eta:.1f} min)", flush=True)

    if not label_rows or not feature_rows:
        raise SystemExit("No rows produced")

    args.out_labels.parent.mkdir(parents=True, exist_ok=True)
    with args.out_labels.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(label_rows[0].keys()))
        writer.writeheader()
        writer.writerows(label_rows)
    with args.out_features.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(feature_rows[0].keys()))
        writer.writeheader()
        writer.writerows(feature_rows)

    f1s = np.asarray([float(r["held_out_F1"]) for r in label_rows], dtype=float)
    summary = {
        "model_dir": str(args.model_dir.resolve().relative_to(ROOT)),
        "input": str(args.input.resolve().relative_to(ROOT)),
        "out_labels": str(args.out_labels.resolve().relative_to(ROOT)),
        "out_features": str(args.out_features.resolve().relative_to(ROOT)),
        "n_scored": len(label_rows),
        "n_failed": n_failed,
        "elapsed_min": (time.perf_counter() - t0) / 60.0,
        "f1_mean": float(f1s.mean()),
        "f1_p10": float(np.percentile(f1s, 10)),
        "bad_rate_f050": float((f1s < 0.5).mean()),
        "bad_rate_f060": float((f1s < 0.6).mean()),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDone: {len(label_rows)} scored, {n_failed} failed")
    print(f"Wrote {args.out_labels}")
    print(f"Wrote {args.out_features}")
    print(f"Wrote {OUT_JSON}")
    print(
        f"F1 mean={summary['f1_mean']:.4f} P10={summary['f1_p10']:.4f} "
        f"bad<.5={summary['bad_rate_f050']:.3f} bad<.6={summary['bad_rate_f060']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
