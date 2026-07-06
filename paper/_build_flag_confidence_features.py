#!/usr/bin/env python3
"""Build deployment-safe confidence/geometry features for the flag model.

This reruns the full v12 pipeline on rows from ``heldout_per_cell_f1.csv`` and
stores per-cell summaries derived only from predictions, confidences, and raw
coordinates. It does not use GT neurite labels as features.

Usage:
    python -m paper._build_flag_confidence_features --max-cells 20
    python -m paper._build_flag_confidence_features

Output:
    paper/results/flag_confidence_features.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.confidence import ConfidenceConfig, summarize_confidence  # noqa: E402
from hybrid.features import parse_swc  # noqa: E402
from hybrid.pipeline import run_pipeline_on_nodes  # noqa: E402
from paper.gnn_branch3_inference import load_branch3  # noqa: E402
from paper.gnn_inference import load_gnn  # noqa: E402

HELDOUT_CSV = ROOT / "paper" / "results" / "heldout_per_cell_f1.csv"
QC_CSV = ROOT / "paper" / "results" / "corpus_qc_v12_uncurated.csv"
OUT_CSV = ROOT / "paper" / "results" / "flag_confidence_features.csv"
SEEDS = [42, 789]


def _load_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _load_qc_paths() -> dict[str, Path]:
    out: dict[str, Path] = {}
    with QC_CSV.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("qc_pass", "").strip().lower() not in {"true", "1"}:
                continue
            p = Path(row["path"])
            if p.is_file():
                out[p.name] = p
    return out


def _load_models() -> dict[int, dict]:
    models: dict[int, dict] = {}
    for seed in SEEDS:
        d = ROOT / "paper" / "models" / f"v12_gentle_seed{seed}"
        s1 = d / "cell_type_classifier.pkl"
        s2 = d / "branch_classifier.pkl"
        gnn = d / "gnn_apical_basal.pt"
        branch3 = d / "gnn_branch3_rescue.pt"
        for p in (s1, s2, gnn):
            if not p.is_file():
                raise SystemExit(f"MISSING: {p}")
        models[seed] = {
            "stage1_model": s1,
            "stage2_model": s2,
            "gnn_state": load_gnn(gnn),
            "branch3_state": load_branch3(branch3) if branch3.is_file() else None,
        }
    return models


def _seeds_for_row(row: dict) -> list[int]:
    raw = str(row.get("seed_used", "")).strip().lower()
    if raw == "both":
        return SEEDS[:]
    if raw in {"42", "789"}:
        return [int(raw)]
    # Fallback for older/partial CSVs: use seed 42.
    return [42]


def _combine_seed_predictions(
    per_seed_labels: dict[int, list[int]],
    per_seed_confs: dict[int, list[float]],
) -> tuple[list[int], list[float], dict[str, float]]:
    seeds = list(per_seed_labels)
    if len(seeds) == 1:
        seed = seeds[0]
        return (
            list(per_seed_labels[seed]),
            [float(c) for c in per_seed_confs[seed]],
            {
                "node_label_disagree_frac": 0.0,
                "node_conf_abs_delta_mean": 0.0,
            },
        )

    a, b = seeds[:2]
    la = per_seed_labels[a]
    lb = per_seed_labels[b]
    ca = per_seed_confs[a]
    cb = per_seed_confs[b]
    labels: list[int] = []
    confs: list[float] = []
    disagree = 0
    deltas: list[float] = []
    for ya, yb, pa, pb in zip(la, lb, ca, cb):
        pa = float(pa)
        pb = float(pb)
        deltas.append(abs(pa - pb))
        if ya == yb:
            labels.append(int(ya))
            confs.append((pa + pb) / 2.0)
        else:
            disagree += 1
            if pa >= pb:
                labels.append(int(ya))
                confs.append(pa)
            else:
                labels.append(int(yb))
                confs.append(pb)
    n = max(1, len(labels))
    return (
        labels,
        confs,
        {
            "node_label_disagree_frac": disagree / n,
            "node_conf_abs_delta_mean": float(np.mean(deltas)) if deltas else 0.0,
        },
    )


def _safe_percentile(values: list[float], q: float, default: float = 0.0) -> float:
    return float(np.percentile(values, q)) if values else default


def _confidence_summary(labels: list[int], confs: list[float]) -> dict[str, float]:
    arr = np.asarray(confs, dtype=float)
    out: dict[str, float] = {
        "conf_mean": float(arr.mean()) if arr.size else 0.0,
        "conf_std": float(arr.std()) if arr.size else 0.0,
        "conf_min": float(arr.min()) if arr.size else 0.0,
        "conf_p05": float(np.percentile(arr, 5)) if arr.size else 0.0,
        "conf_p10": float(np.percentile(arr, 10)) if arr.size else 0.0,
        "conf_p25": float(np.percentile(arr, 25)) if arr.size else 0.0,
        "conf_median": float(np.median(arr)) if arr.size else 0.0,
    }
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        key = str(thr).replace(".", "")
        out[f"conf_frac_lt_{key}"] = float((arr < thr).mean()) if arr.size else 0.0

    label_names = {2: "axon", 3: "basal", 4: "apical"}
    lab_arr = np.asarray(labels, dtype=int)
    for label, name in label_names.items():
        vals = arr[lab_arr == label]
        out[f"{name}_conf_mean"] = float(vals.mean()) if vals.size else 0.0
        out[f"{name}_conf_p10"] = float(np.percentile(vals, 10)) if vals.size else 0.0
        out[f"{name}_conf_frac_lt_07"] = float((vals < 0.7).mean()) if vals.size else 0.0
    return out


def _branch_summary(nodes, labels: list[int], confs: list[float], stage1_type: str, stage1_conf: float) -> dict[str, float]:
    cfg = ConfidenceConfig(node_low_threshold=0.7)
    branches, _cell = summarize_confidence(nodes, labels, confs, stage1_type, stage1_conf, cfg)
    means = [float(b.mean_confidence) for b in branches]
    mins = [float(b.min_confidence) for b in branches]
    n_nodes = [int(b.n_nodes) for b in branches]
    return {
        "n_pred_branches": len(branches),
        "branch_conf_mean": float(np.mean(means)) if means else 0.0,
        "branch_conf_min": min(mins) if mins else 0.0,
        "branch_conf_p10": _safe_percentile(means, 10),
        "branch_frac_conf_lt_07": float(np.mean([m < 0.7 for m in means])) if means else 0.0,
        "branch_frac_conf_lt_08": float(np.mean([m < 0.8 for m in means])) if means else 0.0,
        "branch_n_nodes_p90": _safe_percentile(n_nodes, 90),
    }


def _predicted_geometry(nodes, labels: list[int]) -> dict[str, float]:
    label_names = {2: "axon", 3: "basal", 4: "apical"}
    labels_arr = np.asarray(labels, dtype=int)
    soma_zs = [nd.z for nd, lab in zip(nodes, labels_arr) if lab == 1]
    if not soma_zs:
        soma_zs = [nd.z for nd in nodes if nd.type == 1]
    soma_z = float(np.mean(soma_zs)) if soma_zs else 0.0

    out: dict[str, float] = {"pred_soma_z": soma_z}
    for label, name in label_names.items():
        xs = [nd.x for nd, lab in zip(nodes, labels_arr) if lab == label]
        ys = [nd.y for nd, lab in zip(nodes, labels_arr) if lab == label]
        zs = [nd.z for nd, lab in zip(nodes, labels_arr) if lab == label]
        if zs:
            out[f"pred_{name}_z_extent"] = float(max(zs) - min(zs))
            out[f"pred_{name}_z_mean_rel_soma"] = float(np.mean(zs) - soma_z)
            out[f"pred_{name}_z_std"] = float(np.std(zs))
            out[f"pred_{name}_x_extent"] = float(max(xs) - min(xs))
            out[f"pred_{name}_y_extent"] = float(max(ys) - min(ys))
        else:
            out[f"pred_{name}_z_extent"] = 0.0
            out[f"pred_{name}_z_mean_rel_soma"] = 0.0
            out[f"pred_{name}_z_std"] = 0.0
            out[f"pred_{name}_x_extent"] = 0.0
            out[f"pred_{name}_y_extent"] = 0.0
    return out


def _round_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, float):
            out[k] = f"{v:.6g}"
        else:
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=HELDOUT_CSV)
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--progress-every", type=int, default=100)
    args = ap.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"MISSING: {args.input}")
    rows = _load_csv_rows(args.input)
    if args.max_cells is not None:
        rows = rows[: args.max_cells]
    path_by_file = _load_qc_paths()
    models = _load_models()
    print(f"Loaded {len(rows)} held-out rows and {len(path_by_file)} QC paths")

    out_rows: list[dict] = []
    t0 = time.perf_counter()
    n_failed = 0
    for i, row in enumerate(rows, 1):
        basename = row["file"]
        path = path_by_file.get(basename)
        if path is None:
            n_failed += 1
            continue
        try:
            nodes = parse_swc(path)
            per_seed_labels: dict[int, list[int]] = {}
            per_seed_confs: dict[int, list[float]] = {}
            stage1_preds: list[str] = []
            stage1_confs: list[float] = []
            for seed in _seeds_for_row(row):
                m = models[seed]
                pr = run_pipeline_on_nodes(
                    nodes,
                    file_path="",
                    stage1_model=m["stage1_model"],
                    stage2_model=m["stage2_model"],
                    gnn_state=m["gnn_state"],
                    branch3_state=m.get("branch3_state"),
                    use_subtree_stage2=True,
                )
                per_seed_labels[seed] = list(pr.node_labels)
                per_seed_confs[seed] = [float(c) for c in pr.node_confidences]
                stage1_preds.append(pr.stage1.cell_type)
                stage1_confs.append(float(pr.stage1.confidence))

            labels, confs, seed_agreement = _combine_seed_predictions(per_seed_labels, per_seed_confs)
            pred_counts = Counter(labels)
            stage1_pred = Counter(stage1_preds).most_common(1)[0][0] if stage1_preds else ""
            stage1_conf = float(np.mean(stage1_confs)) if stage1_confs else 0.0

            feat = {
                "file": basename,
                "n_nodes_conf": len(nodes),
                "stage1_pred_conf_run": stage1_pred,
                "stage1_conf_conf_run": stage1_conf,
                "pred_axon_conf_run": pred_counts.get(2, 0),
                "pred_basal_conf_run": pred_counts.get(3, 0),
                "pred_apical_conf_run": pred_counts.get(4, 0),
                **seed_agreement,
                **_confidence_summary(labels, confs),
                **_branch_summary(nodes, labels, confs, stage1_pred, stage1_conf),
                **_predicted_geometry(nodes, labels),
            }
            out_rows.append(_round_row(feat))
        except Exception as exc:
            print(f"  WARN {basename}: {exc}", flush=True)
            n_failed += 1
            continue

        if i % args.progress_every == 0:
            elapsed = (time.perf_counter() - t0) / 60.0
            eta = elapsed * (len(rows) - i) / max(1, i)
            print(f"  ... {i}/{len(rows)} ({elapsed:.1f} min, ETA {eta:.1f} min)", flush=True)

    if not out_rows:
        raise SystemExit("No feature rows produced")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys())
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    elapsed = (time.perf_counter() - t0) / 60.0
    print(f"\nDone: wrote {len(out_rows)} rows to {args.out} ({n_failed} failed, {elapsed:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
