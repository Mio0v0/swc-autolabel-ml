#!/usr/bin/env python3
"""Build multi-v12 disagreement features for the per-cell flag model.

The canonical labeler is seed123+Branch3. This script runs other v12 seeds on
the same cells and records deployment-safe disagreement features, then merges
them into the existing flag feature CSV.
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
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.features import parse_swc  # noqa: E402
from hybrid.pipeline import run_pipeline_on_nodes  # noqa: E402
from paper.gnn_branch3_inference import load_branch3  # noqa: E402
from paper.gnn_inference import load_gnn  # noqa: E402

QC_CSV = ROOT / "paper" / "results" / "corpus_qc_v12_uncurated.csv"
DEFAULT_LABELS = ROOT / "paper" / "results" / "heldout_per_cell_f1_seed123_branch3.csv"
DEFAULT_BASE_FEATURES = ROOT / "paper" / "results" / "flag_features_seed123_branch3_xmodel.csv"
DEFAULT_OUT_V12ENS = ROOT / "paper" / "results" / "flag_v12_ensemble_features_seed123_branch3.csv"
DEFAULT_OUT_FEATURES = ROOT / "paper" / "results" / "flag_features_seed123_branch3_xmodel_v12ens.csv"
SEEDS = (42, 789)


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


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


def _load_model(seed: int) -> dict:
    model_dir = ROOT / "paper" / "models" / f"v12_gentle_seed{seed}"
    s1 = model_dir / "cell_type_classifier.pkl"
    s2 = model_dir / "branch_classifier.pkl"
    gnn = model_dir / "gnn_apical_basal.pt"
    for p in (s1, s2, gnn):
        if not p.is_file():
            raise SystemExit(f"MISSING: {p}")
    b3 = model_dir / "gnn_branch3_rescue.pt"
    return {
        "seed": seed,
        "stage1_model": s1,
        "stage2_model": s2,
        "gnn_state": load_gnn(gnn),
        "branch3_state": load_branch3(b3) if b3.is_file() else None,
    }


def _frac_counts(labels: list[int], n_nodes: int) -> dict[str, float]:
    denom = max(1.0, float(n_nodes - 1))
    counts = Counter(int(x) for x in labels)
    return {
        "axon": counts.get(2, 0) / denom,
        "basal": counts.get(3, 0) / denom,
        "apical": counts.get(4, 0) / denom,
    }


def _safe_std(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(arr.std()) if arr.size else 0.0


def _build_features(row: dict, nodes: list, models: list[dict]) -> dict:
    n_nodes = int(row["n_nodes"])
    denom = max(1.0, float(n_nodes - 1))
    v12 = {
        "axon": float(row.get("pred_axon", 0) or 0) / denom,
        "basal": float(row.get("pred_basal", 0) or 0) / denom,
        "apical": float(row.get("pred_apical", 0) or 0) / denom,
    }
    v12_stage1 = str(row.get("stage1_pred", ""))
    out: dict[str, float] = {}
    l1s: list[float] = []
    ap_deltas: list[float] = []
    ax_deltas: list[float] = []
    ap_present: list[float] = []
    ax_present: list[float] = []
    class_counts: list[float] = []
    stage1_is_pyr: list[float] = []
    stage1_disagree: list[float] = []
    stage1_confs: list[float] = []
    seed_fracs: list[dict[str, float]] = []

    for model in models:
        seed = int(model["seed"])
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
        frac = _frac_counts(labels, n_nodes)
        seed_fracs.append(frac)
        prefix = f"xmodel_v12ens_seed{seed}"
        class_count = float(sum(1 for cls in ("axon", "basal", "apical") if frac[cls] > 0.0))
        l1 = float(sum(abs(frac[cls] - v12[cls]) for cls in ("axon", "basal", "apical")))
        ap_delta = float(abs(frac["apical"] - v12["apical"]))
        ax_delta = float(abs(frac["axon"] - v12["axon"]))
        out[f"{prefix}_axon_frac"] = frac["axon"]
        out[f"{prefix}_basal_frac"] = frac["basal"]
        out[f"{prefix}_apical_frac"] = frac["apical"]
        out[f"{prefix}_class_count"] = class_count
        out[f"{prefix}_v12_l1_frac_delta"] = l1
        out[f"{prefix}_v12_apical_frac_delta"] = ap_delta
        out[f"{prefix}_v12_axon_frac_delta"] = ax_delta
        out[f"{prefix}_v12_apical_zero_mismatch"] = float((frac["apical"] == 0.0) != (v12["apical"] == 0.0))
        out[f"{prefix}_v12_axon_zero_mismatch"] = float((frac["axon"] == 0.0) != (v12["axon"] == 0.0))
        out[f"{prefix}_stage1_is_pyramidal"] = float(pr.stage1.cell_type == "pyramidal")
        out[f"{prefix}_stage1_disagree"] = float(str(pr.stage1.cell_type) != v12_stage1)
        out[f"{prefix}_stage1_conf"] = float(pr.stage1.confidence)
        l1s.append(l1)
        ap_deltas.append(ap_delta)
        ax_deltas.append(ax_delta)
        ap_present.append(float(frac["apical"] > 0.0))
        ax_present.append(float(frac["axon"] > 0.0))
        class_counts.append(class_count)
        stage1_is_pyr.append(float(pr.stage1.cell_type == "pyramidal"))
        stage1_disagree.append(float(str(pr.stage1.cell_type) != v12_stage1))
        stage1_confs.append(float(pr.stage1.confidence))

    out["xmodel_v12ens_l1_mean"] = float(np.mean(l1s)) if l1s else 0.0
    out["xmodel_v12ens_l1_max"] = float(np.max(l1s)) if l1s else 0.0
    out["xmodel_v12ens_l1_std"] = _safe_std(l1s)
    out["xmodel_v12ens_apical_delta_mean"] = float(np.mean(ap_deltas)) if ap_deltas else 0.0
    out["xmodel_v12ens_apical_delta_max"] = float(np.max(ap_deltas)) if ap_deltas else 0.0
    out["xmodel_v12ens_axon_delta_mean"] = float(np.mean(ax_deltas)) if ax_deltas else 0.0
    out["xmodel_v12ens_axon_delta_max"] = float(np.max(ax_deltas)) if ax_deltas else 0.0
    out["xmodel_v12ens_apical_present_vote_frac"] = float(np.mean(ap_present)) if ap_present else 0.0
    out["xmodel_v12ens_axon_present_vote_frac"] = float(np.mean(ax_present)) if ax_present else 0.0
    out["xmodel_v12ens_apical_presence_disagreement"] = float(np.max(ap_present) - np.min(ap_present)) if ap_present else 0.0
    out["xmodel_v12ens_axon_presence_disagreement"] = float(np.max(ax_present) - np.min(ax_present)) if ax_present else 0.0
    out["xmodel_v12ens_class_count_mean"] = float(np.mean(class_counts)) if class_counts else 0.0
    out["xmodel_v12ens_class_count_std"] = _safe_std(class_counts)
    out["xmodel_v12ens_stage1_pyramidal_vote_frac"] = float(np.mean(stage1_is_pyr)) if stage1_is_pyr else 0.0
    out["xmodel_v12ens_stage1_disagree_mean"] = float(np.mean(stage1_disagree)) if stage1_disagree else 0.0
    out["xmodel_v12ens_stage1_conf_mean"] = float(np.mean(stage1_confs)) if stage1_confs else 0.0
    out["xmodel_v12ens_stage1_conf_min"] = float(np.min(stage1_confs)) if stage1_confs else 0.0
    if len(seed_fracs) >= 2:
        out["xmodel_v12ens_interseed_l1"] = float(
            sum(abs(seed_fracs[0][cls] - seed_fracs[1][cls]) for cls in ("axon", "basal", "apical"))
        )
        out["xmodel_v12ens_interseed_apical_delta"] = float(abs(seed_fracs[0]["apical"] - seed_fracs[1]["apical"]))
        out["xmodel_v12ens_interseed_axon_delta"] = float(abs(seed_fracs[0]["axon"] - seed_fracs[1]["axon"]))
    else:
        out["xmodel_v12ens_interseed_l1"] = 0.0
        out["xmodel_v12ens_interseed_apical_delta"] = 0.0
        out["xmodel_v12ens_interseed_axon_delta"] = 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--base-features", type=Path, default=DEFAULT_BASE_FEATURES)
    ap.add_argument("--out-v12ens", type=Path, default=DEFAULT_OUT_V12ENS)
    ap.add_argument("--out-features", type=Path, default=DEFAULT_OUT_FEATURES)
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--progress-every", type=int, default=50)
    args = ap.parse_args()

    labels = pd.read_csv(args.labels)
    if args.max_cells is not None:
        labels = labels.head(args.max_cells).copy()
    qc_paths = _load_qc_paths()
    print("Loading v12 ensemble models...", flush=True)
    models = [_load_model(seed) for seed in SEEDS]
    print(f"Building multi-v12 features for {len(labels)} rows", flush=True)

    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, row in enumerate(labels.to_dict("records"), 1):
        p = qc_paths.get(str(row["file"]))
        if p is None:
            raise SystemExit(f"Missing path for {row['file']}")
        nodes = parse_swc(p)
        feat = {"file": row["file"]}
        feat.update(_build_features(row, nodes, models))
        rows.append(feat)
        if i % args.progress_every == 0:
            elapsed = (time.perf_counter() - t0) / 60.0
            eta = elapsed * (len(labels) - i) / max(1, i)
            print(f"  ... {i}/{len(labels)} ({elapsed:.1f} min, ETA {eta:.1f} min)", flush=True)

    vdf = pd.DataFrame(rows)
    args.out_v12ens.parent.mkdir(parents=True, exist_ok=True)
    vdf.to_csv(args.out_v12ens, index=False)
    base = pd.read_csv(args.base_features)
    merged = base.merge(vdf, on="file", how="left")
    merged.to_csv(args.out_features, index=False)
    summary = {
        "labels": _display(args.labels),
        "base_features": _display(args.base_features),
        "out_v12ens": _display(args.out_v12ens),
        "out_features": _display(args.out_features),
        "n_rows": int(len(merged)),
        "n_v12ens_columns": int(len([c for c in merged.columns if c.startswith("xmodel_v12ens_")])),
        "elapsed_min": (time.perf_counter() - t0) / 60.0,
        "seeds": list(SEEDS),
    }
    args.out_features.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
