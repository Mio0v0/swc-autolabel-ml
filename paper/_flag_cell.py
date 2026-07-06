#!/usr/bin/env python3
"""Score one SWC with the learned per-cell flag model.

Run with the project venv because v12 GNN inference needs torch_geometric:

    D:/Desktop/SWC-Studio/.venv/Scripts/python.exe -m paper._flag_cell path/to/cell.swc
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.cell_type_detector import detect_cell_type_from_nodes  # noqa: E402
from hybrid.features import parse_swc  # noqa: E402
from hybrid.pipeline import run_pipeline_on_nodes  # noqa: E402
from paper._build_flag_confidence_features import (  # noqa: E402
    _branch_summary,
    _confidence_summary,
    _predicted_geometry,
)
from paper._build_flag_cross_model_features import (  # noqa: E402
    BASELINE_CACHE,
    METHODS,
    _baseline_features,
)
from paper._build_flag_v12_ensemble_features import (  # noqa: E402
    SEEDS as V12ENS_SEEDS,
    _build_features as _v12ens_features,
    _load_model as _load_v12ens_model,
)
from paper._eval_baselines_on_v12 import _load_split_from_qc  # noqa: E402
from paper._score_flag_dataset_branch3 import _branch3_disagreement_features  # noqa: E402
from paper._train_flag_model import _engineer_features  # noqa: E402
from paper.external_baselines import predict_with_cache  # noqa: E402
from paper.gnn_branch3_inference import load_branch3  # noqa: E402
from paper.gnn_inference import load_gnn  # noqa: E402

DEFAULT_FLAG_MODEL = ROOT / "paper" / "models" / "flag_model_seed123_branch3_pyramidal" / "flag_model_f060.joblib"
DEFAULT_V12_DIR = ROOT / "paper" / "models" / "v12_gentle_seed123"
DEFAULT_SECOND_STAGE1_DIR = None
DEFAULT_BASELINE_CACHE = BASELINE_CACHE
DEFAULT_BASELINE_SEED = 123
DEFAULT_V12_ENSEMBLE_SEEDS = ",".join(str(s) for s in V12ENS_SEEDS)


def _n_components(nodes) -> int:
    n = len(nodes)
    if n == 0:
        return 0
    id_to_idx = {nd.id: i for i, nd in enumerate(nodes)}
    parent_idx = [-1] * n
    children: list[list[int]] = [[] for _ in nodes]
    for i, nd in enumerate(nodes):
        if nd.parent != -1 and nd.parent in id_to_idx:
            p = id_to_idx[nd.parent]
            parent_idx[i] = p
            children[p].append(i)

    visited = [False] * n
    n_comp = 0
    for start in range(n):
        if visited[start]:
            continue
        n_comp += 1
        stack = [start]
        while stack:
            i = stack.pop()
            if visited[i]:
                continue
            visited[i] = True
            p = parent_idx[i]
            if p >= 0 and not visited[p]:
                stack.append(p)
            stack.extend(j for j in children[i] if not visited[j])
    return n_comp


def _source_from_name(path: Path) -> str:
    name = path.name
    return name.split("__", 1)[0] if "__" in name else ""


def _build_feature_row(swc_path: Path, v12_dir: Path, second_stage1_dir: Path | None) -> dict:
    s1_path = v12_dir / "cell_type_classifier.pkl"
    s2_path = v12_dir / "branch_classifier.pkl"
    gnn_path = v12_dir / "gnn_apical_basal.pt"
    branch3_path = v12_dir / "gnn_branch3_rescue.pt"
    for p in (s1_path, s2_path, gnn_path):
        if not p.is_file():
            raise SystemExit(f"MISSING: {p}")

    nodes = parse_swc(swc_path)
    gnn_state = load_gnn(gnn_path)
    branch3_state = load_branch3(branch3_path) if branch3_path.is_file() else None
    pr = run_pipeline_on_nodes(
        nodes,
        file_path="",
        stage1_model=s1_path,
        stage2_model=s2_path,
        gnn_state=gnn_state,
        branch3_state=branch3_state,
        use_subtree_stage2=True,
    )
    labels = list(pr.node_labels)
    confs = [float(c) for c in pr.node_confidences]
    counts = Counter(labels)
    base_labels: list[int] | None = None
    base_confs: list[float] | None = None
    if branch3_state is not None and pr.stage1.cell_type == "pyramidal":
        base_pr = run_pipeline_on_nodes(
            nodes,
            file_path="",
            stage1_model=s1_path,
            stage2_model=s2_path,
            gnn_state=gnn_state,
            branch3_state=None,
            use_subtree_stage2=True,
        )
        base_labels = list(base_pr.node_labels)
        base_confs = [float(c) for c in base_pr.node_confidences]

    # Cross-seed Stage 1 features are cheap and were part of training.
    seed42 = detect_cell_type_from_nodes(nodes, s1_path)
    if second_stage1_dir is not None:
        seed789_path = second_stage1_dir / "cell_type_classifier.pkl"
        seed789 = detect_cell_type_from_nodes(nodes, seed789_path) if seed789_path.is_file() else seed42
    else:
        seed789 = seed42

    row = {
        "file": swc_path.name,
        "n_nodes": len(nodes),
        "stage1_pred": pr.stage1.cell_type,
        "stage1_conf": float(pr.stage1.confidence),
        "n_components": _n_components(nodes),
        "pred_axon": counts.get(2, 0),
        "pred_basal": counts.get(3, 0),
        "pred_apical": counts.get(4, 0),
        "source": _source_from_name(swc_path),
        "seed42_pred": seed42.cell_type,
        "seed42_conf": float(seed42.confidence),
        "seed789_pred": seed789.cell_type,
        "seed789_conf": float(seed789.confidence),
        # Present only so _engineer_features can run on the same shape. It is
        # not used as a feature.
        "held_out_F1": np.nan,
    }
    row.update(
        {
            "n_nodes_conf": len(nodes),
            "stage1_pred_conf_run": pr.stage1.cell_type,
            "stage1_conf_conf_run": float(pr.stage1.confidence),
            "pred_axon_conf_run": counts.get(2, 0),
            "pred_basal_conf_run": counts.get(3, 0),
            "pred_apical_conf_run": counts.get(4, 0),
            "node_label_disagree_frac": 0.0,
            "node_conf_abs_delta_mean": 0.0,
            "confidence_features_missing": 0,
        }
    )
    row.update(_confidence_summary(labels, confs))
    row.update(_branch_summary(nodes, labels, confs, pr.stage1.cell_type, float(pr.stage1.confidence)))
    row.update(_branch3_disagreement_features(labels, confs, base_labels, base_confs))
    row.update(_predicted_geometry(nodes, labels))
    return row


def _needs_xmodel_features(bundle: dict) -> bool:
    return any(
        str(c).startswith("xmodel_")
        for c in bundle.get("numeric_features", [])
    )


def _needs_prefix(bundle: dict, prefix: str) -> bool:
    return any(str(c).startswith(prefix) for c in bundle.get("numeric_features", []))


def _parse_seed_list(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _add_xmodel_features(
    row: dict,
    swc_path: Path,
    baseline_cache_dir: Path,
    baseline_seed: int,
) -> None:
    nodes = parse_swc(swc_path)
    train, _test = _load_split_from_qc(baseline_seed)
    predictors = {
        method: predict_with_cache(
            method,
            train,
            seed=baseline_seed,
            cache_path=baseline_cache_dir / f"{method}.pkl",
            force_retrain=False,
        )
        for method in METHODS
    }
    row.update(_baseline_features(row, nodes, predictors))


def _add_v12ens_features(row: dict, swc_path: Path, seeds: list[int]) -> None:
    nodes = parse_swc(swc_path)
    models = [_load_v12ens_model(seed) for seed in seeds]
    row.update(_v12ens_features(row, nodes, models))


def _add_sanity_features(row: dict, claimed_cell_type: str | None) -> None:
    claimed = claimed_cell_type or str(row.get("stage1_pred", ""))
    stage1 = str(row.get("stage1_pred", ""))
    n_nodes = float(row.get("n_nodes", 0) or 0)
    denom = max(1.0, n_nodes - 1.0)
    pred_axon = float(row.get("pred_axon", 0) or 0)
    pred_basal = float(row.get("pred_basal", 0) or 0)
    pred_apical = float(row.get("pred_apical", 0) or 0)
    ax_frac = pred_axon / denom
    ba_frac = pred_basal / denom
    ap_frac = pred_apical / denom
    conf_p10 = float(row.get("conf_p10", np.nan))
    stage1_conf = float(row.get("stage1_conf", np.nan))
    x_pyr_l1 = float(row.get("xmodel_pyr_l1_mean", 0.0) or 0.0)
    x_s1_l1 = float(row.get("xmodel_s1_l1_mean", 0.0) or 0.0)
    x_neurom_pyr = float(row.get("xmodel_neurom_rf_pyr_v12_l1_frac_delta", 0.0) or 0.0)

    row.update(
        {
            "xmodel_sanity_claimed_is_pyramidal": float(claimed == "pyramidal"),
            "xmodel_sanity_claimed_is_interneuron": float(claimed == "interneuron"),
            "xmodel_sanity_stage1_claim_mismatch": float(claimed != stage1),
            "xmodel_sanity_claimed_pyr_stage1_int": float(
                claimed == "pyramidal" and stage1 == "interneuron"
            ),
            "xmodel_sanity_claimed_int_stage1_pyr": float(
                claimed == "interneuron" and stage1 == "pyramidal"
            ),
            "xmodel_sanity_pred_no_axon": float(pred_axon <= 0),
            "xmodel_sanity_pred_no_basal": float(pred_basal <= 0),
            "xmodel_sanity_pred_no_apical": float(pred_apical <= 0),
            "xmodel_sanity_pred_single_neurite_class": float(
                sum(v > 0 for v in (pred_axon, pred_basal, pred_apical)) <= 1
            ),
            "xmodel_sanity_claimed_pyr_no_apical": float(
                claimed == "pyramidal" and pred_apical <= 0
            ),
            "xmodel_sanity_claimed_int_no_axon": float(
                claimed == "interneuron" and pred_axon <= 0
            ),
            "xmodel_sanity_claimed_pyr_tiny_apical": float(
                claimed == "pyramidal" and ap_frac < 0.01
            ),
            "xmodel_sanity_claimed_int_tiny_axon": float(
                claimed == "interneuron" and ax_frac < 0.01
            ),
            "xmodel_sanity_pred_axon_frac": ax_frac,
            "xmodel_sanity_pred_basal_frac": ba_frac,
            "xmodel_sanity_pred_apical_frac": ap_frac,
            "xmodel_sanity_low_conf_tail": float(conf_p10 < 0.7),
            "xmodel_sanity_low_stage1_conf": float(stage1_conf < 0.8),
            "xmodel_sanity_xmodel_delta_high": float(x_pyr_l1 >= 0.2),
            "xmodel_sanity_neurom_delta_high": float(x_neurom_pyr >= 0.5),
            "xmodel_sanity_low_conf_x_delta": float(conf_p10 < 0.9 and x_pyr_l1 >= 0.2),
            "xmodel_sanity_stage1_or_x_delta": float(
                claimed != stage1 or x_s1_l1 >= 0.2 or x_pyr_l1 >= 0.2
            ),
        }
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("swc", type=Path)
    ap.add_argument("--flag-model", type=Path, default=DEFAULT_FLAG_MODEL)
    ap.add_argument("--v12-dir", type=Path, default=DEFAULT_V12_DIR)
    ap.add_argument("--second-stage1-dir", type=Path, default=DEFAULT_SECOND_STAGE1_DIR)
    ap.add_argument("--baseline-cache-dir", type=Path, default=DEFAULT_BASELINE_CACHE)
    ap.add_argument("--baseline-seed", type=int, default=DEFAULT_BASELINE_SEED)
    ap.add_argument("--v12-ensemble-seeds", default=DEFAULT_V12_ENSEMBLE_SEEDS)
    ap.add_argument(
        "--claimed-cell-type",
        choices=["pyramidal", "interneuron"],
        default=None,
        help="Corpus/GT cell type label used for deployment guards when known.",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.swc.is_file():
        raise SystemExit(f"MISSING: {args.swc}")
    if not args.flag_model.is_file():
        raise SystemExit(f"MISSING: {args.flag_model}")

    bundle = joblib.load(args.flag_model)
    row = _build_feature_row(args.swc, args.v12_dir, args.second_stage1_dir)
    if _needs_xmodel_features(bundle):
        _add_xmodel_features(row, args.swc, args.baseline_cache_dir, args.baseline_seed)
    if _needs_prefix(bundle, "xmodel_v12ens_"):
        _add_v12ens_features(row, args.swc, _parse_seed_list(args.v12_ensemble_seeds))
    if _needs_prefix(bundle, "xmodel_sanity_"):
        _add_sanity_features(row, args.claimed_cell_type)
    feature_df, _numeric, _categorical = _engineer_features(pd.DataFrame([row]))

    rank_model = bundle["rank_model"]
    calibrated_model = bundle["calibrated_model"]
    rank_score = float(rank_model.predict_proba(feature_df)[:, 1][0])
    prob_bad = float(calibrated_model.predict_proba(feature_df)[:, 1][0])

    thresholds = bundle.get("validation_rank_thresholds", [])
    cell_type_filter = bundle.get("cell_type_filter", "all")
    filter_cell_type = args.claimed_cell_type or row["stage1_pred"]
    eligible_by_filter = (
        cell_type_filter == "all" or filter_cell_type == cell_type_filter
    )
    flags = [
        {
            "reject_rate": t["reject_rate"],
            "rank_score_threshold": t["rank_score_threshold"],
            "flag": bool(rank_score >= float(t["rank_score_threshold"])),
        }
        for t in thresholds
    ]
    guarded_flags = [
        {
            **flag,
            "flag": bool(flag["flag"] and row["stage1_pred"] == "pyramidal"),
        }
        for flag in flags
    ]
    filter_guarded_flags = [
        {
            **flag,
            "flag": bool(flag["flag"] and eligible_by_filter),
        }
        for flag in flags
    ]
    precision_flags = []
    for t in bundle.get("validation_precision_thresholds", []):
        score_threshold = float(t.get("score_threshold", float("inf")))
        finite_threshold = bool(np.isfinite(score_threshold))
        precision_flags.append(
            {
                "target_precision": t.get("target_precision"),
                "validation_precision": t.get("validation_precision"),
                "rank_score_threshold": score_threshold if finite_threshold else None,
                "flag": bool(finite_threshold and rank_score >= score_threshold),
            }
        )
    recall_flags = []
    for t in bundle.get("validation_recall_thresholds", []):
        score_threshold = float(t.get("score_threshold", float("inf")))
        finite_threshold = bool(np.isfinite(score_threshold))
        recall_flags.append(
            {
                "target_recall": t.get("target_recall"),
                "validation_recall": t.get("validation_recall"),
                "validation_precision": t.get("validation_precision"),
                "rank_score_threshold": score_threshold if finite_threshold else None,
                "flag": bool(
                    finite_threshold and rank_score >= score_threshold and eligible_by_filter
                ),
            }
        )

    result = {
        "file": args.swc.name,
        "target_threshold": bundle.get("target_threshold"),
        "selected_model": bundle.get("selected_model_name"),
        "selected_feature_set": bundle.get("selected_feature_set"),
        "cell_type_filter": cell_type_filter,
        "filter_cell_type_basis": "claimed_cell_type" if args.claimed_cell_type else "stage1_pred",
        "filter_cell_type": filter_cell_type,
        "eligible_by_cell_type_filter": eligible_by_filter,
        "rank_score": rank_score,
        "prob_bad": prob_bad,
        "threshold_flags": flags,
        "cell_type_filter_guarded_threshold_flags": filter_guarded_flags,
        "stage1_pyramidal_guarded_threshold_flags": guarded_flags,
        "precision_target_flags": precision_flags,
        "recall_target_flags": recall_flags,
        "stage1_pred": row["stage1_pred"],
        "stage1_conf": row["stage1_conf"],
        "pred_counts": {
            "axon": row["pred_axon"],
            "basal": row["pred_basal"],
            "apical": row["pred_apical"],
        },
        "branch3_enabled": bool((args.v12_dir / "gnn_branch3_rescue.pt").is_file()),
    }
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
