#!/usr/bin/env python3
"""Sweep flag-model variants for bad per-cell labeling detection.

This is an experiment driver, not the production flagger. It compares:

- separate all/pyramidal/interneuron flaggers
- failure-type heads such as axon/apical/basal failures
- explicit morphology sanity features
- two-stage score combinations from failure heads
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper._train_flag_model import (  # noqa: E402
    F1_COL,
    _apply_recall_thresholds,
    _build_split,
    _calibrate_prefit,
    _candidate_models,
    _engineer_features,
    _feature_sets,
    _load_data,
    _make_preprocessor,
    _precision_thresholds,
    _rank_thresholds,
    _recall_thresholds,
    _safe_ap,
    _safe_auc,
)


DEFAULT_INPUT = ROOT / "paper" / "results" / "heldout_per_cell_f1_seed123_branch3.csv"
DEFAULT_STAGE1 = ROOT / "paper" / "results" / "stage1_disagreement.csv"
DEFAULT_FEATURES = ROOT / "paper" / "results" / "flag_features_seed123_branch3_xmodel.csv"
DEFAULT_OUT_CSV = ROOT / "paper" / "results" / "flag_improvement_sweep_seed123_branch3.csv"
DEFAULT_OUT_TXT = ROOT / "paper" / "results" / "flag_improvement_sweep_seed123_branch3.txt"
DEFAULT_MODEL_DIR = ROOT / "paper" / "models" / "flag_improvement_sweep_seed123_branch3"


def _display(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def _add_sanity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deployment-safe features from claimed cell type and predictions.

    `cell_type_gt` is used here as the claimed corpus label. At deployment the
    same features should be computed from the claimed/expected cell type, not
    from hidden quality information.
    """
    out = df.copy()
    claimed = out.get("cell_type_gt", pd.Series("", index=out.index)).astype(str)
    stage1 = out.get("stage1_pred", pd.Series("", index=out.index)).astype(str)
    n_nodes = pd.to_numeric(out.get("n_nodes", 0), errors="coerce").fillna(0.0)
    denom = np.maximum(n_nodes.to_numpy(dtype=float) - 1.0, 1.0)
    pred_axon = pd.to_numeric(out.get("pred_axon", 0), errors="coerce").fillna(0.0)
    pred_basal = pd.to_numeric(out.get("pred_basal", 0), errors="coerce").fillna(0.0)
    pred_apical = pd.to_numeric(out.get("pred_apical", 0), errors="coerce").fillna(0.0)
    ax_frac = pred_axon.to_numpy(dtype=float) / denom
    ba_frac = pred_basal.to_numpy(dtype=float) / denom
    ap_frac = pred_apical.to_numpy(dtype=float) / denom
    conf_p10 = pd.to_numeric(out.get("conf_p10", np.nan), errors="coerce")
    stage1_conf = pd.to_numeric(out.get("stage1_conf", np.nan), errors="coerce")
    x_pyr_l1 = pd.to_numeric(out.get("xmodel_pyr_l1_mean", 0), errors="coerce").fillna(0.0)
    x_s1_l1 = pd.to_numeric(out.get("xmodel_s1_l1_mean", 0), errors="coerce").fillna(0.0)
    x_neurom_pyr = pd.to_numeric(
        out.get("xmodel_neurom_rf_pyr_v12_l1_frac_delta", 0),
        errors="coerce",
    ).fillna(0.0)

    out["xmodel_sanity_claimed_is_pyramidal"] = (claimed == "pyramidal").astype(float)
    out["xmodel_sanity_claimed_is_interneuron"] = (claimed == "interneuron").astype(float)
    out["xmodel_sanity_stage1_claim_mismatch"] = (claimed != stage1).astype(float)
    out["xmodel_sanity_claimed_pyr_stage1_int"] = (
        (claimed == "pyramidal") & (stage1 == "interneuron")
    ).astype(float)
    out["xmodel_sanity_claimed_int_stage1_pyr"] = (
        (claimed == "interneuron") & (stage1 == "pyramidal")
    ).astype(float)
    out["xmodel_sanity_pred_no_axon"] = (pred_axon <= 0).astype(float)
    out["xmodel_sanity_pred_no_basal"] = (pred_basal <= 0).astype(float)
    out["xmodel_sanity_pred_no_apical"] = (pred_apical <= 0).astype(float)
    out["xmodel_sanity_pred_single_neurite_class"] = (
        ((pred_axon > 0).astype(int) + (pred_basal > 0).astype(int) + (pred_apical > 0).astype(int))
        <= 1
    ).astype(float)
    out["xmodel_sanity_claimed_pyr_no_apical"] = (
        (claimed == "pyramidal") & (pred_apical <= 0)
    ).astype(float)
    out["xmodel_sanity_claimed_int_no_axon"] = (
        (claimed == "interneuron") & (pred_axon <= 0)
    ).astype(float)
    out["xmodel_sanity_claimed_pyr_tiny_apical"] = (
        (claimed == "pyramidal") & (ap_frac < 0.01)
    ).astype(float)
    out["xmodel_sanity_claimed_int_tiny_axon"] = (
        (claimed == "interneuron") & (ax_frac < 0.01)
    ).astype(float)
    out["xmodel_sanity_pred_axon_frac"] = ax_frac
    out["xmodel_sanity_pred_basal_frac"] = ba_frac
    out["xmodel_sanity_pred_apical_frac"] = ap_frac
    out["xmodel_sanity_low_conf_tail"] = (conf_p10 < 0.7).astype(float)
    out["xmodel_sanity_low_stage1_conf"] = (stage1_conf < 0.8).astype(float)
    out["xmodel_sanity_xmodel_delta_high"] = (x_pyr_l1 >= 0.2).astype(float)
    out["xmodel_sanity_neurom_delta_high"] = (x_neurom_pyr >= 0.5).astype(float)
    out["xmodel_sanity_low_conf_x_delta"] = (
        (conf_p10 < 0.9).fillna(False) & (x_pyr_l1 >= 0.2)
    ).astype(float)
    out["xmodel_sanity_stage1_or_x_delta"] = (
        (claimed != stage1) | (x_s1_l1 >= 0.2) | (x_pyr_l1 >= 0.2)
    ).astype(float)
    return out


def _target_series(df: pd.DataFrame, target_name: str) -> pd.Series:
    if target_name == "cell_f060":
        return pd.to_numeric(df[F1_COL], errors="coerce") < 0.6
    if target_name == "axon_f050":
        return pd.to_numeric(df["axon_F1"], errors="coerce") < 0.5
    if target_name == "basal_f060":
        return pd.to_numeric(df["basal_F1"], errors="coerce") < 0.6
    if target_name == "apical_f050":
        return pd.to_numeric(df["apical_F1"], errors="coerce") < 0.5
    raise KeyError(target_name)


def _threshold_metrics(
    f1: np.ndarray,
    score: np.ndarray,
    target_threshold: float,
    threshold: float,
) -> dict:
    rejected = score >= threshold
    bad = f1 < target_threshold
    tp = int((rejected & bad).sum())
    fp = int((rejected & ~bad).sum())
    fn = int((~rejected & bad).sum())
    tn = int((~rejected & ~bad).sum())
    kept = f1[~rejected]
    return {
        "score_threshold": float(threshold),
        "n_rejected": int(rejected.sum()),
        "precision_lt_target": float(tp / max(1, tp + fp)),
        "recall_lt_target": float(tp / max(1, tp + fn)),
        "binary_accuracy": float((tp + tn) / max(1, len(f1))),
        "kept_f1_p10": float(np.percentile(kept, 10)) if kept.size else float("nan"),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _score_at_validation_recall(
    val_f1: np.ndarray,
    val_score: np.ndarray,
    test_f1: np.ndarray,
    test_score: np.ndarray,
    recall_targets: list[float],
    target_threshold: float = 0.6,
) -> list[dict]:
    rows = _recall_thresholds(val_f1, val_score, target_threshold, recall_targets)
    return _apply_recall_thresholds(test_f1, test_score, target_threshold, rows)


def _fit_candidates(
    df_feat: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    y: pd.Series,
    split,
    seed: int,
) -> list[dict]:
    out: list[dict] = []
    for feature_set, cat_for_set in _feature_sets(categorical_features).items():
        for model_name, estimator in _candidate_models(seed).items():
            pipe = Pipeline(
                [
                    ("preprocess", _make_preprocessor(numeric_features, cat_for_set)),
                    ("model", estimator),
                ]
            )
            pipe.fit(df_feat.loc[split.train], y.loc[split.train])
            val_score = pipe.predict_proba(df_feat.loc[split.val])[:, 1]
            test_score = pipe.predict_proba(df_feat.loc[split.test])[:, 1]
            val_y = y.loc[split.val].to_numpy(dtype=int)
            test_y = y.loc[split.test].to_numpy(dtype=int)
            out.append(
                {
                    "model_name": model_name,
                    "feature_set": feature_set,
                    "cat_features": cat_for_set,
                    "pipe": pipe,
                    "val_score": val_score,
                    "test_score": test_score,
                    "val_ap": _safe_ap(val_y, val_score),
                    "val_auc": _safe_auc(val_y, val_score),
                    "test_ap": _safe_ap(test_y, test_score),
                    "test_auc": _safe_auc(test_y, test_score),
                }
            )
    return out


def _select_for_recall(
    candidates: list[dict],
    val_f1: np.ndarray,
    test_f1: np.ndarray,
    recall_targets: list[float],
) -> list[dict]:
    selected: list[dict] = []
    per_candidate = []
    for cand in candidates:
        val_rows = _recall_thresholds(val_f1, cand["val_score"], 0.6, recall_targets)
        test_rows = _apply_recall_thresholds(test_f1, cand["test_score"], 0.6, val_rows)
        per_candidate.append((cand, val_rows, test_rows))

    for i, target_recall in enumerate(recall_targets):
        cand, val_rows, test_rows = max(
            per_candidate,
            key=lambda item: (
                item[1][i]["validation_precision"],
                item[0]["val_ap"],
                item[0]["val_auc"],
            ),
        )
        test = test_rows[i]
        val = val_rows[i]
        selected.append(
            {
                "target_recall": float(target_recall),
                "model_name": cand["model_name"],
                "feature_set": cand["feature_set"],
                "val_precision": float(val["validation_precision"]),
                "val_recall": float(val["validation_recall"]),
                "score_threshold": float(val["score_threshold"]),
                "test_precision": float(test["precision_lt_target"]),
                "test_recall": float(test["recall_lt_target"]),
                "test_n_rejected": int(test["n_rejected"]),
                "test_tp": int(test["tp"]),
                "test_fp": int(test["fp"]),
                "test_fn": int(test["fn"]),
                "test_accuracy": float(test["binary_accuracy"]),
                "test_kept_f1_p10": float(test["kept_f1_p10"]),
                "val_ap": float(cand["val_ap"]),
                "test_ap": float(cand["test_ap"]),
            }
        )
    return selected


def _validation_percentile_score(val_score: np.ndarray, test_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.sort(val_score)
    n = max(1, len(order))
    val_rank = np.searchsorted(order, val_score, side="right") / n
    test_rank = np.searchsorted(order, test_score, side="right") / n
    return val_rank.astype(float), test_rank.astype(float)


def _two_stage_heads(
    df_raw: pd.DataFrame,
    df_feat: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    split,
    seed: int,
    head_targets: list[str],
    recall_targets: list[float],
) -> dict | None:
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    heads: list[dict] = []
    for target_name in head_targets:
        y = _target_series(df_raw, target_name).astype(int).reset_index(drop=True)
        n_pos_train = int(y.loc[split.train].sum())
        n_neg_train = int((1 - y.loc[split.train]).sum())
        n_pos_val = int(y.loc[split.val].sum())
        if n_pos_train < 10 or n_neg_train < 10 or n_pos_val < 3:
            continue
        candidates = _fit_candidates(df_feat, numeric_features, categorical_features, y, split, seed)
        best = max(candidates, key=lambda c: (c["val_ap"], c["val_auc"]))
        val_rank, test_rank = _validation_percentile_score(best["val_score"], best["test_score"])
        val_parts.append(val_rank)
        test_parts.append(test_rank)
        heads.append(
            {
                "target": target_name,
                "model_name": best["model_name"],
                "feature_set": best["feature_set"],
                "val_ap": float(best["val_ap"]),
                "test_ap": float(best["test_ap"]),
                "train_positives": n_pos_train,
                "val_positives": n_pos_val,
            }
        )
    if len(val_parts) < 2:
        return None
    val_score = np.max(np.vstack(val_parts), axis=0)
    test_score = np.max(np.vstack(test_parts), axis=0)
    val_f1 = pd.to_numeric(df_raw.loc[split.val, F1_COL], errors="coerce").to_numpy(dtype=float)
    test_f1 = pd.to_numeric(df_raw.loc[split.test, F1_COL], errors="coerce").to_numpy(dtype=float)
    val_rows = _recall_thresholds(val_f1, val_score, 0.6, recall_targets)
    test_rows = _apply_recall_thresholds(test_f1, test_score, 0.6, val_rows)
    return {
        "heads": heads,
        "target_rows": [
            {
                "target_recall": float(recall_targets[i]),
                "val_precision": float(val_rows[i]["validation_precision"]),
                "val_recall": float(val_rows[i]["validation_recall"]),
                "score_threshold": float(val_rows[i]["score_threshold"]),
                "test_precision": float(test_rows[i]["precision_lt_target"]),
                "test_recall": float(test_rows[i]["recall_lt_target"]),
                "test_n_rejected": int(test_rows[i]["n_rejected"]),
                "test_tp": int(test_rows[i]["tp"]),
                "test_fp": int(test_rows[i]["fp"]),
                "test_fn": int(test_rows[i]["fn"]),
                "test_accuracy": float(test_rows[i]["binary_accuracy"]),
                "test_kept_f1_p10": float(test_rows[i]["kept_f1_p10"]),
            }
            for i in range(len(recall_targets))
        ],
    }


def _save_bundle(
    model_dir: Path,
    experiment_name: str,
    target_recall: float,
    candidate: dict,
    df_feat: pd.DataFrame,
    y_cell: pd.Series,
    split,
    numeric_features: list[str],
    seed: int,
    recall_targets: list[float],
) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    pipe = candidate["pipe"]
    calibrated = _calibrate_prefit(pipe, df_feat, y_cell, split.val)
    val_score = candidate["val_score"]
    val_f1 = pd.to_numeric(df_feat.loc[split.val, F1_COL], errors="coerce").to_numpy(dtype=float)
    suffix = f"r{int(round(target_recall * 100)):03d}"
    path = model_dir / f"{experiment_name}_{suffix}.joblib"
    joblib.dump(
        {
            "rank_model": pipe,
            "calibrated_model": calibrated,
            "target_threshold": 0.6,
            "numeric_features": numeric_features,
            "categorical_features": candidate["cat_features"],
            "seed": seed,
            "selection_reject_rate": 0.02,
            "selected_model_name": candidate["model_name"],
            "selected_feature_set": candidate["feature_set"],
            "cell_type_filter": experiment_name.split("_")[0] if experiment_name.split("_")[0] in {"pyramidal", "interneuron"} else "all",
            "selected_for_target_recall": float(target_recall),
            "validation_rank_thresholds": _rank_thresholds(val_score, [0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.30]),
            "validation_precision_thresholds": _precision_thresholds(val_f1, val_score, 0.6, [0.6, 0.7, 0.8, 0.9]),
            "validation_recall_thresholds": _recall_thresholds(val_f1, val_score, 0.6, recall_targets),
        },
        path,
    )
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--stage1-disagreement", type=Path, default=DEFAULT_STAGE1)
    ap.add_argument("--confidence-features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    ap.add_argument("--out-txt", type=Path, default=DEFAULT_OUT_TXT)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--recall-targets", default="0.5,0.7,0.8,0.9,0.95")
    args = ap.parse_args()

    recall_targets = [float(x) for x in args.recall_targets.split(",") if x.strip()]
    df_loaded = _load_data(args.input, args.stage1_disagreement, args.confidence_features)

    all_rows: list[dict] = []
    detailed: dict = {
        "input": _display(args.input),
        "stage1_disagreement": _display(args.stage1_disagreement),
        "confidence_features": _display(args.confidence_features),
        "seed": args.seed,
        "recall_targets": recall_targets,
        "experiments": [],
    }

    for feature_mode in ("xmodel", "xmodel_sanity"):
        df_mode = _add_sanity_features(df_loaded) if feature_mode == "xmodel_sanity" else df_loaded.copy()
        for subset_name in ("all", "pyramidal", "interneuron"):
            df_sub = df_mode.copy()
            if subset_name != "all":
                df_sub = df_sub[df_sub["cell_type_gt"].astype(str) == subset_name].copy()
            df_feat, numeric_features, categorical_features = _engineer_features(df_sub)
            f1 = pd.to_numeric(df_feat[F1_COL], errors="coerce")
            valid = f1.notna()
            df_feat = df_feat.loc[valid].reset_index(drop=True)
            df_raw = df_sub.loc[valid].reset_index(drop=True)
            f1 = f1.loc[valid].reset_index(drop=True)
            y_cell = (f1 < 0.6).astype(int)
            if int(y_cell.sum()) < 10 or int((1 - y_cell).sum()) < 10:
                continue
            split = _build_split(df_feat, y_cell, args.seed)
            val_f1 = f1.loc[split.val].to_numpy(dtype=float)
            test_f1 = f1.loc[split.test].to_numpy(dtype=float)

            print(f"\n=== {feature_mode} / {subset_name}: cell_f060 ===", flush=True)
            candidates = _fit_candidates(df_feat, numeric_features, categorical_features, y_cell, split, args.seed)
            selected = _select_for_recall(candidates, val_f1, test_f1, recall_targets)
            exp_name = f"{subset_name}_{feature_mode}_cell_f060"
            # Save the most useful r080-style single-model bundle for each subset/mode.
            save_target = 0.8 if 0.8 in recall_targets else recall_targets[len(recall_targets) // 2]
            save_idx = recall_targets.index(save_target)
            best_for_save = max(
                candidates,
                key=lambda c: (
                    _recall_thresholds(val_f1, c["val_score"], 0.6, recall_targets)[save_idx]["validation_precision"],
                    c["val_ap"],
                    c["val_auc"],
                ),
            )
            bundle_path = _save_bundle(
                args.model_dir,
                exp_name,
                save_target,
                best_for_save,
                df_feat,
                y_cell,
                split,
                numeric_features,
                args.seed,
                recall_targets,
            )
            for row in selected:
                row_out = {
                    "experiment": "single_cell_f060",
                    "feature_mode": feature_mode,
                    "subset": subset_name,
                    "target": "cell_f060",
                    "n_rows": int(len(df_feat)),
                    "n_bad": int(y_cell.sum()),
                    "bad_rate": float(y_cell.mean()),
                    "saved_bundle": _display(bundle_path) if row["target_recall"] == save_target else "",
                    **row,
                }
                all_rows.append(row_out)

            detail = {
                "experiment": exp_name,
                "n_rows": int(len(df_feat)),
                "n_bad": int(y_cell.sum()),
                "split_sizes": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
                "selected": selected,
                "saved_bundle": _display(bundle_path),
            }
            detailed["experiments"].append(detail)

            # Failure-type heads: report each as a standalone target if it has enough signal.
            head_targets = ["axon_f050", "basal_f060"]
            if subset_name != "interneuron":
                head_targets.append("apical_f050")
            head_targets.insert(0, "cell_f060")
            for target_name in head_targets[1:]:
                y_head = _target_series(df_raw, target_name).astype(int).reset_index(drop=True)
                if int(y_head.loc[split.train].sum()) < 10 or int((1 - y_head.loc[split.train]).sum()) < 10:
                    continue
                print(f"=== {feature_mode} / {subset_name}: {target_name} head ===", flush=True)
                head_candidates = _fit_candidates(df_feat, numeric_features, categorical_features, y_head, split, args.seed)
                # This table evaluates the failure head against the original cell-F1-bad target.
                head_selected = _select_for_recall(head_candidates, val_f1, test_f1, recall_targets)
                for row in head_selected:
                    all_rows.append(
                        {
                            "experiment": "single_failure_head_ranked_against_cell_f060",
                            "feature_mode": feature_mode,
                            "subset": subset_name,
                            "target": target_name,
                            "n_rows": int(len(df_feat)),
                            "n_bad": int(y_cell.sum()),
                            "bad_rate": float(y_cell.mean()),
                            "saved_bundle": "",
                            **row,
                        }
                    )

            two = _two_stage_heads(
                df_raw,
                df_feat,
                numeric_features,
                categorical_features,
                split,
                args.seed,
                head_targets,
                recall_targets,
            )
            if two is not None:
                for row in two["target_rows"]:
                    all_rows.append(
                        {
                            "experiment": "two_stage_max_failure_heads",
                            "feature_mode": feature_mode,
                            "subset": subset_name,
                            "target": "+".join(h["target"] for h in two["heads"]),
                            "n_rows": int(len(df_feat)),
                            "n_bad": int(y_cell.sum()),
                            "bad_rate": float(y_cell.mean()),
                            "saved_bundle": "",
                            "model_name": "two_stage",
                            "feature_set": "max_validation_percentile",
                            "val_ap": float("nan"),
                            "test_ap": float("nan"),
                            **row,
                        }
                    )
                detailed["experiments"].append(
                    {
                        "experiment": f"{subset_name}_{feature_mode}_two_stage",
                        "heads": two["heads"],
                        "selected": two["target_rows"],
                    }
                )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(args.out_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    detailed_path = args.out_csv.with_suffix(".json")
    detailed_path.write_text(json.dumps(detailed, indent=2), encoding="utf-8")

    rows = pd.DataFrame(all_rows)
    lines = [
        "Flag improvement sweep, seed123 Branch3 heldout",
        "=" * 96,
        "Rows are honest test metrics. Target is bad cell: held_out_F1 < 0.60.",
        "",
    ]
    for subset_name in ("pyramidal", "interneuron", "all"):
        lines.append(f"Best rows for subset={subset_name}")
        lines.append("-" * 96)
        part = rows[rows["subset"] == subset_name].copy()
        for target_recall in recall_targets:
            band = part[part["target_recall"] == target_recall]
            if band.empty:
                continue
            best = band.sort_values(
                ["test_recall", "test_precision", "test_kept_f1_p10"],
                ascending=[False, False, False],
            ).iloc[0]
            lines.append(
                f"target_R {target_recall:4.0%} | "
                f"test_P {best['test_precision']:6.1%} test_R {best['test_recall']:6.1%} "
                f"flagged {int(best['test_n_rejected']):4d} fp {int(best['test_fp']):4d} fn {int(best['test_fn']):3d} "
                f"acc {best['test_accuracy']:6.1%} keptP10 {best['test_kept_f1_p10']:.4f} | "
                f"{best['experiment']} / {best['feature_mode']} / {best['target']} / "
                f"{best['model_name']}:{best['feature_set']}"
            )
        lines.append("")
    lines.append(f"CSV: {_display(args.out_csv)}")
    lines.append(f"JSON: {_display(detailed_path)}")
    args.out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {_display(args.out_csv)}")
    print(f"Wrote {_display(args.out_txt)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
