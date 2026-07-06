#!/usr/bin/env python3
"""Train a per-cell learned flag/rejection model.

This is intentionally a meta-model over the full v12 pipeline outputs, not a
replacement for the labeler. It learns to predict whether a cell's held-out
per-cell neurite F1 falls below an operational threshold, then reports what
happens if the highest-risk cells are rejected.

Inputs are deployment-safe columns from ``heldout_per_cell_f1.csv``:
predicted class counts, Stage 1 prediction/confidence, source/lab tokens, and
optional cross-seed Stage 1 predictions. GT-only columns such as per-class F1,
accuracy, GT fractions, and GT apical geometry are used only for labels/reporting.

Usage:
    python -m paper._train_flag_model

Outputs:
    paper/models/flag_model/flag_model_f050.joblib
    paper/models/flag_model/flag_model_f060.joblib
    paper/models/flag_model/metadata.json
    paper/results/flag_model_eval.json
    paper/results/flag_model_eval.txt
    paper/results/flag_model_test_predictions.csv

The saved prediction CSV contains both:
    - rank_score_fXXX: raw model score used for top-K rejection
    - prob_bad_fXXX: isotonic-calibrated display probability
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = ROOT / "paper" / "results" / "heldout_per_cell_f1.csv"
DEFAULT_STAGE1_DISAGREE = ROOT / "paper" / "results" / "stage1_disagreement.csv"
DEFAULT_CONFIDENCE_FEATURES = ROOT / "paper" / "results" / "flag_confidence_features.csv"
DEFAULT_MODEL_DIR = ROOT / "paper" / "models" / "flag_model"
DEFAULT_OUT_STEM = ROOT / "paper" / "results" / "flag_model_eval"

F1_COL = "held_out_F1"

# Columns that may exist in the source CSV but must never enter X.
GT_OR_LABEL_COLUMNS = {
    "cell_type_gt",
    "held_out_F1",
    "acc",
    "axon_F1",
    "basal_F1",
    "apical_F1",
    "stage1_correct",
    "axon_frac",
    "apical_frac",
    "basal_frac",
    "apical_mean_z",
    "apical_above_soma",
    "apical_z_extent",
    "elapsed_s",
}

CONFIDENCE_NUMERIC_FEATURES = [
    "confidence_features_missing",
    "n_nodes_conf",
    "stage1_conf_conf_run",
    "pred_axon_conf_run",
    "pred_basal_conf_run",
    "pred_apical_conf_run",
    "node_label_disagree_frac",
    "node_conf_abs_delta_mean",
    "conf_mean",
    "conf_std",
    "conf_min",
    "conf_p05",
    "conf_p10",
    "conf_p25",
    "conf_median",
    "conf_frac_lt_05",
    "conf_frac_lt_06",
    "conf_frac_lt_07",
    "conf_frac_lt_08",
    "conf_frac_lt_09",
    "axon_conf_mean",
    "axon_conf_p10",
    "axon_conf_frac_lt_07",
    "basal_conf_mean",
    "basal_conf_p10",
    "basal_conf_frac_lt_07",
    "apical_conf_mean",
    "apical_conf_p10",
    "apical_conf_frac_lt_07",
    "n_pred_branches",
    "branch_conf_mean",
    "branch_conf_min",
    "branch_conf_p10",
    "branch_frac_conf_lt_07",
    "branch_frac_conf_lt_08",
    "branch_n_nodes_p90",
    "pred_soma_z",
    "pred_axon_z_extent",
    "pred_axon_z_mean_rel_soma",
    "pred_axon_z_std",
    "pred_axon_x_extent",
    "pred_axon_y_extent",
    "pred_basal_z_extent",
    "pred_basal_z_mean_rel_soma",
    "pred_basal_z_std",
    "pred_basal_x_extent",
    "pred_basal_y_extent",
    "pred_apical_z_extent",
    "pred_apical_z_mean_rel_soma",
    "pred_apical_z_std",
    "pred_apical_x_extent",
    "pred_apical_y_extent",
    "branch3_changed_frac",
    "branch3_label_disagree_frac",
    "branch3_conf_abs_delta_mean",
    "branch3_axon_delta_frac",
    "branch3_basal_delta_frac",
    "branch3_apical_delta_frac",
    "branch3_to_apical_frac",
    "branch3_from_apical_frac",
]
CONFIDENCE_CATEGORICAL_FEATURES = ["stage1_pred_conf_run"]
EXTRA_NUMERIC_PREFIXES = ("xmodel_", "baseline_")


def _display_path(path: Path | str | None) -> str:
    if path is None:
        return ""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


@dataclass
class SplitInfo:
    train: list[int]
    val: list[int]
    test: list[int]


@dataclass
class CandidateResult:
    name: str
    feature_set: str
    threshold: float
    val_average_precision: float
    val_roc_auc: float
    val_precision_at_selection_rate: float
    val_kept_f1_p10_at_selection_rate: float
    test_average_precision: float
    test_roc_auc: float
    selected: bool = False


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return out


def _lab_tokens(filename: str) -> tuple[str, str]:
    rest = filename.split("__", 1)[1] if "__" in filename else filename
    stem = Path(rest).stem
    toks = [t.lower() for t in re.split(r"[-_\s]+", stem) if t]
    if not toks:
        return "", ""
    lab_prefix = toks[0]
    study_prefix = "_".join(toks[:2]) if len(toks) >= 2 else toks[0]
    return lab_prefix, study_prefix


def _load_data(
    input_csv: Path,
    stage1_disagreement_csv: Path | None,
    confidence_features_csv: Path | None = None,
) -> pd.DataFrame:
    if not input_csv.is_file():
        raise SystemExit(f"MISSING: {input_csv}")
    df = pd.read_csv(input_csv)
    if F1_COL not in df.columns:
        raise SystemExit(f"{input_csv} is missing required column {F1_COL!r}")

    if stage1_disagreement_csv is not None and stage1_disagreement_csv.is_file():
        st = pd.read_csv(stage1_disagreement_csv)
        keep = ["file", "seed42_pred", "seed42_conf", "seed789_pred", "seed789_conf"]
        missing = [c for c in keep if c not in st.columns]
        if missing:
            raise SystemExit(
                f"{stage1_disagreement_csv} missing expected columns: {missing}"
            )
        df = df.merge(st[keep], on="file", how="left")
    else:
        df["seed42_pred"] = ""
        df["seed42_conf"] = np.nan
        df["seed789_pred"] = ""
        df["seed789_conf"] = np.nan

    df["confidence_features_missing"] = 1
    if confidence_features_csv is not None and confidence_features_csv.is_file():
        cf = pd.read_csv(confidence_features_csv)
        if "file" not in cf.columns:
            raise SystemExit(f"{confidence_features_csv} is missing required column 'file'")
        keep = [
            c
            for c in ["file", *CONFIDENCE_NUMERIC_FEATURES, *CONFIDENCE_CATEGORICAL_FEATURES]
            if c in cf.columns
        ]
        keep.extend(
            c
            for c in cf.columns
            if c not in keep and c.startswith(EXTRA_NUMERIC_PREFIXES)
        )
        df = df.merge(cf[keep], on="file", how="left", indicator="_confidence_merge")
        df["confidence_features_missing"] = (
            df["_confidence_merge"].astype(str) == "left_only"
        ).astype(int)
        df = df.drop(columns=["_confidence_merge"])

    return df


def _engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = df.copy()

    numeric_base = [
        "n_nodes",
        "stage1_conf",
        "n_components",
        "pred_axon",
        "pred_basal",
        "pred_apical",
        "seed42_conf",
        "seed789_conf",
    ]
    for col in numeric_base:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "source" not in df.columns:
        df["source"] = df["file"].astype(str).str.split("__", n=1).str[0]
    if "stage1_pred" not in df.columns:
        df["stage1_pred"] = ""

    lab_parts = df["file"].astype(str).map(_lab_tokens)
    df["lab_prefix"] = [x[0] for x in lab_parts]
    df["study_prefix"] = [x[1] for x in lab_parts]

    non_soma = np.maximum(df["n_nodes"].fillna(0).to_numpy(dtype=float) - 1.0, 1.0)
    for cls in ("axon", "basal", "apical"):
        count_col = f"pred_{cls}"
        frac_col = f"pred_{cls}_frac"
        log_col = f"log_pred_{cls}"
        df[frac_col] = df[count_col].fillna(0).to_numpy(dtype=float) / non_soma
        df[log_col] = np.log1p(df[count_col].fillna(0).to_numpy(dtype=float))

    frac_cols = ["pred_axon_frac", "pred_basal_frac", "pred_apical_frac"]
    fracs = df[frac_cols].fillna(0.0).clip(lower=0.0)
    df["pred_entropy"] = (-(fracs * np.log(fracs.replace(0.0, np.nan)))).sum(axis=1).fillna(0.0)
    df["pred_class_count"] = (
        df[["pred_axon", "pred_basal", "pred_apical"]].fillna(0.0) > 0.0
    ).sum(axis=1)
    df["max_pred_frac"] = fracs.max(axis=1)
    df["pred_apical_zero"] = (df["pred_apical"].fillna(0.0) == 0.0).astype(int)
    df["pred_axon_zero"] = (df["pred_axon"].fillna(0.0) == 0.0).astype(int)
    df["single_class_pred"] = (df["pred_class_count"] == 1).astype(int)
    df["log_n_nodes"] = np.log1p(df["n_nodes"].fillna(0.0))
    df["stage1_uncertainty"] = 1.0 - df["stage1_conf"]
    df["stage1_is_pyramidal"] = (df["stage1_pred"] == "pyramidal").astype(int)
    df["pyr_no_apical"] = (
        (df["stage1_pred"] == "pyramidal") & (df["pred_apical"].fillna(0.0) == 0.0)
    ).astype(int)
    df["int_has_apical"] = (
        (df["stage1_pred"] == "interneuron") & (df["pred_apical"].fillna(0.0) > 0.0)
    ).astype(int)
    df["seed_pred_disagree"] = (df["seed42_pred"] != df["seed789_pred"]).astype(int)
    df["seed_conf_min"] = df[["seed42_conf", "seed789_conf"]].min(axis=1)
    df["seed_conf_max"] = df[["seed42_conf", "seed789_conf"]].max(axis=1)
    df["seed_conf_delta"] = (df["seed42_conf"] - df["seed789_conf"]).abs()

    numeric_features = [
        "log_n_nodes",
        "n_nodes",
        "stage1_conf",
        "stage1_uncertainty",
        "n_components",
        "pred_axon",
        "pred_basal",
        "pred_apical",
        "log_pred_axon",
        "log_pred_basal",
        "log_pred_apical",
        "pred_axon_frac",
        "pred_basal_frac",
        "pred_apical_frac",
        "pred_entropy",
        "pred_class_count",
        "max_pred_frac",
        "pred_apical_zero",
        "pred_axon_zero",
        "single_class_pred",
        "stage1_is_pyramidal",
        "pyr_no_apical",
        "int_has_apical",
        "seed42_conf",
        "seed789_conf",
        "seed_pred_disagree",
        "seed_conf_min",
        "seed_conf_max",
        "seed_conf_delta",
    ]
    categorical_features = [
        "source",
        "stage1_pred",
        "lab_prefix",
        "study_prefix",
        "seed42_pred",
        "seed789_pred",
    ]

    for col in CONFIDENCE_NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            numeric_features.append(col)
    for col in CONFIDENCE_CATEGORICAL_FEATURES:
        if col in df.columns:
            categorical_features.append(col)
    for col in sorted(c for c in df.columns if c.startswith(EXTRA_NUMERIC_PREFIXES)):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        numeric_features.append(col)

    numeric_features = [
        c for c in numeric_features if c in df.columns and not df[c].isna().all()
    ]

    forbidden = GT_OR_LABEL_COLUMNS & (set(numeric_features) | set(categorical_features))
    if forbidden:
        raise AssertionError(f"GT/label leakage in feature list: {sorted(forbidden)}")
    return df, numeric_features, categorical_features


def _make_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                    sparse_output=False,
                ),
                categorical_features,
            ),
        ],
        verbose_feature_names_out=False,
    )


def _candidate_models(seed: int) -> dict[str, object]:
    return {
        "logreg": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=0.5,
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.03,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            class_weight="balanced",
            random_state=seed,
        ),
        "rf": RandomForestClassifier(
            n_estimators=700,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=700,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "gb": GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=3,
            random_state=seed,
        ),
    }


def _feature_sets(categorical_features: list[str]) -> dict[str, list[str]]:
    """Feature-set variants.

    Lab/study IDs can be useful for known source quirks, but they also risk
    sweeping up good cells from a hard lab. Try both context-rich and
    context-light variants and select by validation precision.
    """
    no_lab = [c for c in categorical_features if c not in {"lab_prefix", "study_prefix"}]
    return {
        "full": categorical_features,
        "no_lab": no_lab,
        "numeric_only": [],
    }


def _build_split(df: pd.DataFrame, y_split: pd.Series, seed: int) -> SplitInfo:
    train_val_idx, test_idx = train_test_split(
        df.index,
        test_size=0.20,
        random_state=seed,
        stratify=y_split,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        random_state=seed,
        stratify=y_split.loc[train_val_idx],
    )
    return SplitInfo(
        train=list(map(int, train_idx)),
        val=list(map(int, val_idx)),
        test=list(map(int, test_idx)),
    )


def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def _safe_ap(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, score))


def _rejection_metrics(
    f1: np.ndarray,
    score: np.ndarray,
    reject_rates: Iterable[float],
    target_threshold: float,
) -> list[dict]:
    out: list[dict] = []
    n = len(f1)
    order = np.argsort(-score)
    n_bad_target = int((f1 < target_threshold).sum())
    n_bad_05 = int((f1 < 0.5).sum())
    n_bad_06 = int((f1 < 0.6).sum())

    for rate in reject_rates:
        k = max(1, int(round(rate * n)))
        rejected_idx = order[:k]
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[rejected_idx] = False
        rejected_f1 = f1[rejected_idx]
        kept_f1 = f1[keep_mask]
        tp = int((rejected_f1 < target_threshold).sum())
        fp = int(k - tp)
        fn = int(n_bad_target - tp)
        tn = int(n - k - fn)
        out.append(
            {
                "reject_rate": float(rate),
                "n_rejected": int(k),
                "precision_lt_target": float((rejected_f1 < target_threshold).mean()),
                "precision_lt_0_5": float((rejected_f1 < 0.5).mean()),
                "precision_lt_0_6": float((rejected_f1 < 0.6).mean()),
                "recall_lt_target": float((rejected_f1 < target_threshold).sum() / max(1, n_bad_target)),
                "recall_lt_0_5": float((rejected_f1 < 0.5).sum() / max(1, n_bad_05)),
                "recall_lt_0_6": float((rejected_f1 < 0.6).sum() / max(1, n_bad_06)),
                "kept_f1_mean": float(kept_f1.mean()) if kept_f1.size else float("nan"),
                "kept_f1_p10": float(np.percentile(kept_f1, 10)) if kept_f1.size else float("nan"),
                "kept_f1_p25": float(np.percentile(kept_f1, 25)) if kept_f1.size else float("nan"),
                "rejected_f1_mean": float(rejected_f1.mean()),
                "rejected_f1_median": float(np.median(rejected_f1)),
                "binary_accuracy": float((tp + tn) / max(1, n)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return out


def _threshold_rejection_metrics(
    f1: np.ndarray,
    score: np.ndarray,
    score_threshold: float,
    target_threshold: float,
) -> dict:
    rejected_mask = score >= score_threshold
    rejected_f1 = f1[rejected_mask]
    kept_f1 = f1[~rejected_mask]
    n = len(f1)
    n_bad = int((f1 < target_threshold).sum())
    tp = int((rejected_f1 < target_threshold).sum()) if rejected_f1.size else 0
    fp = int(rejected_f1.size - tp)
    fn = int(n_bad - tp)
    tn = int(n - rejected_f1.size - fn)
    return {
        "score_threshold": float(score_threshold),
        "reject_rate": float(rejected_f1.size / max(1, n)),
        "n_rejected": int(rejected_f1.size),
        "precision_lt_target": float(tp / max(1, rejected_f1.size)),
        "recall_lt_target": float(tp / max(1, n_bad)),
        "kept_f1_p10": float(np.percentile(kept_f1, 10)) if kept_f1.size else float("nan"),
        "rejected_f1_mean": float(rejected_f1.mean()) if rejected_f1.size else float("nan"),
        "binary_accuracy": float((tp + tn) / max(1, n)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _precision_thresholds(
    f1: np.ndarray,
    score: np.ndarray,
    target_threshold: float,
    precision_targets: Iterable[float],
) -> list[dict]:
    order = np.argsort(-score)
    sorted_score = score[order]
    sorted_bad = (f1[order] < target_threshold).astype(int)
    cum_bad = np.cumsum(sorted_bad)
    out: list[dict] = []
    n = len(f1)
    n_bad = int((f1 < target_threshold).sum())
    for target_precision in precision_targets:
        best: dict | None = None
        for i in range(n):
            k = i + 1
            if i + 1 < n and sorted_score[i + 1] == sorted_score[i]:
                continue
            precision = float(cum_bad[i] / k)
            if precision + 1e-12 < target_precision:
                continue
            threshold = float(sorted_score[i])
            metrics = _threshold_rejection_metrics(
                f1,
                score,
                threshold,
                target_threshold,
            )
            metrics.update(
                {
                    "target_precision": float(target_precision),
                    "validation_precision": precision,
                    "validation_recall": float(cum_bad[i] / max(1, n_bad)),
                }
            )
            best = metrics
        if best is not None:
            out.append(best)
        else:
            out.append(
                {
                    "target_precision": float(target_precision),
                    "score_threshold": float("inf"),
                    "reject_rate": 0.0,
                    "n_rejected": 0,
                    "precision_lt_target": 0.0,
                    "recall_lt_target": 0.0,
                    "kept_f1_p10": float(np.percentile(f1, 10)) if n else float("nan"),
                    "rejected_f1_mean": float("nan"),
                    "binary_accuracy": float((f1 >= target_threshold).mean()) if n else float("nan"),
                    "tp": 0,
                    "fp": 0,
                    "fn": n_bad,
                    "tn": n - n_bad,
                    "validation_precision": 0.0,
                    "validation_recall": 0.0,
                }
            )
    return out


def _apply_precision_thresholds(
    f1: np.ndarray,
    score: np.ndarray,
    target_threshold: float,
    threshold_rows: list[dict],
) -> list[dict]:
    out: list[dict] = []
    for row in threshold_rows:
        score_threshold = float(row["score_threshold"])
        if np.isinf(score_threshold):
            metrics = _threshold_rejection_metrics(f1, score, float("inf"), target_threshold)
        else:
            metrics = _threshold_rejection_metrics(f1, score, score_threshold, target_threshold)
        metrics["target_precision"] = float(row["target_precision"])
        metrics["validation_precision"] = float(row["validation_precision"])
        metrics["validation_recall"] = float(row["validation_recall"])
        out.append(metrics)
    return out


def _recall_thresholds(
    f1: np.ndarray,
    score: np.ndarray,
    target_threshold: float,
    recall_targets: Iterable[float],
) -> list[dict]:
    bad_scores = np.sort(score[f1 < target_threshold])[::-1]
    n_bad = int(bad_scores.size)
    out: list[dict] = []
    for target_recall in recall_targets:
        if n_bad == 0:
            metrics = _threshold_rejection_metrics(
                f1, score, float("inf"), target_threshold
            )
            metrics.update(
                {
                    "target_recall": float(target_recall),
                    "validation_precision": 0.0,
                    "validation_recall": 0.0,
                }
            )
            out.append(metrics)
            continue
        k = max(1, int(np.ceil(float(target_recall) * n_bad)))
        k = min(k, n_bad)
        threshold = float(bad_scores[k - 1])
        metrics = _threshold_rejection_metrics(
            f1, score, threshold, target_threshold
        )
        metrics.update(
            {
                "target_recall": float(target_recall),
                "validation_precision": metrics["precision_lt_target"],
                "validation_recall": metrics["recall_lt_target"],
            }
        )
        out.append(metrics)
    return out


def _apply_recall_thresholds(
    f1: np.ndarray,
    score: np.ndarray,
    target_threshold: float,
    threshold_rows: list[dict],
) -> list[dict]:
    out: list[dict] = []
    for row in threshold_rows:
        metrics = _threshold_rejection_metrics(
            f1,
            score,
            float(row["score_threshold"]),
            target_threshold,
        )
        metrics["target_recall"] = float(row["target_recall"])
        metrics["validation_precision"] = float(row["validation_precision"])
        metrics["validation_recall"] = float(row["validation_recall"])
        out.append(metrics)
    return out


def _stage1_pyramidal_guarded_rejection_metrics(
    df_eval: pd.DataFrame,
    f1: np.ndarray,
    score: np.ndarray,
    reject_rates: Iterable[float],
    target_threshold: float,
) -> list[dict]:
    if "stage1_pred" in df_eval.columns:
        stage1_pred = df_eval["stage1_pred"].astype(str).to_numpy()
    elif "stage1_pred_conf_run" in df_eval.columns:
        stage1_pred = df_eval["stage1_pred_conf_run"].astype(str).to_numpy()
    else:
        stage1_pred = np.array([""] * len(df_eval), dtype=object)
    eligible = stage1_pred == "pyramidal"
    out: list[dict] = []
    n = len(f1)
    eligible_idx = np.flatnonzero(eligible)
    eligible_order = eligible_idx[np.argsort(-score[eligible_idx])] if eligible_idx.size else np.array([], dtype=int)
    n_bad = int((f1 < target_threshold).sum())
    for rate in reject_rates:
        k_target = max(1, int(round(rate * n)))
        rejected_idx = eligible_order[: min(k_target, len(eligible_order))]
        rejected_mask = np.zeros(n, dtype=bool)
        rejected_mask[rejected_idx] = True
        rejected_f1 = f1[rejected_mask]
        kept_f1 = f1[~rejected_mask]
        tp = int((rejected_f1 < target_threshold).sum()) if rejected_f1.size else 0
        fp = int(rejected_f1.size - tp)
        fn = int(n_bad - tp)
        tn = int(n - rejected_f1.size - fn)
        out.append(
            {
                "reject_rate": float(rejected_f1.size / max(1, n)),
                "target_reject_rate": float(rate),
                "n_eligible": int(eligible.sum()),
                "n_rejected": int(rejected_f1.size),
                "precision_lt_target": float(tp / max(1, rejected_f1.size)),
                "recall_lt_target": float(tp / max(1, n_bad)),
                "kept_f1_p10": float(np.percentile(kept_f1, 10)) if kept_f1.size else float("nan"),
                "rejected_f1_mean": float(rejected_f1.mean()) if rejected_f1.size else float("nan"),
                "binary_accuracy": float((tp + tn) / max(1, n)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return out


def _rank_thresholds(score: np.ndarray, reject_rates: Iterable[float]) -> list[dict]:
    n = len(score)
    sorted_scores = np.sort(score)[::-1]
    out: list[dict] = []
    for rate in reject_rates:
        k = max(1, int(round(rate * n)))
        threshold = float(sorted_scores[k - 1])
        out.append(
            {
                "reject_rate": float(rate),
                "n_flagged_on_validation": int((score >= threshold).sum()),
                "rank_score_threshold": threshold,
            }
        )
    return out


def _score_candidate(
    name: str,
    feature_set: str,
    threshold: float,
    pipe: Pipeline,
    df: pd.DataFrame,
    y: pd.Series,
    f1: pd.Series,
    split: SplitInfo,
    selection_rate: float,
) -> tuple[CandidateResult, np.ndarray, np.ndarray]:
    val_score = pipe.predict_proba(df.loc[split.val])[:, 1]
    test_score = pipe.predict_proba(df.loc[split.test])[:, 1]
    val_y = y.loc[split.val].to_numpy(dtype=int)
    test_y = y.loc[split.test].to_numpy(dtype=int)
    val_f1 = f1.loc[split.val].to_numpy(dtype=float)

    val_rej = _rejection_metrics(val_f1, val_score, [selection_rate], threshold)[0]
    res = CandidateResult(
        name=name,
        feature_set=feature_set,
        threshold=threshold,
        val_average_precision=_safe_ap(val_y, val_score),
        val_roc_auc=_safe_auc(val_y, val_score),
        val_precision_at_selection_rate=val_rej["precision_lt_target"],
        val_kept_f1_p10_at_selection_rate=val_rej["kept_f1_p10"],
        test_average_precision=_safe_ap(test_y, test_score),
        test_roc_auc=_safe_auc(test_y, test_score),
    )
    return res, val_score, test_score


def _calibrate_prefit(pipe: Pipeline, df: pd.DataFrame, y: pd.Series, val_idx: list[int]):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal = CalibratedClassifierCV(pipe, method="isotonic", cv="prefit")
        cal.fit(df.loc[val_idx], y.loc[val_idx])
    return cal


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:5.1f}%"


def _write_txt_report(report: dict, path: Path) -> None:
    lines: list[str] = []
    p = lines.append
    p("=" * 88)
    p("  LEARNED PER-CELL FLAG MODEL")
    p("=" * 88)
    p("")
    p(f"Input rows: {report['n_rows']}")
    p(f"Split: train={report['split_sizes']['train']}  val={report['split_sizes']['val']}  test={report['split_sizes']['test']}")
    p(f"Selection reject rate: {_fmt_pct(report['selection_reject_rate'])}")
    p("")
    p("Targets:")
    for t in report["targets"]:
        p(
            f"  F1 < {t['threshold']:.2f}: base train={_fmt_pct(t['base_rate_train'])} "
            f"val={_fmt_pct(t['base_rate_val'])} test={_fmt_pct(t['base_rate_test'])}; "
            f"selected={t['selected_model']} ({t['selected_feature_set']})"
        )
    p("")

    for t in report["targets"]:
        p("-" * 88)
        p(
            f"Target: F1 < {t['threshold']:.2f}  "
            f"(selected: {t['selected_model']}, features: {t['selected_feature_set']})"
        )
        p("")
        p("Candidates:")
        p(
            f"  {'model':<24} {'val AP':>8} {'val AUC':>8} "
            f"{'val prec@sel':>13} {'val kept P10':>13} "
            f"{'test AP':>8} {'test AUC':>8}"
        )
        for c in t["candidates"]:
            mark = "*" if c["selected"] else " "
            display_name = f"{c['name']}:{c['feature_set']}"
            p(
                f"{mark} {display_name:<23} "
                f"{c['val_average_precision']:>8.4f} {c['val_roc_auc']:>8.4f} "
                f"{c['val_precision_at_selection_rate']:>13.4f} "
                f"{c['val_kept_f1_p10_at_selection_rate']:>13.4f} "
                f"{c['test_average_precision']:>8.4f} {c['test_roc_auc']:>8.4f}"
            )
        p("")
        p("Test rejection sweep using raw rank score:")
        p(
            f"  {'reject':>7} {'n':>5} {'prec<tgt':>10} {'prec<.5':>9} "
            f"{'prec<.6':>9} {'rec<tgt':>9} {'accuracy':>9} {'kept P10':>10} {'rej mean F1':>12}"
        )
        for r in t["test_rejection"]:
            p(
                f"  {_fmt_pct(r['reject_rate']):>7} {r['n_rejected']:>5} "
                f"{_fmt_pct(r['precision_lt_target']):>10} "
                f"{_fmt_pct(r['precision_lt_0_5']):>9} "
                f"{_fmt_pct(r['precision_lt_0_6']):>9} "
                f"{_fmt_pct(r['recall_lt_target']):>9} "
                f"{_fmt_pct(r['binary_accuracy']):>9} "
                f"{r['kept_f1_p10']:>10.4f} {r['rejected_f1_mean']:>12.4f}"
            )
        p("")
        if t.get("test_stage1_pyramidal_only_rejection"):
            p("Test rejection sweep, deployment guard: rank only Stage1-pyramidal cells:")
            p(
                f"  {'target':>7} {'actual':>7} {'n':>5} {'eligible':>8} "
                f"{'prec<tgt':>10} {'rec<tgt':>9} {'accuracy':>9} {'kept P10':>10}"
            )
            for r in t["test_stage1_pyramidal_only_rejection"]:
                p(
                    f"  {_fmt_pct(r['target_reject_rate']):>7} "
                    f"{_fmt_pct(r['reject_rate']):>7} {r['n_rejected']:>5} "
                    f"{r['n_eligible']:>8} {_fmt_pct(r['precision_lt_target']):>10} "
                    f"{_fmt_pct(r['recall_lt_target']):>9} "
                    f"{_fmt_pct(r['binary_accuracy']):>9} "
                    f"{r['kept_f1_p10']:>10.4f}"
                )
            p("")
        if t.get("test_at_precision_thresholds"):
            p("Validation-derived high-precision thresholds, applied to test:")
            p(
                f"  {'target P':>9} {'val P':>9} {'score thr':>10} "
                f"{'test n':>7} {'test P':>8} {'test R':>8} {'accuracy':>9} {'kept P10':>10}"
            )
            for r in t["test_at_precision_thresholds"]:
                p(
                    f"  {_fmt_pct(r['target_precision']):>9} "
                    f"{_fmt_pct(r['validation_precision']):>9} "
                    f"{r['score_threshold']:>10.6f} "
                    f"{r['n_rejected']:>7} "
                    f"{_fmt_pct(r['precision_lt_target']):>8} "
                    f"{_fmt_pct(r['recall_lt_target']):>8} "
                    f"{_fmt_pct(r['binary_accuracy']):>9} "
                    f"{r['kept_f1_p10']:>10.4f}"
                )
            p("")
        if t.get("test_at_recall_thresholds"):
            p("Validation-derived high-recall thresholds, applied to test:")
            p(
                f"  {'target R':>9} {'val R':>9} {'val P':>9} {'score thr':>10} "
                f"{'test n':>7} {'test P':>8} {'test R':>8} {'accuracy':>9} {'kept P10':>10}"
            )
            for r in t["test_at_recall_thresholds"]:
                p(
                    f"  {_fmt_pct(r['target_recall']):>9} "
                    f"{_fmt_pct(r['validation_recall']):>9} "
                    f"{_fmt_pct(r['validation_precision']):>9} "
                    f"{r['score_threshold']:>10.6f} "
                    f"{r['n_rejected']:>7} "
                    f"{_fmt_pct(r['precision_lt_target']):>8} "
                    f"{_fmt_pct(r['recall_lt_target']):>8} "
                    f"{_fmt_pct(r['binary_accuracy']):>9} "
                    f"{r['kept_f1_p10']:>10.4f}"
                )
            p("")
        p("Validation-derived rank-score thresholds:")
        p(f"  {'target reject':>13} {'score threshold':>16} {'val flagged':>12}")
        for r in t["validation_rank_thresholds"]:
            p(
                f"  {_fmt_pct(r['reject_rate']):>13} "
                f"{r['rank_score_threshold']:>16.6f} "
                f"{r['n_flagged_on_validation']:>12}"
            )
        p("")
        if t.get("test_by_cell_type_at_selection"):
            p(f"Test by GT cell type at {_fmt_pct(report['selection_reject_rate'])} rejection:")
            p(
                f"  {'cell type':<12} {'n':>5} {'rejected':>8} "
                f"{'prec<tgt':>10} {'kept P10':>10}"
            )
            for ct, row in t["test_by_cell_type_at_selection"].items():
                p(
                    f"  {ct:<12} {row['n']:>5} {row['n_rejected']:>8} "
                    f"{_fmt_pct(row['precision_lt_target']):>10} "
                    f"{row['kept_f1_p10']:>10.4f}"
                )
            p("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _by_cell_type_at_rate(
    df_test: pd.DataFrame,
    f1: np.ndarray,
    score: np.ndarray,
    rate: float,
    threshold: float,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if "cell_type_gt" not in df_test.columns:
        return out
    n = len(df_test)
    k = max(1, int(round(rate * n)))
    rejected_global = np.zeros(n, dtype=bool)
    rejected_global[np.argsort(-score)[:k]] = True
    cell_types = sorted(str(x) for x in df_test["cell_type_gt"].dropna().unique())
    for ct in cell_types:
        mask = df_test["cell_type_gt"].astype(str).to_numpy() == ct
        if not mask.any():
            continue
        ct_f1 = f1[mask]
        ct_rej = rejected_global[mask]
        kept = ct_f1[~ct_rej]
        rejected = ct_f1[ct_rej]
        out[ct] = {
            "n": int(mask.sum()),
            "n_rejected": int(ct_rej.sum()),
            "precision_lt_target": float((rejected < threshold).mean()) if rejected.size else 0.0,
            "kept_f1_p10": float(np.percentile(kept, 10)) if kept.size else float("nan"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--stage1-disagreement", type=Path, default=DEFAULT_STAGE1_DISAGREE)
    ap.add_argument(
        "--confidence-features",
        type=Path,
        default=DEFAULT_CONFIDENCE_FEATURES,
        help="Optional CSV from paper._build_flag_confidence_features.",
    )
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out-stem", type=Path, default=DEFAULT_OUT_STEM)
    ap.add_argument("--target-thresholds", type=_parse_float_list, default=[0.5, 0.6])
    ap.add_argument("--reject-rates", type=_parse_float_list, default=[0.05, 0.075, 0.10, 0.125, 0.15])
    ap.add_argument(
        "--precision-targets",
        type=_parse_float_list,
        default=[0.8, 0.9],
        help="Validation precision targets used to derive high-precision score thresholds.",
    )
    ap.add_argument(
        "--recall-targets",
        type=_parse_float_list,
        default=[0.8, 0.9, 0.95, 1.0],
        help="Validation recall targets used to derive high-recall score thresholds.",
    )
    ap.add_argument(
        "--selection-reject-rate",
        type=float,
        default=0.05,
        help="Reject rate used to select among candidate models. Default favors high precision.",
    )
    ap.add_argument(
        "--cell-type-filter",
        choices=["all", "pyramidal", "interneuron"],
        default="all",
        help="Optional GT cell-type filter for offline flag-model experiments.",
    )
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    df_raw = _load_data(args.input, args.stage1_disagreement, args.confidence_features)
    if args.cell_type_filter != "all":
        if "cell_type_gt" not in df_raw.columns:
            raise SystemExit("--cell-type-filter requires input column 'cell_type_gt'")
        before = len(df_raw)
        df_raw = df_raw[df_raw["cell_type_gt"].astype(str) == args.cell_type_filter].copy()
        print(
            f"Filtered cell_type_gt={args.cell_type_filter}: {len(df_raw)}/{before} rows",
            flush=True,
        )
    df, numeric_features, categorical_features = _engineer_features(df_raw)
    f1 = pd.to_numeric(df[F1_COL], errors="coerce")
    valid = f1.notna()
    df = df.loc[valid].reset_index(drop=True)
    f1 = f1.loc[valid].reset_index(drop=True)

    split_y = (f1 < 0.6).astype(int)
    split = _build_split(df, split_y, args.seed)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.out_stem.parent.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "input": _display_path(args.input),
        "stage1_disagreement": _display_path(args.stage1_disagreement),
        "confidence_features": (
            _display_path(args.confidence_features)
            if args.confidence_features and args.confidence_features.is_file()
            else ""
        ),
        "n_rows": int(len(df)),
        "cell_type_filter": args.cell_type_filter,
        "seed": args.seed,
        "selection_reject_rate": float(args.selection_reject_rate),
        "reject_rates": [float(x) for x in args.reject_rates],
        "precision_targets": [float(x) for x in args.precision_targets],
        "recall_targets": [float(x) for x in args.recall_targets],
        "split_sizes": {
            "train": len(split.train),
            "val": len(split.val),
            "test": len(split.test),
        },
        "feature_columns": {
            "numeric": numeric_features,
            "categorical": categorical_features,
            "excluded_gt_or_label": sorted(GT_OR_LABEL_COLUMNS),
        },
        "targets": [],
    }

    predictions_out = df.loc[split.test, ["file"]].copy()
    if "cell_type_gt" in df.columns:
        predictions_out["cell_type_gt"] = df.loc[split.test, "cell_type_gt"].to_numpy()
    predictions_out["held_out_F1"] = f1.loc[split.test].to_numpy(dtype=float)

    for threshold in args.target_thresholds:
        y = (f1 < threshold).astype(int)
        candidates: list[tuple[CandidateResult, Pipeline, list[str]]] = []

        print(f"\n=== Training flag target: F1 < {threshold:.2f} ===", flush=True)
        print(
            f"Base rates: train={y.loc[split.train].mean():.3f} "
            f"val={y.loc[split.val].mean():.3f} "
            f"test={y.loc[split.test].mean():.3f}",
            flush=True,
        )

        for feature_set, cat_for_set in _feature_sets(categorical_features).items():
            for name, estimator in _candidate_models(args.seed).items():
                preprocessor = _make_preprocessor(numeric_features, cat_for_set)
                pipe = Pipeline(
                    steps=[
                        ("preprocess", preprocessor),
                        ("model", estimator),
                    ]
                )
                pipe.fit(df.loc[split.train], y.loc[split.train])
                res, _, _ = _score_candidate(
                    name,
                    feature_set,
                    threshold,
                    pipe,
                    df,
                    y,
                    f1,
                    split,
                    args.selection_reject_rate,
                )
                print(
                    f"  {name + ':' + feature_set:<24} "
                    f"val AP={res.val_average_precision:.4f} "
                    f"val prec@{args.selection_reject_rate:.3f}="
                    f"{res.val_precision_at_selection_rate:.4f} "
                    f"test AP={res.test_average_precision:.4f}",
                    flush=True,
                )
                candidates.append((res, pipe, cat_for_set))

        # High precision at the operating reject rate is the primary goal.
        # AP breaks ties so the chosen model still ranks the whole tail well.
        best_res, best_pipe, best_categorical_features = max(
            candidates,
            key=lambda item: (
                item[0].val_precision_at_selection_rate,
                item[0].val_kept_f1_p10_at_selection_rate,
                item[0].val_average_precision,
            ),
        )
        best_res.selected = True

        calibrated = _calibrate_prefit(best_pipe, df, y, split.val)
        # Use the raw model score for ranking/rejection. Isotonic calibration is
        # monotone in expectation but can create large tied plateaus, which makes
        # top-K rejection unstable. The calibrated value is still useful as the
        # displayed P(bad).
        val_rank_score = best_pipe.predict_proba(df.loc[split.val])[:, 1]
        test_rank_score = best_pipe.predict_proba(df.loc[split.test])[:, 1]
        test_prob_bad = calibrated.predict_proba(df.loc[split.test])[:, 1]
        test_y = y.loc[split.test].to_numpy(dtype=int)
        val_f1 = f1.loc[split.val].to_numpy(dtype=float)
        test_f1 = f1.loc[split.test].to_numpy(dtype=float)
        precision_thresholds = _precision_thresholds(
            val_f1,
            val_rank_score,
            threshold,
            args.precision_targets,
        )
        recall_thresholds = _recall_thresholds(
            val_f1,
            val_rank_score,
            threshold,
            args.recall_targets,
        )

        model_suffix = f"f{int(round(threshold * 100)):03d}"
        model_path = args.model_dir / f"flag_model_{model_suffix}.joblib"
        joblib.dump(
            {
                "rank_model": best_pipe,
                "calibrated_model": calibrated,
                "target_threshold": float(threshold),
                "numeric_features": numeric_features,
                "categorical_features": best_categorical_features,
                "seed": args.seed,
                "selection_reject_rate": float(args.selection_reject_rate),
                "selected_model_name": best_res.name,
                "selected_feature_set": best_res.feature_set,
                "cell_type_filter": args.cell_type_filter,
                "validation_rank_thresholds": _rank_thresholds(
                    val_rank_score,
                    args.reject_rates,
                ),
                "validation_precision_thresholds": precision_thresholds,
                "validation_recall_thresholds": recall_thresholds,
            },
            model_path,
        )

        for res, _pipe, _cat_features in candidates:
            if res.name == best_res.name and res.feature_set == best_res.feature_set:
                res.selected = True

        target_report = {
            "threshold": float(threshold),
            "base_rate_train": float(y.loc[split.train].mean()),
            "base_rate_val": float(y.loc[split.val].mean()),
            "base_rate_test": float(y.loc[split.test].mean()),
            "selected_model": best_res.name,
            "selected_feature_set": best_res.feature_set,
            "model_path": _display_path(model_path),
            "raw_rank_test_average_precision": _safe_ap(test_y, test_rank_score),
            "raw_rank_test_roc_auc": _safe_auc(test_y, test_rank_score),
            "calibrated_test_average_precision": _safe_ap(test_y, test_prob_bad),
            "calibrated_test_roc_auc": _safe_auc(test_y, test_prob_bad),
            "validation_rank_thresholds": _rank_thresholds(
                val_rank_score,
                args.reject_rates,
            ),
            "validation_precision_thresholds": precision_thresholds,
            "test_at_precision_thresholds": _apply_precision_thresholds(
                test_f1,
                test_rank_score,
                threshold,
                precision_thresholds,
            ),
            "validation_recall_thresholds": recall_thresholds,
            "test_at_recall_thresholds": _apply_recall_thresholds(
                test_f1,
                test_rank_score,
                threshold,
                recall_thresholds,
            ),
            "candidates": [asdict(res) for res, _pipe, _cat_features in candidates],
            "test_rejection": _rejection_metrics(
                test_f1,
                test_rank_score,
                args.reject_rates,
                threshold,
            ),
            "test_stage1_pyramidal_only_rejection": _stage1_pyramidal_guarded_rejection_metrics(
                df.loc[split.test].reset_index(drop=True),
                test_f1,
                test_rank_score,
                args.reject_rates,
                threshold,
            ),
            "test_by_cell_type_at_selection": _by_cell_type_at_rate(
                df.loc[split.test].reset_index(drop=True),
                test_f1,
                test_rank_score,
                args.selection_reject_rate,
                threshold,
            ),
        }
        report["targets"].append(target_report)
        predictions_out[f"rank_score_{model_suffix}"] = test_rank_score
        predictions_out[f"prob_bad_{model_suffix}"] = test_prob_bad
        predictions_out[f"target_{model_suffix}"] = test_y

    metadata_path = args.model_dir / "metadata.json"
    metadata_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    json_path = args.out_stem.with_suffix(".json")
    txt_path = args.out_stem.with_suffix(".txt")
    csv_path = args.out_stem.parent / f"{args.out_stem.name}_test_predictions.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_txt_report(report, txt_path)
    predictions_out.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print("\nWrote:")
    print(f"  {_display_path(metadata_path)}")
    print(f"  {_display_path(json_path)}")
    print(f"  {_display_path(txt_path)}")
    print(f"  {_display_path(csv_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
