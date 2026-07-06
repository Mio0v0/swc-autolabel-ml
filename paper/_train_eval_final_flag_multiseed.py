#!/usr/bin/env python3
"""No-leak leave-one-seed-out flag model training/evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper._train_flag_model import (  # noqa: E402
    F1_COL,
    SplitInfo,
    _apply_precision_thresholds,
    _apply_recall_thresholds,
    _candidate_models,
    _engineer_features,
    _feature_sets,
    _make_preprocessor,
    _precision_thresholds,
    _rank_thresholds,
    _recall_thresholds,
    _rejection_metrics,
    _safe_ap,
    _safe_auc,
)


TARGET_THRESHOLD = 0.6
DEFAULT_RECALL_TARGETS = (0.5, 0.7, 0.8, 0.9)
DEFAULT_REJECT_RATES = (0.05, 0.10, 0.15)
DEFAULT_PRECISION_TARGETS = (0.7, 0.8, 0.9)
FEATURE_MODES = ("compact", "branch3", "baseline_oof", "v12_oof")

warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")


def _display(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def _parse_csv_list(raw: str, cast=str) -> list:
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def _load_joined(labels_path: Path, features_path: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    features = pd.read_csv(features_path)
    required = {"row_id", "model_seed", "file"}
    if not required.issubset(labels.columns) or not required.issubset(features.columns):
        raise SystemExit("labels/features must include row_id, model_seed, file")
    df = labels.merge(features, on=["row_id", "model_seed", "file"], how="left", suffixes=("", "_feature"))
    if len(df) != len(labels):
        raise SystemExit("labels/features merge changed row count")
    df = df.copy()
    for col in ("seed42_pred", "seed789_pred"):
        if col not in df.columns:
            df[col] = ""
    for col in ("seed42_conf", "seed789_conf"):
        if col not in df.columns:
            df[col] = np.nan
    if "source" not in df.columns:
        df["source"] = df["file"].astype(str).str.split("__", n=1).str[0]
    return df


def _feature_mode_frame(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = df.copy()
    drop: list[str] = []
    for col in out.columns:
        is_branch3 = col.startswith("branch3_")
        is_baseline = col.startswith("baseline_oof_")
        is_v12 = col.startswith("xmodel_v12_oof_")
        if mode == "compact" and (is_branch3 or is_baseline or is_v12):
            drop.append(col)
        elif mode == "branch3" and (is_baseline or is_v12):
            drop.append(col)
        elif mode == "baseline_oof" and is_v12:
            drop.append(col)
        elif mode == "v12_oof":
            pass
        elif mode not in FEATURE_MODES:
            raise KeyError(mode)
    out = out.drop(columns=drop, errors="ignore")
    if mode == "v12_oof":
        v12_cols = [c for c in out.columns if c.startswith("xmodel_v12_oof_")]
        for col in v12_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def _group_val_split(df: pd.DataFrame, y: pd.Series, idx: list[int], seed: int) -> tuple[list[int], list[int]]:
    pool = pd.DataFrame({"idx": idx, "file": df.loc[idx, "file"].astype(str), "y": y.loc[idx].astype(int)})
    grouped = pool.groupby("file", as_index=False)["y"].max()
    strat = grouped["y"] if grouped["y"].nunique() == 2 and grouped["y"].value_counts().min() >= 2 else None
    train_files, val_files = train_test_split(
        grouped["file"],
        test_size=0.25,
        random_state=seed,
        stratify=strat,
    )
    train_set = set(train_files.astype(str))
    val_set = set(val_files.astype(str))
    train_idx = [int(i) for i in idx if str(df.loc[i, "file"]) in train_set]
    val_idx = [int(i) for i in idx if str(df.loc[i, "file"]) in val_set]
    return train_idx, val_idx


def _loso_split(df: pd.DataFrame, y: pd.Series, test_seed: int, seed: int) -> tuple[SplitInfo, dict]:
    test_idx = df.index[df["model_seed"].astype(int) == int(test_seed)].tolist()
    test_files = set(df.loc[test_idx, "file"].astype(str))
    pool_idx = df.index[
        (df["model_seed"].astype(int) != int(test_seed))
        & (~df["file"].astype(str).isin(test_files))
    ].tolist()
    train_idx, val_idx = _group_val_split(df, y, pool_idx, seed + int(test_seed))
    leak = {
        "test_seed": int(test_seed),
        "train": len(train_idx),
        "val": len(val_idx),
        "test": len(test_idx),
        "train_val_dropped_for_test_file_overlap": int(
            ((df["model_seed"].astype(int) != int(test_seed)) & df["file"].astype(str).isin(test_files)).sum()
        ),
        "train_test_file_overlap": len(set(df.loc[train_idx, "file"].astype(str)) & test_files),
        "val_test_file_overlap": len(set(df.loc[val_idx, "file"].astype(str)) & test_files),
        "train_val_file_overlap": len(set(df.loc[train_idx, "file"].astype(str)) & set(df.loc[val_idx, "file"].astype(str))),
    }
    return SplitInfo(train=train_idx, val=val_idx, test=test_idx), leak


def _fit_candidates(
    df_feat: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    y: pd.Series,
    split: SplitInfo,
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


def _select_candidate(
    candidates: list[dict],
    val_f1: np.ndarray,
    recall_target: float,
) -> tuple[dict, dict]:
    scored: list[tuple[tuple[float, float, float], dict, dict]] = []
    for cand in candidates:
        val_rows = _recall_thresholds(val_f1, cand["val_score"], TARGET_THRESHOLD, [recall_target])
        val = val_rows[0]
        scored.append(
            (
                (
                    float(val["validation_precision"]),
                    float(cand["val_ap"]) if not np.isnan(cand["val_ap"]) else -1.0,
                    float(cand["val_auc"]) if not np.isnan(cand["val_auc"]) else -1.0,
                ),
                cand,
                val,
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    _key, cand, val = scored[0]
    return cand, val


def _metric_row_base(
    scope: str,
    feature_mode: str,
    test_seed: int,
    split: SplitInfo,
    df_feat: pd.DataFrame,
    f1: pd.Series,
    y: pd.Series,
    cand: dict,
) -> dict:
    test_y = y.loc[split.test].to_numpy(dtype=int)
    test_f1 = f1.loc[split.test].to_numpy(dtype=float)
    return {
        "scope": scope,
        "feature_mode": feature_mode,
        "test_seed": int(test_seed),
        "selected_model": cand["model_name"],
        "selected_feature_set": cand["feature_set"],
        "n_train": len(split.train),
        "n_val": len(split.val),
        "n_test": len(split.test),
        "n_bad_test": int(test_y.sum()),
        "bad_rate_test": float(test_y.mean()) if len(test_y) else float("nan"),
        "before_f1_mean": float(test_f1.mean()) if len(test_f1) else float("nan"),
        "before_f1_p10": float(np.percentile(test_f1, 10)) if len(test_f1) else float("nan"),
        "test_ap": _safe_ap(test_y, cand["test_score"]),
        "test_auc": _safe_auc(test_y, cand["test_score"]),
    }


def _append_eval_rows(
    rows: list[dict],
    before_after: list[dict],
    base: dict,
    test_f1: np.ndarray,
    test_score: np.ndarray,
    val_f1: np.ndarray,
    val_score: np.ndarray,
    recall_targets: list[float],
    reject_rates: list[float],
    precision_targets: list[float],
) -> None:
    val_recall_rows = _recall_thresholds(val_f1, val_score, TARGET_THRESHOLD, recall_targets)
    test_recall_rows = _apply_recall_thresholds(test_f1, test_score, TARGET_THRESHOLD, val_recall_rows)
    val_precision_rows = _precision_thresholds(val_f1, val_score, TARGET_THRESHOLD, precision_targets)
    test_precision_rows = _apply_precision_thresholds(test_f1, test_score, TARGET_THRESHOLD, val_precision_rows)
    test_rate_rows = _rejection_metrics(test_f1, test_score, reject_rates, TARGET_THRESHOLD)

    for val, test in zip(val_recall_rows, test_recall_rows):
        row = {
            **base,
            "operating_point": "validation_recall",
            "target_value": float(val["target_recall"]),
            "validation_precision": float(val["validation_precision"]),
            "validation_recall": float(val["validation_recall"]),
            **{k: test.get(k) for k in ("score_threshold", "reject_rate", "n_rejected", "precision_lt_target", "recall_lt_target", "binary_accuracy", "kept_f1_p10", "rejected_f1_mean", "tp", "fp", "fn", "tn")},
        }
        rows.append(row)
        if abs(float(val["target_recall"]) - 0.8) < 1e-9:
            before_after.append(row)

    for val, test in zip(val_precision_rows, test_precision_rows):
        rows.append(
            {
                **base,
                "operating_point": "validation_precision",
                "target_value": float(val["target_precision"]),
                "validation_precision": float(val["validation_precision"]),
                "validation_recall": float(val["validation_recall"]),
                **{k: test.get(k) for k in ("score_threshold", "reject_rate", "n_rejected", "precision_lt_target", "recall_lt_target", "binary_accuracy", "kept_f1_p10", "rejected_f1_mean", "tp", "fp", "fn", "tn")},
            }
        )

    for test in test_rate_rows:
        row = {
            **base,
            "operating_point": "fixed_reject_rate",
            "target_value": float(test["reject_rate"]),
            "validation_precision": float("nan"),
            "validation_recall": float("nan"),
            **{k: test.get(k) for k in ("reject_rate", "n_rejected", "precision_lt_target", "recall_lt_target", "binary_accuracy", "kept_f1_mean", "kept_f1_p10", "kept_f1_p25", "rejected_f1_mean", "tp", "fp", "fn", "tn")},
        }
        rows.append(row)
        if abs(float(test["reject_rate"]) - 0.10) < 1e-9:
            before_after.append(row)


def _summary_table(rows: pd.DataFrame) -> pd.DataFrame:
    keep = rows[rows["operating_point"].isin(["validation_recall", "fixed_reject_rate"])].copy()
    metrics = [
        "precision_lt_target",
        "recall_lt_target",
        "binary_accuracy",
        "reject_rate",
        "n_rejected",
        "tp",
        "fp",
        "fn",
        "kept_f1_p10",
        "test_ap",
        "test_auc",
    ]
    group_cols = ["scope", "feature_mode", "operating_point", "target_value"]
    out_rows: list[dict] = []
    for keys, part in keep.groupby(group_cols, dropna=False):
        rec = dict(zip(group_cols, keys))
        rec["n_folds"] = int(len(part))
        for metric in metrics:
            vals = pd.to_numeric(part[metric], errors="coerce")
            rec[f"{metric}_mean"] = float(vals.mean())
            rec[f"{metric}_sd"] = float(vals.std(ddof=1)) if len(vals.dropna()) > 1 else 0.0
        out_rows.append(rec)
    return pd.DataFrame(out_rows)


def _write_txt_summary(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "Final multi-seed flag leave-one-seed-out summary",
        "=" * 72,
        "Target bad label: per-cell neurite F1 < 0.60",
        "",
    ]
    if summary.empty:
        lines.append("No rows.")
    else:
        view = summary.sort_values(["scope", "feature_mode", "operating_point", "target_value"])
        for _, row in view.iterrows():
            lines.append(
                f"{row['scope']:<11} {row['feature_mode']:<13} "
                f"{row['operating_point']:<19} {float(row['target_value']):>4.2f} | "
                f"P {row['precision_lt_target_mean']:.3f}+/-{row['precision_lt_target_sd']:.3f} "
                f"R {row['recall_lt_target_mean']:.3f}+/-{row['recall_lt_target_sd']:.3f} "
                f"flag {row['n_rejected_mean']:.1f} "
                f"fp {row['fp_mean']:.1f} fn {row['fn_mean']:.1f} "
                f"keptP10 {row['kept_f1_p10_mean']:.4f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=ROOT / "paper" / "results" / "final_flag_multiseed_labels.csv")
    ap.add_argument(
        "--features",
        type=Path,
        default=ROOT / "paper" / "results" / "final_flag_multiseed_features_oof_heavy.csv",
    )
    ap.add_argument("--out-csv", type=Path, default=ROOT / "paper" / "results" / "final_flag_leave_one_seed_out.csv")
    ap.add_argument("--out-summary", type=Path, default=ROOT / "paper" / "results" / "final_flag_leave_one_seed_out_summary.csv")
    ap.add_argument("--out-before-after", type=Path, default=ROOT / "paper" / "results" / "final_flag_before_after.csv")
    ap.add_argument("--out-json", type=Path, default=ROOT / "paper" / "results" / "final_flag_leave_one_seed_out.json")
    ap.add_argument("--model-dir", type=Path, default=ROOT / "paper" / "models" / "final_flag_multiseed_loso")
    ap.add_argument("--seeds", default="123,42,789")
    ap.add_argument("--scopes", default="all,pyramidal,interneuron")
    ap.add_argument("--feature-modes", default="compact,branch3,baseline_oof,v12_oof")
    ap.add_argument("--recall-targets", default="0.5,0.7,0.8,0.9")
    ap.add_argument("--reject-rates", default="0.05,0.10,0.15")
    ap.add_argument("--precision-targets", default="0.7,0.8,0.9")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    seeds = _parse_csv_list(args.seeds, int)
    scopes = _parse_csv_list(args.scopes, str)
    feature_modes = _parse_csv_list(args.feature_modes, str)
    recall_targets = _parse_csv_list(args.recall_targets, float)
    reject_rates = _parse_csv_list(args.reject_rates, float)
    precision_targets = _parse_csv_list(args.precision_targets, float)

    df_loaded = _load_joined(args.labels, args.features)
    rows: list[dict] = []
    before_after: list[dict] = []
    leak_checks: list[dict] = []
    saved_models: list[str] = []
    args.model_dir.mkdir(parents=True, exist_ok=True)

    for scope in scopes:
        df_scope = df_loaded.copy()
        if scope != "all":
            if "cell_type_gt" not in df_scope.columns:
                raise SystemExit("scope filtering requires cell_type_gt")
            df_scope = df_scope[df_scope["cell_type_gt"].astype(str) == scope].copy()
        if df_scope.empty:
            continue
        for feature_mode in feature_modes:
            print(f"\n=== scope={scope} feature_mode={feature_mode} ===", flush=True)
            df_mode = _feature_mode_frame(df_scope, feature_mode)
            df_feat, numeric_features, categorical_features = _engineer_features(df_mode)
            f1 = pd.to_numeric(df_feat[F1_COL], errors="coerce")
            valid = f1.notna()
            df_feat = df_feat.loc[valid].reset_index(drop=True)
            f1 = f1.loc[valid].reset_index(drop=True)
            y = (f1 < TARGET_THRESHOLD).astype(int)
            if int(y.sum()) < 5 or int((1 - y).sum()) < 5:
                print(f"  skip: not enough positives/negatives ({int(y.sum())}/{len(y)})", flush=True)
                continue
            for test_seed in seeds:
                if int((df_feat["model_seed"].astype(int) == test_seed).sum()) == 0:
                    continue
                split, leak = _loso_split(df_feat, y, test_seed, args.seed)
                leak.update({"scope": scope, "feature_mode": feature_mode})
                leak_checks.append(leak)
                if leak["train_test_file_overlap"] or leak["val_test_file_overlap"] or leak["train_val_file_overlap"]:
                    raise SystemExit(f"Flag split leakage detected: {leak}")
                if int(y.loc[split.train].sum()) < 3 or int(y.loc[split.val].sum()) < 1 or int(y.loc[split.test].sum()) < 1:
                    print(f"  seed {test_seed}: skip, positives too sparse in train/val/test", flush=True)
                    continue
                candidates = _fit_candidates(df_feat, numeric_features, categorical_features, y, split, args.seed + int(test_seed))
                val_f1 = f1.loc[split.val].to_numpy(dtype=float)
                test_f1 = f1.loc[split.test].to_numpy(dtype=float)
                cand, _selected_val = _select_candidate(candidates, val_f1, recall_target=0.8)
                test_score = cand["test_score"]
                val_score = cand["val_score"]
                model_path = args.model_dir / f"{scope}_{feature_mode}_testseed{test_seed}_f060.joblib"
                joblib.dump(
                    {
                        "rank_model": cand["pipe"],
                        "target_threshold": TARGET_THRESHOLD,
                        "numeric_features": numeric_features,
                        "categorical_features": cand["cat_features"],
                        "scope": scope,
                        "feature_mode": feature_mode,
                        "test_seed_held_out": int(test_seed),
                        "validation_rank_thresholds": _rank_thresholds(val_score, reject_rates),
                        "validation_precision_thresholds": _precision_thresholds(val_f1, val_score, TARGET_THRESHOLD, precision_targets),
                        "validation_recall_thresholds": _recall_thresholds(val_f1, val_score, TARGET_THRESHOLD, recall_targets),
                    },
                    model_path,
                )
                saved_models.append(_display(model_path))
                base = _metric_row_base(scope, feature_mode, test_seed, split, df_feat, f1, y, cand)
                base["model_path"] = _display(model_path)
                _append_eval_rows(
                    rows,
                    before_after,
                    base,
                    test_f1,
                    test_score,
                    val_f1,
                    val_score,
                    recall_targets,
                    reject_rates,
                    precision_targets,
                )
                print(
                    f"  seed {test_seed}: {cand['model_name']}:{cand['feature_set']} "
                    f"test AP={base['test_ap']:.3f} bad={base['n_bad_test']}/{base['n_test']}",
                    flush=True,
                )

    out = pd.DataFrame(rows)
    summary = _summary_table(out) if not out.empty else pd.DataFrame()
    before = pd.DataFrame(before_after)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    summary.to_csv(args.out_summary, index=False, quoting=csv.QUOTE_MINIMAL)
    before.to_csv(args.out_before_after, index=False, quoting=csv.QUOTE_MINIMAL)
    txt_path = args.out_summary.with_suffix(".txt")
    _write_txt_summary(summary, txt_path)
    payload = {
        "labels": _display(args.labels),
        "features": _display(args.features),
        "target_threshold": TARGET_THRESHOLD,
        "seeds": seeds,
        "scopes": scopes,
        "feature_modes": feature_modes,
        "recall_targets": recall_targets,
        "reject_rates": reject_rates,
        "precision_targets": precision_targets,
        "n_rows": int(len(df_loaded)),
        "saved_models": saved_models,
        "leak_checks": leak_checks,
        "outputs": {
            "csv": _display(args.out_csv),
            "summary": _display(args.out_summary),
            "summary_txt": _display(txt_path),
            "before_after": _display(args.out_before_after),
        },
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {_display(args.out_csv)}")
    print(f"Wrote {_display(args.out_summary)}")
    print(f"Wrote {_display(txt_path)}")
    print(f"Wrote {_display(args.out_before_after)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
