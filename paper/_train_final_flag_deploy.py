#!/usr/bin/env python3
"""Train deployable multi-seed flag bundles.

The leave-one-seed-out flag models are for paper evaluation only. This
script trains production-shaped bundles from the multi-seed table, using a
grouped file split for model selection/calibration so duplicate seed rows for
the same SWC do not cross train/validation/test during bundle construction.
"""
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper._train_eval_final_flag_multiseed import (  # noqa: E402
    TARGET_THRESHOLD,
    _feature_mode_frame,
)
from paper._train_flag_model import (  # noqa: E402
    F1_COL,
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


def _display(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def _split_by_file(df: pd.DataFrame, y: pd.Series, seed: int) -> tuple[list[int], list[int], list[int]]:
    grouped = pd.DataFrame({"file": df["file"].astype(str), "y": y.astype(int)})
    by_file = grouped.groupby("file", as_index=False)["y"].max()
    strat = by_file["y"] if by_file["y"].nunique() == 2 and by_file["y"].value_counts().min() >= 2 else None
    train_val_files, test_files = train_test_split(
        by_file["file"],
        test_size=0.20,
        random_state=seed,
        stratify=strat,
    )
    train_val = by_file[by_file["file"].isin(set(train_val_files))]
    strat_tv = (
        train_val["y"]
        if train_val["y"].nunique() == 2 and train_val["y"].value_counts().min() >= 2
        else None
    )
    train_files, val_files = train_test_split(
        train_val["file"],
        test_size=0.25,
        random_state=seed,
        stratify=strat_tv,
    )
    train_set = set(train_files.astype(str))
    val_set = set(val_files.astype(str))
    test_set = set(test_files.astype(str))
    return (
        [int(i) for i in df.index if str(df.loc[i, "file"]) in train_set],
        [int(i) for i in df.index if str(df.loc[i, "file"]) in val_set],
        [int(i) for i in df.index if str(df.loc[i, "file"]) in test_set],
    )


def _load_joined(labels_path: Path, features_path: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    features = pd.read_csv(features_path)
    keys = ["row_id", "model_seed", "file"]
    df = labels.merge(features, on=keys, how="left", suffixes=("", "_feature"))
    if len(df) != len(labels):
        raise SystemExit("labels/features merge changed row count")
    for col in ("seed42_pred", "seed789_pred"):
        if col not in df.columns:
            df[col] = ""
    for col in ("seed42_conf", "seed789_conf"):
        if col not in df.columns:
            df[col] = np.nan
    return df


def _fit_scope(
    df_loaded: pd.DataFrame,
    *,
    scope: str,
    feature_mode: str,
    out_dir: Path,
    seed: int,
    reject_rates: list[float],
    precision_targets: list[float],
    recall_targets: list[float],
    selection_reject_rate: float,
) -> dict:
    df_scope = df_loaded.copy()
    if scope != "all":
        df_scope = df_scope[df_scope["cell_type_gt"].astype(str) == scope].copy()
    df_mode = _feature_mode_frame(df_scope, feature_mode)
    df_feat, numeric_features, categorical_features = _engineer_features(df_mode)
    f1 = pd.to_numeric(df_feat[F1_COL], errors="coerce")
    valid = f1.notna()
    df_feat = df_feat.loc[valid].reset_index(drop=True)
    f1 = f1.loc[valid].reset_index(drop=True)
    y = (f1 < TARGET_THRESHOLD).astype(int)
    if int(y.sum()) < 5 or int((1 - y).sum()) < 5:
        raise SystemExit(f"not enough positives/negatives for scope={scope}: {int(y.sum())}/{len(y)}")

    train_idx, val_idx, test_idx = _split_by_file(df_feat, y, seed)
    leak = {
        "train_val_file_overlap": len(set(df_feat.loc[train_idx, "file"]) & set(df_feat.loc[val_idx, "file"])),
        "train_test_file_overlap": len(set(df_feat.loc[train_idx, "file"]) & set(df_feat.loc[test_idx, "file"])),
        "val_test_file_overlap": len(set(df_feat.loc[val_idx, "file"]) & set(df_feat.loc[test_idx, "file"])),
    }
    if any(leak.values()):
        raise SystemExit(f"split leakage detected for {scope}: {leak}")

    candidates: list[dict] = []
    for feature_set, cat_for_set in _feature_sets(categorical_features).items():
        for model_name, estimator in _candidate_models(seed).items():
            pipe = Pipeline(
                [
                    ("preprocess", _make_preprocessor(numeric_features, cat_for_set)),
                    ("model", estimator),
                ]
            )
            pipe.fit(df_feat.loc[train_idx], y.loc[train_idx])
            val_score = pipe.predict_proba(df_feat.loc[val_idx])[:, 1]
            val_rej = _rejection_metrics(
                f1.loc[val_idx].to_numpy(dtype=float),
                val_score,
                [selection_reject_rate],
                TARGET_THRESHOLD,
            )[0]
            candidates.append(
                {
                    "model_name": model_name,
                    "feature_set": feature_set,
                    "cat_features": cat_for_set,
                    "pipe": pipe,
                    "val_score": val_score,
                    "val_ap": _safe_ap(y.loc[val_idx].to_numpy(dtype=int), val_score),
                    "val_auc": _safe_auc(y.loc[val_idx].to_numpy(dtype=int), val_score),
                    "val_precision_at_selection": val_rej["precision_lt_target"],
                    "val_kept_f1_p10_at_selection": val_rej["kept_f1_p10"],
                }
            )

    best = max(
        candidates,
        key=lambda c: (
            c["val_precision_at_selection"],
            c["val_kept_f1_p10_at_selection"],
            c["val_ap"] if not np.isnan(c["val_ap"]) else -1.0,
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrated = CalibratedClassifierCV(best["pipe"], method="isotonic", cv="prefit")
        calibrated.fit(df_feat.loc[val_idx], y.loc[val_idx])

    val_f1 = f1.loc[val_idx].to_numpy(dtype=float)
    test_f1 = f1.loc[test_idx].to_numpy(dtype=float)
    test_y = y.loc[test_idx].to_numpy(dtype=int)
    test_score = best["pipe"].predict_proba(df_feat.loc[test_idx])[:, 1]
    test_prob = calibrated.predict_proba(df_feat.loc[test_idx])[:, 1]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"flag_model_{scope}_{feature_mode}_f060.joblib"
    bundle = {
        "rank_model": best["pipe"],
        "calibrated_model": calibrated,
        "target_threshold": float(TARGET_THRESHOLD),
        "numeric_features": numeric_features,
        "categorical_features": best["cat_features"],
        "seed": int(seed),
        "selection_reject_rate": float(selection_reject_rate),
        "selected_model_name": best["model_name"],
        "selected_feature_set": best["feature_set"],
        "feature_mode": feature_mode,
        "cell_type_filter": scope,
        "validation_rank_thresholds": _rank_thresholds(best["val_score"], reject_rates),
        "validation_precision_thresholds": _precision_thresholds(
            val_f1,
            best["val_score"],
            TARGET_THRESHOLD,
            precision_targets,
        ),
        "validation_recall_thresholds": _recall_thresholds(
            val_f1,
            best["val_score"],
            TARGET_THRESHOLD,
            recall_targets,
        ),
    }
    joblib.dump(bundle, out_path)
    return {
        "scope": scope,
        "feature_mode": feature_mode,
        "model_path": _display(out_path),
        "n_rows": int(len(df_feat)),
        "n_bad": int(y.sum()),
        "split": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "leak": leak,
        "selected_model": best["model_name"],
        "selected_feature_set": best["feature_set"],
        "n_numeric_features": len(numeric_features),
        "n_categorical_features": len(best["cat_features"]),
        "test_ap_rank": _safe_ap(test_y, test_score),
        "test_auc_rank": _safe_auc(test_y, test_score),
        "test_ap_calibrated": _safe_ap(test_y, test_prob),
        "test_auc_calibrated": _safe_auc(test_y, test_prob),
        "test_fixed_reject": _rejection_metrics(test_f1, test_score, reject_rates, TARGET_THRESHOLD),
    }


def _parse_list(raw: str, cast=str) -> list:
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=ROOT / "paper" / "results" / "final_flag_multiseed_labels.csv")
    ap.add_argument("--features", type=Path, default=ROOT / "paper" / "results" / "final_flag_multiseed_features_oof_heavy.csv")
    ap.add_argument("--model-dir", type=Path, default=ROOT / "paper" / "models" / "final_flag_multiseed_deploy")
    ap.add_argument("--out-json", type=Path, default=ROOT / "paper" / "results" / "final_flag_multiseed_deploy.json")
    ap.add_argument("--scopes", default="all,pyramidal")
    ap.add_argument("--feature-mode", default="baseline_oof")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--reject-rates", default="0.05,0.10,0.15")
    ap.add_argument("--precision-targets", default="0.7,0.8,0.9")
    ap.add_argument("--recall-targets", default="0.5,0.7,0.8,0.9")
    ap.add_argument("--selection-reject-rate", type=float, default=0.05)
    args = ap.parse_args()

    df = _load_joined(args.labels, args.features)
    rows = []
    for scope in _parse_list(args.scopes):
        print(f"Training deploy flagger: scope={scope} feature_mode={args.feature_mode}", flush=True)
        row = _fit_scope(
            df,
            scope=scope,
            feature_mode=args.feature_mode,
            out_dir=args.model_dir,
            seed=args.seed,
            reject_rates=_parse_list(args.reject_rates, float),
            precision_targets=_parse_list(args.precision_targets, float),
            recall_targets=_parse_list(args.recall_targets, float),
            selection_reject_rate=args.selection_reject_rate,
        )
        rows.append(row)
        print(
            f"  saved {row['model_path']}  AP={row['test_ap_rank']:.3f}  "
            f"features={row['n_numeric_features']}+{row['n_categorical_features']}",
            flush=True,
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "labels": _display(args.labels),
                "features": _display(args.features),
                "target_threshold": TARGET_THRESHOLD,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (args.out_json.with_suffix(".csv")).open("w", encoding="utf-8", newline="") as fh:
        flat_rows = [
            {k: v for k, v in row.items() if not isinstance(v, (list, dict))}
            for row in rows
        ]
        writer = csv.DictWriter(fh, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"Wrote {_display(args.out_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
