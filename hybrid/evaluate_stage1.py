#!/usr/bin/env python3
"""Evaluate Stage 1 only with a clean train/test split.

Usage:
    python -m hybrid.evaluate_stage1 --data-dir data/benchmark_pyramidal_interneuron_v1_qc_diag_pruned
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid.features import FEATURE_NAMES
from hybrid.train_stage1 import _collect_dataset, build_pipeline


def _save_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _write_misclassified_csv(path: Path, rows: list[dict[str, object]], class_names: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "ground_truth", "predicted", "confidence", *class_names])
        for row in rows:
            probs = row["probabilities"]
            writer.writerow([
                row["path"],
                row["ground_truth"],
                row["predicted"],
                row["confidence"],
                *[probs[name] for name in class_names],
            ])


def _write_feature_importance_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["feature", "importance_mean", "importance_std"])
        for row in rows:
            importance_std = row.get("importance_std", row.get("importance_std_across_seeds", ""))
            writer.writerow([row["feature"], row["importance_mean"], importance_std])


def _write_recurring_error_diagnostics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "path",
            "ground_truth",
            "predicted",
            "n_misclassified",
            "n_tested",
            "misclassification_rate_when_tested",
            "mean_confidence_when_wrong",
            "top_feature_drivers",
        ])
        for row in rows:
            top_features = "; ".join(
                (
                    f"{item['feature']}="
                    f"value {item['value']:.4f}, "
                    f"gt_med {item['ground_truth_median']:.4f}, "
                    f"pred_med {item['predicted_median']:.4f}, "
                    f"delta {item['predicted_vs_gt_delta']:.4f}"
                )
                for item in row["top_feature_drivers"]
            )
            writer.writerow([
                row["path"],
                row["ground_truth"],
                row["predicted"],
                row["n_misclassified"],
                row["n_tested"],
                row["misclassification_rate_when_tested"],
                row["mean_confidence_when_wrong"],
                top_features,
            ])


def _print_classification_report(report: dict, class_names: list[str]) -> None:
    print("\nClassification report:")
    print(f"{'Class':<16} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for name in class_names:
        stats = report[name]
        print(
            f"{name:<16} {stats['precision']:>10.4f} {stats['recall']:>10.4f} "
            f"{stats['f1-score']:>10.4f} {int(stats['support']):>10d}"
        )
    for avg_name in ("macro avg", "weighted avg"):
        avg = report[avg_name]
        print(
            f"{avg_name:<16} {avg['precision']:>10.4f} {avg['recall']:>10.4f} "
            f"{avg['f1-score']:>10.4f} {int(avg['support']):>10d}"
        )


def _print_feature_importance(rows: list[dict[str, float]], top_k: int) -> None:
    if not rows:
        return
    print(f"\nTop {min(top_k, len(rows))} permutation importances:")
    print(f"{'Feature':<28} {'Mean':>10} {'Std':>10}")
    for row in rows[:top_k]:
        std_val = row.get("importance_std", row.get("importance_std_across_seeds", 0.0))
        print(
            f"{row['feature']:<28} {row['importance_mean']:>10.4f} "
            f"{float(std_val):>10.4f}"
        )


def _compute_feature_importance(
    pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    repeats: int,
) -> list[dict[str, float]]:
    from sklearn.inspection import permutation_importance

    perm = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=repeats,
        random_state=seed,
        scoring="accuracy",
        n_jobs=-1,
    )
    rows = []
    for idx, feature_name in enumerate(FEATURE_NAMES):
        rows.append({
            "feature": feature_name,
            "importance_mean": round(float(perm.importances_mean[idx]), 6),
            "importance_std": round(float(perm.importances_std[idx]), 6),
        })
    rows.sort(key=lambda row: row["importance_mean"], reverse=True)
    return rows


def _compute_class_medians(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    class_medians: dict[str, np.ndarray] = {}
    for class_idx, class_name in enumerate(class_names):
        class_rows = X[y == class_idx]
        class_medians[class_name] = np.median(class_rows, axis=0) if len(class_rows) else np.zeros(X.shape[1])
    feature_scale = np.median(np.abs(X - np.median(X, axis=0)), axis=0)
    feature_scale = np.where(feature_scale > 1e-9, feature_scale, 1.0)
    return class_medians, feature_scale


def _build_recurring_error_diagnostics(
    recurring_errors: list[dict[str, object]],
    X: np.ndarray,
    paths: list[str],
    class_medians: dict[str, np.ndarray],
    feature_scale: np.ndarray,
    top_k: int = 8,
) -> list[dict[str, object]]:
    path_to_idx = {path: i for i, path in enumerate(paths)}
    diagnostics: list[dict[str, object]] = []

    for row in recurring_errors:
        predicted = row.get("most_common_predicted_label")
        ground_truth = row["ground_truth"]
        if predicted is None or predicted not in class_medians or ground_truth not in class_medians:
            continue
        sample_idx = path_to_idx.get(str(row["path"]))
        if sample_idx is None:
            continue

        sample = X[sample_idx]
        gt_med = class_medians[str(ground_truth)]
        pred_med = class_medians[str(predicted)]
        gt_dist = np.abs(sample - gt_med) / feature_scale
        pred_dist = np.abs(sample - pred_med) / feature_scale
        delta = gt_dist - pred_dist

        feature_rows = []
        for idx, feature_name in enumerate(FEATURE_NAMES):
            if delta[idx] <= 0:
                continue
            feature_rows.append({
                "feature": feature_name,
                "value": round(float(sample[idx]), 4),
                "ground_truth_median": round(float(gt_med[idx]), 4),
                "predicted_median": round(float(pred_med[idx]), 4),
                "ground_truth_distance": round(float(gt_dist[idx]), 4),
                "predicted_distance": round(float(pred_dist[idx]), 4),
                "predicted_vs_gt_delta": round(float(delta[idx]), 4),
            })
        feature_rows.sort(key=lambda item: item["predicted_vs_gt_delta"], reverse=True)

        diagnostics.append({
            "path": row["path"],
            "ground_truth": ground_truth,
            "predicted": predicted,
            "n_tested": row["n_tested"],
            "n_misclassified": row["n_misclassified"],
            "misclassification_rate_when_tested": row["misclassification_rate_when_tested"],
            "mean_confidence_when_wrong": row["mean_confidence_when_wrong"],
            "top_feature_drivers": feature_rows[:top_k],
        })

    return diagnostics


def _run_single_seed(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    paths: list[str],
    test_size: float,
    seed: int,
    importance_repeats: int,
) -> dict:
    """Train Stage 1 on one split and return evaluation artifacts."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(paths))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    print(f"\nSplit (test_size={test_size}, seed={seed}):")
    print(f"  train: {len(train_idx)} files")
    print(f"  test:  {len(test_idx)} files")

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    pipeline = build_pipeline()
    print("\nTraining Stage 1 ...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))
    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        digits=4,
        zero_division=0,
    )
    feature_importance = _compute_feature_importance(
        pipeline,
        X_test,
        y_test,
        seed=seed,
        repeats=importance_repeats,
    )

    print(f"\nStage 1 holdout accuracy: {acc:.4f}")
    print("\nConfusion matrix (GT rows, Pred cols):")
    header = "GT \\ Pred".ljust(16) + "".join(name.rjust(16) for name in class_names)
    print(header)
    for i, gt_name in enumerate(class_names):
        row = gt_name.ljust(16) + "".join(f"{int(cm[i, j]):>16d}" for j in range(len(class_names)))
        print(row)
    _print_classification_report(report, class_names)
    _print_feature_importance(feature_importance, top_k=15)

    misclassified_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, str]] = []
    for local_i, dataset_i in enumerate(test_idx):
        gt_idx = int(y_test[local_i])
        pred_idx = int(y_pred[local_i])
        test_rows.append({
            "path": paths[dataset_i],
            "ground_truth": class_names[gt_idx],
        })
        if gt_idx == pred_idx:
            continue
        probs = {
            class_names[j]: round(float(y_prob[local_i, j]), 4)
            for j in range(len(class_names))
        }
        misclassified_rows.append({
            "path": paths[dataset_i],
            "ground_truth": class_names[gt_idx],
            "predicted": class_names[pred_idx],
            "confidence": round(float(np.max(y_prob[local_i])), 4),
            "probabilities": probs,
        })

    misclassified_rows.sort(key=lambda row: float(row["confidence"]), reverse=True)

    return {
        "accuracy": round(acc, 4),
        "test_size": test_size,
        "seed": seed,
        "class_names": class_names,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "confusion_matrix": {
            class_names[i]: {
                class_names[j]: int(cm[i, j])
                for j in range(len(class_names))
            }
            for i in range(len(class_names))
        },
        "classification_report": report,
        "feature_importance": feature_importance,
        "test_rows": test_rows,
        "misclassified": misclassified_rows,
    }


def evaluate_stage1(
    data_dir: Path,
    test_size: float = 0.2,
    seed: int = 42,
    output_dir: Path | None = None,
    importance_repeats: int = 10,
) -> dict:
    """Train Stage 1 on a train split and evaluate on holdout files."""
    output_dir = output_dir or (Path(__file__).parent / "models" / "stage1_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting dataset from {data_dir} ...")
    X, y, class_names, paths = _collect_dataset(data_dir)
    print(f"Dataset: {len(X)} files, classes={class_names}")
    for i, name in enumerate(class_names):
        print(f"  {name}: {int((y == i).sum())} files")

    results = _run_single_seed(
        X=X,
        y=y,
        class_names=class_names,
        paths=paths,
        test_size=test_size,
        seed=seed,
        importance_repeats=importance_repeats,
    )

    json_path = output_dir / "stage1_evaluation.json"
    _save_json(json_path, results)
    print(f"\nSaved JSON report to {json_path}")

    csv_path = output_dir / "stage1_misclassified.csv"
    _write_misclassified_csv(csv_path, results["misclassified"], class_names)
    print(f"Saved misclassified files CSV to {csv_path}")

    feature_csv_path = output_dir / "stage1_feature_importance.csv"
    _write_feature_importance_csv(feature_csv_path, results["feature_importance"])
    print(f"Saved feature importance CSV to {feature_csv_path}")

    return results


def evaluate_stage1_across_seeds(
    data_dir: Path,
    seeds: list[int],
    test_size: float = 0.2,
    output_dir: Path | None = None,
    importance_repeats: int = 10,
) -> dict:
    """Run Stage 1 holdout evaluation across multiple seeds and summarize."""
    output_dir = output_dir or (Path(__file__).parent / "models" / "stage1_eval_multi")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting dataset from {data_dir} ...")
    X, y, class_names, paths = _collect_dataset(data_dir)
    print(f"Dataset: {len(X)} files, classes={class_names}")
    for i, name in enumerate(class_names):
        print(f"  {name}: {int((y == i).sum())} files")

    all_runs: list[dict] = []
    feature_rows_by_name: dict[str, list[tuple[float, float]]] = {name: [] for name in FEATURE_NAMES}
    file_stats: dict[str, dict[str, object]] = {}
    confusions: list[np.ndarray] = []
    class_medians, feature_scale = _compute_class_medians(X, y, class_names)

    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 72}")
        print(f"Seed {seed}")
        print(f"{'=' * 72}")
        run = _run_single_seed(
            X=X,
            y=y,
            class_names=class_names,
            paths=paths,
            test_size=test_size,
            seed=seed,
            importance_repeats=importance_repeats,
        )
        all_runs.append(run)

        _save_json(seed_dir / "stage1_evaluation.json", run)
        _write_misclassified_csv(seed_dir / "stage1_misclassified.csv", run["misclassified"], class_names)
        _write_feature_importance_csv(seed_dir / "stage1_feature_importance.csv", run["feature_importance"])

        confusions.append(np.array([
            [run["confusion_matrix"][gt][pred] for pred in class_names]
            for gt in class_names
        ], dtype=np.int64))

        for row in run["feature_importance"]:
            feature_rows_by_name[row["feature"]].append(
                (float(row["importance_mean"]), float(row["importance_std"]))
            )

        misclassified_by_path = {row["path"]: row for row in run["misclassified"]}
        for test_row in run["test_rows"]:
            stats = file_stats.setdefault(test_row["path"], {
                "path": test_row["path"],
                "ground_truth": test_row["ground_truth"],
                "n_tested": 0,
                "n_misclassified": 0,
                "misclassification_rate_when_tested": 0.0,
                "mean_confidence_when_wrong": 0.0,
                "most_common_predicted_label": None,
                "_wrong_confidences": [],
                "_pred_counter": Counter(),
            })
            stats["n_tested"] += 1
            wrong = misclassified_by_path.get(test_row["path"])
            if wrong is not None:
                stats["n_misclassified"] += 1
                stats["_wrong_confidences"].append(float(wrong["confidence"]))
                stats["_pred_counter"][str(wrong["predicted"])] += 1

    feature_summary = []
    for feature_name in FEATURE_NAMES:
        vals = feature_rows_by_name[feature_name]
        if not vals:
            continue
        means = np.array([v[0] for v in vals], dtype=np.float64)
        stds = np.array([v[1] for v in vals], dtype=np.float64)
        feature_summary.append({
            "feature": feature_name,
            "importance_mean": round(float(np.mean(means)), 6),
            "importance_std_across_seeds": round(float(np.std(means)), 6),
            "within_seed_std_mean": round(float(np.mean(stds)), 6),
        })
    feature_summary.sort(key=lambda row: row["importance_mean"], reverse=True)

    recurring_errors = []
    for row in file_stats.values():
        n_tested = int(row["n_tested"])
        n_wrong = int(row["n_misclassified"])
        if n_wrong == 0:
            continue
        wrong_confidences = row.pop("_wrong_confidences")
        pred_counter: Counter = row.pop("_pred_counter")
        row["misclassification_rate_when_tested"] = round(n_wrong / max(1, n_tested), 4)
        row["mean_confidence_when_wrong"] = round(
            float(np.mean(wrong_confidences)) if wrong_confidences else 0.0,
            4,
        )
        row["most_common_predicted_label"] = pred_counter.most_common(1)[0][0] if pred_counter else None
        recurring_errors.append(row)
    recurring_errors.sort(
        key=lambda row: (
            -int(row["n_misclassified"]),
            -float(row["misclassification_rate_when_tested"]),
            -float(row["mean_confidence_when_wrong"]),
            str(row["path"]),
        )
    )
    recurring_error_diagnostics = _build_recurring_error_diagnostics(
        recurring_errors=recurring_errors,
        X=X,
        paths=paths,
        class_medians=class_medians,
        feature_scale=feature_scale,
    )

    accuracy_values = np.array([float(run["accuracy"]) for run in all_runs], dtype=np.float64)
    confusion_mean = np.mean(np.stack(confusions, axis=0), axis=0) if confusions else np.zeros((0, 0))
    confusion_mean_dict = {
        class_names[i]: {
            class_names[j]: round(float(confusion_mean[i, j]), 2)
            for j in range(len(class_names))
        }
        for i in range(len(class_names))
    }

    summary = {
        "seeds": seeds,
        "test_size": test_size,
        "n_runs": len(all_runs),
        "accuracy_mean": round(float(np.mean(accuracy_values)), 4) if len(accuracy_values) else 0.0,
        "accuracy_std": round(float(np.std(accuracy_values)), 4) if len(accuracy_values) else 0.0,
        "per_seed_accuracy": [{"seed": run["seed"], "accuracy": run["accuracy"]} for run in all_runs],
        "mean_confusion_matrix": confusion_mean_dict,
        "feature_importance_summary": feature_summary,
        "recurring_errors": recurring_errors,
        "recurring_error_diagnostics": recurring_error_diagnostics,
    }

    print(f"\n{'=' * 72}")
    print("Cross-seed summary")
    print(f"{'=' * 72}")
    print(f"Accuracy mean/std: {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")
    _print_feature_importance(feature_summary, top_k=15)
    if recurring_errors:
        print("\nTop recurring errors:")
        print(f"{'Wrong/Tested':<14} {'Rate':>8} {'Conf':>8} {'GT':<14} {'Pred':<14} Path")
        for row in recurring_errors[:15]:
            print(
                f"{row['n_misclassified']:>5d}/{row['n_tested']:<8d} "
                f"{row['misclassification_rate_when_tested']:>8.4f} "
                f"{row['mean_confidence_when_wrong']:>8.4f} "
                f"{str(row['ground_truth']):<14} "
                f"{str(row['most_common_predicted_label'] or '-'): <14} "
                f"{row['path']}"
            )
    if recurring_error_diagnostics:
        print("\nRecurring-error feature drivers:")
        for row in recurring_error_diagnostics[:5]:
            top_names = ", ".join(item["feature"] for item in row["top_feature_drivers"][:5])
            print(
                f"  {row['ground_truth']} -> {row['predicted']}  "
                f"{row['path']}  drivers: {top_names}"
            )

    _save_json(output_dir / "stage1_cross_seed_summary.json", summary)
    _write_feature_importance_csv(output_dir / "stage1_feature_importance_summary.csv", feature_summary)
    with open(output_dir / "stage1_recurring_errors.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "path",
            "ground_truth",
            "n_tested",
            "n_misclassified",
            "misclassification_rate_when_tested",
            "mean_confidence_when_wrong",
            "most_common_predicted_label",
        ])
        for row in recurring_errors:
            writer.writerow([
                row["path"],
                row["ground_truth"],
                row["n_tested"],
                row["n_misclassified"],
                row["misclassification_rate_when_tested"],
                row["mean_confidence_when_wrong"],
                row["most_common_predicted_label"],
            ])
    _save_json(output_dir / "stage1_recurring_error_diagnostics.json", {
        "class_feature_medians": {
            class_name: {
                FEATURE_NAMES[idx]: round(float(vals[idx]), 4)
                for idx in range(len(FEATURE_NAMES))
            }
            for class_name, vals in class_medians.items()
        },
        "diagnostics": recurring_error_diagnostics,
    })
    _write_recurring_error_diagnostics_csv(
        output_dir / "stage1_recurring_error_diagnostics.csv",
        recurring_error_diagnostics,
    )
    print(f"\nSaved cross-seed summary to {output_dir / 'stage1_cross_seed_summary.json'}")
    print(f"Saved recurring errors CSV to {output_dir / 'stage1_recurring_errors.csv'}")
    print(f"Saved feature importance summary CSV to {output_dir / 'stage1_feature_importance_summary.csv'}")
    print(f"Saved recurring-error diagnostics JSON to {output_dir / 'stage1_recurring_error_diagnostics.json'}")
    print(f"Saved recurring-error diagnostics CSV to {output_dir / 'stage1_recurring_error_diagnostics.csv'}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 only with a holdout split")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "benchmark_pyramidal_interneuron_v1_qc_diag_pruned",
        help="Directory with cell-type subfolders containing SWC files",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of files held out for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/test split",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated list of seeds for cross-seed evaluation",
    )
    parser.add_argument(
        "--importance-repeats",
        type=int,
        default=10,
        help="Permutation-importance repeats per seed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "models" / "stage1_eval",
        help="Directory for JSON and CSV evaluation artifacts",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: data directory not found: {args.data_dir}")
        return 1

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if seeds:
        evaluate_stage1_across_seeds(
            args.data_dir,
            seeds=seeds,
            test_size=args.test_size,
            output_dir=args.output_dir,
            importance_repeats=args.importance_repeats,
        )
    else:
        evaluate_stage1(
            args.data_dir,
            test_size=args.test_size,
            seed=args.seed,
            output_dir=args.output_dir,
            importance_repeats=args.importance_repeats,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
