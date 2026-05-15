#!/usr/bin/env python3
"""Train the Stage 1 cell-type classifier.

Usage:
    python -m hybrid.train_stage1 --data-dir data/benchmark_pyramidal_interneuron_v1_qc_diag_pruned

Expected directory structure under --data-dir:
    pyramidal/
        *.swc
    interneuron/
        *.swc

Each subfolder name is the cell-type label. Files in each folder are
labeled morphologies (type labels in the SWC can be anything — only
global shape features are used, not per-node labels).

The script:
1. Extracts global features from every SWC file
2. Trains a two-class ensemble for pyramidal vs interneuron
3. Reports cross-validation accuracy
4. Saves the model to hybrid/models/cell_type_classifier.pkl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid.features import FEATURE_NAMES, extract_feature_vector, parse_swc
from hybrid.cell_type_detector import (
    CELL_TYPES,
    CellTypeClassifier,
    DEFAULT_MODEL_PATH,
)


def _collect_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Walk data_dir for cell-type folders and extract features.

    Returns (X, y_indices, class_names, file_paths).
    """
    class_names: list[str] = []
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    paths: list[str] = []

    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        cell_type = subdir.name.lower().replace("-", "_").replace(" ", "_")
        if cell_type not in CELL_TYPES:
            print(f"warning: skipping unknown cell type folder '{subdir.name}'")
            continue

        if cell_type not in class_names:
            class_names.append(cell_type)
        class_idx = class_names.index(cell_type)

        swc_root = subdir / "swc" if (subdir / "swc").is_dir() else subdir
        swc_files = sorted(swc_root.glob("*.swc"))
        print(f"  {cell_type}: {len(swc_files)} files")
        for swc_path in swc_files:
            try:
                nodes = parse_swc(swc_path)
                if not nodes:
                    continue
                fv = extract_feature_vector(nodes)
                X_list.append(fv)
                y_list.append(class_idx)
                paths.append(str(swc_path))
            except Exception as e:
                print(f"    skip {swc_path.name}: {e}")

    if not X_list:
        raise RuntimeError("No valid SWC files found in data directory")

    X = np.stack(X_list)
    y = np.array(y_list)
    return X, y, class_names, paths


def build_pipeline():
    """Build the Stage 1 sklearn pipeline."""
    from sklearn.ensemble import VotingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    # GPU drop-ins. Same constructor signatures as sklearn originals.
    from hybrid._xgb_classifiers import (
        XGBRandomForestClassifier as RandomForestClassifier,
        XGBExtraTreesClassifier as ExtraTreesClassifier,
        XGBHistGradientBoostingClassifier as HistGradientBoostingClassifier,
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    et = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    hgb = HistGradientBoostingClassifier(
        max_depth=8,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=20,
        random_state=42,
    )
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("et", et), ("hgb", hgb)],
        voting="soft",
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", ensemble),
    ])


def train(
    data_dir: Path,
    output_path: Path = DEFAULT_MODEL_PATH,
    n_cv_folds: int = 5,
) -> dict:
    """Train classifier and return metrics."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    print(f"Collecting dataset from {data_dir} ...")
    X, y, class_names, paths = _collect_dataset(data_dir)
    print(f"Dataset: {len(X)} samples, {len(class_names)} classes: {class_names}")
    for i, cn in enumerate(class_names):
        print(f"  {cn}: {int((y == i).sum())} samples")

    # Ensure all cell types are present (pad class_names if needed)
    full_classes = list(class_names)
    for ct in CELL_TYPES:
        if ct not in full_classes:
            full_classes.append(ct)

    pipeline = build_pipeline()

    # Cross-validation
    n_folds = min(n_cv_folds, min(np.bincount(y)))
    if n_folds >= 2:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
        print(f"Cross-validation accuracy ({n_folds}-fold): {scores.mean():.4f} +/- {scores.std():.4f}")
        cv_results = {"mean": float(scores.mean()), "std": float(scores.std()), "folds": n_folds}
    else:
        print("Too few samples per class for cross-validation, training on all data")
        cv_results = {"mean": 0.0, "std": 0.0, "folds": 0}

    # Train final model on all data
    pipeline.fit(X, y)

    # Save
    classifier = CellTypeClassifier()
    classifier.model = pipeline
    classifier._classes = full_classes
    classifier.feature_names = list(FEATURE_NAMES)
    classifier.save(output_path)
    print(f"Model saved to {output_path}")

    # Save metadata
    meta = {
        "classes": full_classes,
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "feature_names": list(FEATURE_NAMES),
        "class_counts": {cn: int((y == i).sum()) for i, cn in enumerate(class_names)},
        "cv_accuracy": cv_results,
    }
    meta_path = output_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Train cell-type classifier for Stage 1")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "benchmark_pyramidal_interneuron_v1_qc_diag_pruned",
        help="Directory with cell-type subfolders containing SWC files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to save trained model",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: data directory not found: {args.data_dir}")
        print("Create it with cell-type subfolders (pyramidal/, interneuron/)")
        print("Or run: python -m hybrid.download_allen_data")
        return 1

    train(args.data_dir, args.output, args.cv_folds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
