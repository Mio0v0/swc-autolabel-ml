#!/usr/bin/env python3
"""Train Stage 2 branch-level classifier (per cell type, per node).

Design:
- One model per cell type (pyramidal, interneuron). Label
  spaces differ between cell types, so a shared model wastes capacity
  on the wrong sub-problem.
- Per-node training target: each branch emits one row per distinct
  ground-truth label present in it, weighted by node count. Equivalent
  to replicating branch features per node.
- Node-level class balancing via sample_weight = node_count × class
  factor (equal aggregate weight per class). Directly counters axon
  dominance at the node level.

Data leakage prevention:
- Train/test split is at the FILE level (not branch level), so no
  branches from the same cell appear in both train and test sets.
- Branch features are purely geometric/topological — the SWC type
  column is used only as the target label and for soma detection.

Usage:
    python -m hybrid.train_stage2 --data-dir data/benchmark_pyramidal_interneuron_v1_qc_diag_pruned
    python -m hybrid.train_stage2 --data-dir data/benchmark_pyramidal_interneuron_v1_qc_diag_pruned --test-size 0.2
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid.branch_features import (
    BRANCH_FEATURE_NAMES,
    BranchData,
    MorphologyBranches,
    extract_branches,
)
from hybrid.features import parse_swc
from hybrid.subtree_features import PRIMARY_SUBTREE_FEATURE_NAMES, extract_primary_subtrees

MODEL_DIR = Path(__file__).parent / "models"
STAGE2_MODEL_PATH = MODEL_DIR / "branch_classifier.pkl"
STAGE2_META_PATH = MODEL_DIR / "branch_classifier.json"

# Label names
LABEL_NAMES = {1: "soma", 2: "axon", 3: "basal/dendrite", 4: "apical"}

# Valid neurite labels per cell type (soma branches are excluded from training)
VALID_LABELS = {
    "pyramidal": {2, 3, 4},
    "interneuron": {2, 3},
}

OWNER_AUG_FEATURE_NAMES = [
    "owner_prob_axon",
    "owner_prob_basal",
    "owner_prob_apical",
    "owner_is_axon",
    "owner_is_basal",
    "owner_is_apical",
    "owner_confidence",
]

AUGMENTED_BRANCH_FEATURE_NAMES = list(BRANCH_FEATURE_NAMES) + OWNER_AUG_FEATURE_NAMES


def _collect_branches(
    data_dir: Path,
    valid_cell_types: set[str] | None = None,
) -> list[MorphologyBranches]:
    """Extract branches from all SWC files organized by cell type."""
    all_morphologies: list[MorphologyBranches] = []

    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        cell_type = subdir.name.lower()
        if valid_cell_types and cell_type not in valid_cell_types:
            continue

        swc_root = subdir / "swc" if (subdir / "swc").is_dir() else subdir
        swc_files = sorted(swc_root.glob("*.swc"))
        print(f"  {cell_type}: {len(swc_files)} files")
        valid_labels = VALID_LABELS.get(cell_type, {2, 3})

        for swc_path in swc_files:
            try:
                nodes = parse_swc(swc_path)
                if not nodes:
                    continue
                morph = extract_branches(nodes, cell_type, str(swc_path))

                # Filter branches: keep only those with valid neurite labels
                valid_branches = []
                for br in morph.branches:
                    if br.gt_label in valid_labels:
                        valid_branches.append(br)
                morph.branches = valid_branches

                if morph.branches:
                    all_morphologies.append(morph)
            except Exception as e:
                print(f"    skip {swc_path.name}: {e}")

    return all_morphologies


def _file_level_split(
    morphologies: list[MorphologyBranches],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[MorphologyBranches], list[MorphologyBranches]]:
    """Split morphologies into train/test at file level, stratified by cell type.

    This prevents data leakage: no branches from the same cell
    appear in both train and test sets.
    """
    rng = np.random.RandomState(seed)

    by_type: dict[str, list[MorphologyBranches]] = defaultdict(list)
    for m in morphologies:
        by_type[m.cell_type].append(m)

    train: list[MorphologyBranches] = []
    test: list[MorphologyBranches] = []

    for cell_type, morphs in sorted(by_type.items()):
        indices = np.arange(len(morphs))
        rng.shuffle(indices)
        n_test = max(1, int(len(morphs) * test_size))
        test_idx = set(indices[:n_test].tolist())
        for i, m in enumerate(morphs):
            if i in test_idx:
                test.append(m)
            else:
                train.append(m)

        n_train = len(morphs) - n_test
        print(f"  {cell_type}: {n_train} train, {n_test} test files")

    return train, test


def _morphologies_to_per_node_arrays(
    morphologies: list[MorphologyBranches],
    valid_neurite_labels: set[int],
    owner_maps_by_file: dict[str, dict[int, dict[str, float | int]]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Flatten morphologies into per-NODE weighted arrays.

    For each branch, emit one row per distinct label present in the
    branch's ground-truth node counts, weighted by the number of nodes
    with that label. This is mathematically equivalent to replicating
    the branch feature vector once per node (with the per-node SWC type
    as the target), but stays ~2-3× the branch count instead of 100×+.

    Soma nodes (type==1) are excluded — soma is detected by rule, not
    trained.

    Returns (X, y, node_count_weights, cell_types).
    """
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    w_list: list[float] = []
    ct_list: list[str] = []

    for morph in morphologies:
        owner_map = owner_maps_by_file.get(morph.file_path, {}) if owner_maps_by_file else {}
        for br in morph.branches:
            features = _branch_feature_with_owner(br, owner_map)
            for lbl, count in br.gt_label_counts.items():
                if lbl in valid_neurite_labels and count > 0:
                    X_list.append(features)
                    y_list.append(lbl)
                    w_list.append(float(count))
                    ct_list.append(morph.cell_type)

    if not X_list:
        return (
            np.empty((0, len(AUGMENTED_BRANCH_FEATURE_NAMES))),
            np.array([]),
            np.array([]),
            [],
        )
    X = np.stack(X_list)
    y = np.array(y_list)
    w = np.array(w_list)
    return X, y, w, ct_list


def _branch_feature_with_owner(
    br: BranchData,
    owner_map: dict[int, dict[str, float | int]],
) -> np.ndarray:
    aug = np.zeros(len(OWNER_AUG_FEATURE_NAMES), dtype=np.float64)
    pr = br.primary_root_idx
    if pr is not None and pr in owner_map:
        info = owner_map[pr]
        prob_axon = float(info.get("prob_2", 0.0))
        prob_basal = float(info.get("prob_3", 0.0))
        prob_apical = float(info.get("prob_4", 0.0))
        pred = int(info.get("pred", 3))
        conf = float(info.get("conf", 0.0))
        aug[:] = [
            prob_axon,
            prob_basal,
            prob_apical,
            1.0 if pred == 2 else 0.0,
            1.0 if pred == 3 else 0.0,
            1.0 if pred == 4 else 0.0,
            conf,
        ]
    return np.concatenate([br.features, aug])


def _build_subtree_pipeline(seed: int):
    from sklearn.ensemble import VotingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    # GPU drop-ins. Same constructor signatures as sklearn originals.
    from hybrid._xgb_classifiers import (
        XGBRandomForestClassifier as RandomForestClassifier,
        XGBGradientBoostingClassifier as GradientBoostingClassifier,
    )

    rf = RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        min_samples_leaf=3,
        random_state=seed,
    )
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        voting="soft",
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", ensemble),
    ])


def _train_subtree_owner_for_cell_type(
    train_morphs: list[MorphologyBranches],
    cell_type: str,
    valid_labels: set[int],
    seed: int,
) -> object | None:
    """Train a subtree-owner classifier for a specific cell type.

    One model per cell type. Target = majority node-weighted label of each
    primary subtree restricted to ``valid_labels``. Used both as an
    augmentation feature for Stage 2 and (for interneurons) as a
    subtree-level broadcast signal in Stage 3.
    """
    ct_morphs = [m for m in train_morphs if m.cell_type == cell_type]
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    w_list: list[float] = []
    for morph in ct_morphs:
        try:
            nodes = parse_swc(morph.file_path)
        except Exception:
            continue
        for sub in extract_primary_subtrees(nodes, cell_type):
            if sub.gt_label not in valid_labels:
                continue
            X_list.append(sub.features)
            y_list.append(sub.gt_label)
            w_list.append(float(sum(sub.gt_label_counts.values())))

    if not X_list or len(set(y_list)) < 2:
        return None

    X = np.stack(X_list)
    y = np.array(y_list)
    w = np.array(w_list)
    sw = _node_balanced_weights(y, w)
    model = _build_subtree_pipeline(seed)
    model.fit(X, y, clf__sample_weight=sw)
    return model


def _train_pyramidal_subtree_owner(
    train_morphs: list[MorphologyBranches],
    seed: int,
) -> object | None:
    """Back-compat shim: pyramidal-only subtree owner model."""
    return _train_subtree_owner_for_cell_type(
        train_morphs, "pyramidal", {2, 3, 4}, seed
    )


def _predict_subtree_owner_map(
    file_path: str,
    cell_type: str,
    subtree_model: object | None,
) -> dict[int, dict[str, float | int]]:
    """Predict per-subtree owner probabilities.

    Works for any cell type that has a trained subtree model. ``cell_type``
    is consulted only to drive feature extraction.
    """
    if subtree_model is None:
        return {}

    try:
        nodes = parse_swc(file_path)
    except Exception:
        return {}

    subtrees = extract_primary_subtrees(nodes, cell_type)
    if not subtrees:
        return {}

    X = np.stack([s.features for s in subtrees])
    probs = subtree_model.predict_proba(X)
    classes = subtree_model.classes_
    out: dict[int, dict[str, float | int]] = {}
    for sub, row in zip(subtrees, probs):
        info: dict[str, float | int] = {}
        best_label = 3
        best_prob = -1.0
        for cls, prob in zip(classes, row):
            cls_i = int(cls)
            info[f"prob_{cls_i}"] = float(prob)
            if float(prob) > best_prob:
                best_prob = float(prob)
                best_label = cls_i
        info["pred"] = best_label
        info["conf"] = best_prob
        out[sub.root_idx] = info
    return out


def _predict_subtree_owner_map_by_cell_type(
    file_path: str,
    cell_type: str,
    subtree_models_by_cell_type: dict[str, object] | None,
) -> dict[int, dict[str, float | int]]:
    """Dispatch by cell type to the appropriate subtree-owner model."""
    if not subtree_models_by_cell_type:
        return {}
    model = subtree_models_by_cell_type.get(cell_type)
    if model is None:
        return {}
    return _predict_subtree_owner_map(file_path, cell_type, model)


def _node_balanced_weights(
    y: np.ndarray, w: np.ndarray, power: float | None = None,
) -> np.ndarray:
    """Combine node-count weights with node-level class balancing.

    effective_w = node_count × class_balance_factor

    With ``power=1.0`` (default, backward compatible) the class factor
    equalizes total per-class weight: each class contributes the same
    aggregate mass to the loss.

    With ``power > 1`` the inverse-frequency weighting is amplified,
    over-correcting toward minority classes. Useful when the model
    still over-predicts the majority class even after equal-mass
    balancing — common with axon-vs-dendrite at Stage 2.

    The ``power`` may also be set via the ``SWCAL_CLASS_BALANCE_POWER``
    env var (default 1.0). The function argument takes precedence.
    """
    if power is None:
        import os as _os
        power = float(_os.environ.get("SWCAL_CLASS_BALANCE_POWER", "1.0"))
    class_totals: dict[int, float] = {}
    for lbl, weight in zip(y.tolist(), w.tolist()):
        class_totals[lbl] = class_totals.get(lbl, 0.0) + weight
    n_classes = len(class_totals)
    total = sum(class_totals.values())
    if n_classes == 0 or total <= 0:
        return w
    class_factor = {
        c: (total / (n_classes * t)) ** power
        for c, t in class_totals.items()
    }
    return np.array([weight * class_factor[lbl] for lbl, weight in zip(y.tolist(), w.tolist())])


def _evaluate(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cell_types: list[str],
    label_names: dict[int, str],
) -> dict:
    """Evaluate model and return detailed metrics."""
    y_pred = model.predict(X_test)

    # Overall accuracy
    overall_acc = float(np.mean(y_pred == y_test))

    # Per cell-type accuracy
    per_type: dict[str, dict] = {}
    for ct in sorted(set(cell_types)):
        mask = np.array([c == ct for c in cell_types])
        if not mask.any():
            continue
        yt = y_test[mask]
        yp = y_pred[mask]
        acc = float(np.mean(yt == yp))

        # Per-label within this cell type
        per_label: dict[str, dict] = {}
        labels_present = sorted(set(yt.tolist()) | set(yp.tolist()))
        for label in labels_present:
            tp = int(((yt == label) & (yp == label)).sum())
            fp = int(((yt != label) & (yp == label)).sum())
            fn = int(((yt == label) & (yp != label)).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            support = int((yt == label).sum())
            per_label[label_names.get(label, str(label))] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
                "tp": tp, "fp": fp, "fn": fn,
            }

        # Confusion matrix for this cell type
        confusion: dict[str, dict[str, int]] = {}
        for gt_l in labels_present:
            gt_name = label_names.get(gt_l, str(gt_l))
            confusion[gt_name] = {}
            for pred_l in labels_present:
                pred_name = label_names.get(pred_l, str(pred_l))
                confusion[gt_name][pred_name] = int(((yt == gt_l) & (yp == pred_l)).sum())

        per_type[ct] = {
            "accuracy": round(acc, 4),
            "n_branches": int(mask.sum()),
            "per_label": per_label,
            "confusion": confusion,
        }

    return {
        "overall_accuracy": round(overall_acc, 4),
        "n_branches": len(y_test),
        "per_cell_type": per_type,
    }


def _print_results(results: dict) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'='*70}")
    print(f"Overall branch accuracy: {results['overall_accuracy']:.4f}  "
          f"({results['n_branches']} branches)")
    print(f"{'='*70}")

    for ct, ct_data in results["per_cell_type"].items():
        print(f"\n--- {ct} (accuracy={ct_data['accuracy']:.4f}, "
              f"branches={ct_data['n_branches']}) ---")

        print(f"  {'Label':<16} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
        print(f"  {'-'*48}")
        for label_name, metrics in ct_data["per_label"].items():
            print(f"  {label_name:<16} {metrics['precision']:>8.4f} "
                  f"{metrics['recall']:>8.4f} {metrics['f1']:>8.4f} "
                  f"{metrics['support']:>8d}")

        # Confusion matrix
        print(f"\n  Confusion matrix:")
        labels = list(ct_data["confusion"].keys())
        header = "GT \\ Pred"
        print(f"  {header:<16}", end="")
        for l in labels:
            print(f" {l:>12}", end="")
        print()
        for gt_l in labels:
            print(f"  {gt_l:<16}", end="")
            for pred_l in labels:
                print(f" {ct_data['confusion'][gt_l].get(pred_l, 0):>12d}", end="")
            print()


def _build_pipeline(seed: int):
    """Construct a fresh RF+GB soft-voting pipeline.

    class_weight is NOT set here because we pass node-count × class-balance
    weights via sample_weight at fit time (see _node_balanced_weights).
    Setting class_weight would double-count the balancing.
    """
    from sklearn.ensemble import VotingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    # GPU drop-ins. Same constructor signatures as sklearn originals.
    from hybrid._xgb_classifiers import (
        XGBRandomForestClassifier as RandomForestClassifier,
        XGBGradientBoostingClassifier as GradientBoostingClassifier,
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        min_samples_leaf=5,
        random_state=seed,
    )
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        voting="soft",
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", ensemble),
    ])


def _fit_calibrated_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    sw: np.ndarray,
    seed: int,
    cal_fraction: float = 0.15,
) -> object:
    """Fit a Stage 2 pipeline with isotonic probability calibration.

    sklearn 1.8 in this environment rejects the legacy ``cv='prefit'``
    path. Use direct cross-validated calibration around the full pipeline
    instead, while still passing branch sample weights through to the
    underlying classifier.

    Fallbacks:
      * If the data is too small / too imbalanced for calibrated CV,
        return the uncalibrated pipeline.
    """
    from sklearn.calibration import CalibratedClassifierCV

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or len(X) < 20:
        base = _build_pipeline(seed)
        base.fit(X, y, clf__sample_weight=sw)
        return base

    min_class_count = int(np.min(counts))
    if min_class_count < 2:
        base = _build_pipeline(seed)
        base.fit(X, y, clf__sample_weight=sw)
        return base

    cv = min(3, min_class_count)
    if cv < 2:
        base = _build_pipeline(seed)
        base.fit(X, y, clf__sample_weight=sw)
        return base

    base = _build_pipeline(seed)
    calibrated = CalibratedClassifierCV(
        base,
        method="isotonic",
        cv=cv,
        ensemble=False,
    )
    calibrated.fit(X, y, clf__sample_weight=sw)
    return calibrated


def train(
    data_dir: Path,
    test_size: float = 0.2,
    output_path: Path = STAGE2_MODEL_PATH,
    seed: int = 42,
) -> dict:
    """Train Stage 2: one model per cell type, per-node weighted, with
    node-level class balancing.

    Key changes vs the old single-shared-model design:
      1. One model per cell type (pyramidal, interneuron).
         Label spaces differ, so a shared model wastes capacity on the
         wrong subproblem.
      2. Per-node training: each training row is (branch_features, label)
         weighted by the number of nodes in that branch with that label.
         This makes the loss per-node-accurate, not per-branch-majority.
      3. Node-level class balancing: sample_weight is multiplied by a
         class factor so each class contributes equal total weight. In
         pyramidal this strongly down-weights axons (~91% of nodes) so
         basal/apical discrimination drives the learning.
    """
    print(f"Collecting branches from {data_dir} ...")
    morphologies = _collect_branches(data_dir, {"pyramidal", "interneuron"})
    print(f"Total: {len(morphologies)} morphologies, "
          f"{sum(len(m.branches) for m in morphologies)} branches")

    # File-level split (stratified by cell type)
    print(f"\nSplitting (test_size={test_size}, seed={seed})...")
    train_morphs, test_morphs = _file_level_split(morphologies, test_size, seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    models_by_cell_type: dict[str, object] = {}
    default_labels_by_cell_type: dict[str, int] = {}
    per_cell_type_results: dict[str, dict] = {}

    # --- Subtree-owner models (item 4: whole-subtree classifier) ---
    # One subtree model per cell type. Pyramidal covers {axon, basal, apical};
    # interneuron covers {axon, basal}. The subtree model's predictions are used two ways:
    #   (a) as 7 augmentation features on each branch feature vector;
    #   (b) broadcast at Stage 3 inference time to seed whole-subtree labels
    #       before branch-level overrides (see pipeline.py).
    subtree_owner_models_by_cell_type: dict[str, object] = {}
    subtree_targets = {
        "pyramidal": {2, 3, 4},
        "interneuron": {2, 3},
    }
    for ct, valid_sub_labels in subtree_targets.items():
        m = _train_subtree_owner_for_cell_type(
            train_morphs, ct, valid_sub_labels, seed
        )
        if m is not None:
            subtree_owner_models_by_cell_type[ct] = m
    # Back-compat alias (older pipeline.py versions read this key).
    pyramidal_subtree_owner_model = subtree_owner_models_by_cell_type.get("pyramidal")

    owner_maps_train: dict[str, dict[int, dict[str, float | int]]] = {}
    owner_maps_test: dict[str, dict[int, dict[str, float | int]]] = {}
    for m in train_morphs:
        if m.cell_type in subtree_owner_models_by_cell_type:
            owner_maps_train[m.file_path] = _predict_subtree_owner_map(
                m.file_path,
                m.cell_type,
                subtree_owner_models_by_cell_type[m.cell_type],
            )
    for m in test_morphs:
        if m.cell_type in subtree_owner_models_by_cell_type:
            owner_maps_test[m.file_path] = _predict_subtree_owner_map(
                m.file_path,
                m.cell_type,
                subtree_owner_models_by_cell_type[m.cell_type],
            )

    for ct in ["pyramidal", "interneuron"]:
        print(f"\n{'='*70}\nCell type: {ct.upper()}\n{'='*70}")
        ct_train = [m for m in train_morphs if m.cell_type == ct]
        ct_test = [m for m in test_morphs if m.cell_type == ct]
        if not ct_train:
            print(f"  no training data for {ct}, skipping")
            continue

        valid_neurite = VALID_LABELS[ct] - {1}  # labels this model predicts
        use_owner = ct in subtree_owner_models_by_cell_type
        X_tr, y_tr, w_tr, ct_tr = _morphologies_to_per_node_arrays(
            ct_train,
            valid_neurite,
            owner_maps_train if use_owner else None,
        )
        X_te, y_te, w_te, ct_te = _morphologies_to_per_node_arrays(
            ct_test,
            valid_neurite,
            owner_maps_test if use_owner else None,
        )

        # Node-count summary
        node_counts_tr = {lbl: int(sum(wi for yi, wi in zip(y_tr, w_tr) if yi == lbl))
                           for lbl in sorted(set(y_tr.tolist()))}
        print(f"  {len(ct_train)} train files / {len(ct_test)} test files")
        print(f"  Train rows: {len(X_tr)} (weighted nodes: {int(w_tr.sum())})")
        print(f"  Train node-count by class:")
        for lbl, cnt in sorted(node_counts_tr.items()):
            print(f"    {LABEL_NAMES.get(lbl, str(lbl))}: {cnt}")

        uniq = sorted(set(y_tr.tolist()))
        if len(uniq) < 2:
            # Only one class in training data — model is not learnable.
            # Record as default label so the pipeline can still label nodes.
            default_labels_by_cell_type[ct] = int(uniq[0]) if uniq else 3
            print(f"  Only 1 class in training data ({uniq}); "
                  f"registering default label {default_labels_by_cell_type[ct]}.")
            per_cell_type_results[ct] = {
                "note": "only one class in training data; default-label rule applied",
                "default_label": default_labels_by_cell_type[ct],
                "n_train_files": len(ct_train),
                "n_test_files": len(ct_test),
            }
            continue

        # Compute node-balanced sample weights (combines node_count × class factor)
        sw_tr = _node_balanced_weights(y_tr, w_tr)

        print(f"  Training {ct} model on {len(X_tr)} rows (isotonic-calibrated) ...")
        t0 = time.time()
        pipeline = _fit_calibrated_pipeline(X_tr, y_tr, sw_tr, seed)
        print(f"  done in {time.time() - t0:.1f}s")

        models_by_cell_type[ct] = pipeline

        # Evaluate weighted by node count — this matches per-node evaluation
        # since weight-per-row equals the number of nodes that row represents.
        if len(X_te) > 0:
            print(f"\n  --- {ct} TRAIN (node-weighted) ---")
            tr_res = _evaluate(pipeline, X_tr, y_tr, ct_tr, LABEL_NAMES)
            _print_results(tr_res)
            print(f"\n  --- {ct} TEST (node-weighted, no leakage) ---")
            te_res = _evaluate(pipeline, X_te, y_te, ct_te, LABEL_NAMES)
            _print_results(te_res)
            per_cell_type_results[ct] = {
                "train_results": tr_res,
                "test_results": te_res,
                "n_train_files": len(ct_train),
                "n_test_files": len(ct_test),
                "n_train_rows": int(len(X_tr)),
                "n_test_rows": int(len(X_te)),
                "node_counts_train": node_counts_tr,
            }

    # Save all models together
    with open(output_path, "wb") as f:
        pickle.dump({
            "models_by_cell_type": models_by_cell_type,
            "default_labels_by_cell_type": default_labels_by_cell_type,
            "subtree_owner_models_by_cell_type": subtree_owner_models_by_cell_type,
            # Back-compat alias used by older pipeline.py.
            "pyramidal_subtree_owner_model": pyramidal_subtree_owner_model,
            "feature_names": list(AUGMENTED_BRANCH_FEATURE_NAMES),
            "branch_feature_names": list(AUGMENTED_BRANCH_FEATURE_NAMES),
            "primary_subtree_feature_names": list(PRIMARY_SUBTREE_FEATURE_NAMES),
            "owner_feature_names": list(OWNER_AUG_FEATURE_NAMES),
            "label_names": LABEL_NAMES,
            "valid_labels": {k: sorted(v) for k, v in VALID_LABELS.items()},
            "calibration": "isotonic_prefit_0.15",
        }, f)
    print(f"\nModels saved to {output_path}")

    meta = {
        "per_cell_type": per_cell_type_results,
        "models_trained": sorted(models_by_cell_type.keys()),
        "default_labels_by_cell_type": default_labels_by_cell_type,
        "subtree_owner_models_trained": sorted(subtree_owner_models_by_cell_type.keys()),
        "calibration": "isotonic_prefit_0.15",
        "n_features": len(AUGMENTED_BRANCH_FEATURE_NAMES),
        "feature_names": list(AUGMENTED_BRANCH_FEATURE_NAMES),
        "test_size": test_size,
        "seed": seed,
    }
    meta_path = output_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Metadata saved to {meta_path}")

    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Stage 2 branch classifier")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "benchmark_pyramidal_interneuron_v1_qc_diag_pruned",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=STAGE2_MODEL_PATH)
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: data directory not found: {args.data_dir}")
        return 1

    train(args.data_dir, args.test_size, args.output, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
