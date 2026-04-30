#!/usr/bin/env python3
"""Evaluate the full hybrid pipeline (Stage 1+2+3) with no data leakage.

Performs file-level train/test split, trains fresh models on the train
split only, then evaluates the full pipeline on the held-out test files.

Reports:
- Stage 2 only (ML) accuracy vs Stage 2+3 (ML + refinement) accuracy
- Per cell-type and per-label metrics
- Confusion matrices
- Improvement from Stage 3 refinement

Usage:
    python -m hybrid.evaluate --data-dir data/benchmark_pyramidal_interneuron_v1_qc_diag_pruned
    python -m hybrid.evaluate --data-dir data/benchmark_pyramidal_interneuron_v1_qc_diag_pruned --test-size 0.25
"""
from __future__ import annotations

import argparse
import hashlib
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

from hybrid.features import FEATURE_NAMES, SWCNode, extract_feature_vector, parse_swc
from hybrid.cell_type_detector import (
    CELL_TYPES,
    CELL_TYPE_LABEL_SETS,
    CellTypeClassifier,
    detect_cell_type_from_nodes,
)
from hybrid.branch_features import BRANCH_FEATURE_NAMES, extract_branches
from hybrid.stage3_refine import refine
from hybrid.pipeline import run_pipeline_on_nodes
from hybrid.train_stage2 import (
    AUGMENTED_BRANCH_FEATURE_NAMES,
    _branch_feature_with_owner,
    _build_pipeline as _build_stage2_pipeline,
    _fit_calibrated_pipeline,
    _node_balanced_weights,
    _predict_subtree_owner_map,
    _train_subtree_owner_for_cell_type,
)

LABEL_NAMES = {1: "soma", 2: "axon", 3: "basal/dendrite", 4: "apical"}
VALID_LABELS = {
    "pyramidal": {1, 2, 3, 4},
    "interneuron": {1, 2, 3},
}
NEURITE_LABELS = {
    "pyramidal": {2, 3, 4},
    "interneuron": {2, 3},
}


def _file_in_test_bucket(file_name: str, seed: int, test_size: float) -> bool:
    """Stable file-identity split: hash(seed + cell_type + file_name) → bucket.

    Crucial property: the same file with the same seed ALWAYS lands in the
    same split, regardless of how many other files are in the dataset. This
    makes A/B comparisons across dataset versions robust — removing a file
    doesn't reshuffle the rest into different splits.

    The seed only changes which files land in test; flipping the seed gives
    a different but equally reproducible split.
    """
    h = hashlib.md5(f"{seed}:{file_name}".encode()).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF  # uniform [0, 1)
    return bucket < test_size


def _file_level_split(
    files_by_type: dict[str, list[Path]],
    test_size: float,
    seed: int,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """Split files into train/test per cell type using a stable hash bucket.

    See _file_in_test_bucket for the hash-bucket rationale. Per-class minimum
    of 1 test file is guaranteed (falls back to the lexicographically first
    file if the hash bucket happens to leave a class empty).
    """
    train: dict[str, list[Path]] = {}
    test: dict[str, list[Path]] = {}

    for ct, files in sorted(files_by_type.items()):
        ct_train, ct_test = [], []
        for f in files:
            if _file_in_test_bucket(f.name, seed, test_size):
                ct_test.append(f)
            else:
                ct_train.append(f)
        # Guarantee at least one test file per class
        if not ct_test and files:
            ct_test = [files[0]]
            ct_train = files[1:]
        train[ct] = ct_train
        test[ct] = ct_test

    return train, test


def _train_stage1(train_files: dict[str, list[Path]], model_path: Path) -> None:
    """Train Stage 1 on train split only."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X_list, y_list = [], []
    class_names = sorted(train_files.keys())
    for ci, ct in enumerate(class_names):
        for f in train_files[ct]:
            try:
                nodes = parse_swc(f)
                if not nodes:
                    continue
                X_list.append(extract_feature_vector(nodes))
                y_list.append(ci)
            except Exception:
                pass

    X = np.stack(X_list)
    y = np.array(y_list)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                               class_weight="balanced", random_state=42, n_jobs=-1)),
                ("gb", GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                                   learning_rate=0.1, min_samples_leaf=3, random_state=42)),
            ],
            voting="soft",
        )),
    ])
    pipeline.fit(X, y)

    full_classes = list(class_names)
    for ct in CELL_TYPES:
        if ct not in full_classes:
            full_classes.append(ct)

    classifier = CellTypeClassifier()
    classifier.model = pipeline
    classifier._classes = full_classes
    classifier.feature_names = list(FEATURE_NAMES)
    classifier.save(model_path)


def _train_stage2(train_files: dict[str, list[Path]], model_path: Path) -> None:
    """Train Stage 2 per-cell-type on train split only.

    Mirrors the production design in train_stage2.py:
      - One model per cell type (pyramidal, interneuron)
      - Per-node training: one row per (branch, label) weighted by node count
      - Node-level class balancing via sample_weight (equal per-class mass)
      - Purkinje falls back to a default label if only one class in data
    """
    models_by_cell_type: dict[str, object] = {}
    default_labels_by_cell_type: dict[str, int] = {}

    # Collect per-cell-type train morphologies once — reused below for subtree
    # owner training and for branch feature extraction.
    train_morphs_by_ct: dict[str, list] = {}
    for ct in ("pyramidal", "interneuron"):
        morphs = []
        for f in train_files.get(ct, []):
            try:
                nodes = parse_swc(f)
                if nodes:
                    morphs.append(extract_branches(nodes, ct, str(f)))
            except Exception:
                pass
        train_morphs_by_ct[ct] = morphs

    # Train a subtree-owner model per cell type (item 4). Pyramidal gets
    # {2,3,4}; interneuron {2,3}.
    subtree_owner_models_by_cell_type: dict[str, object] = {}
    subtree_targets = {
        "pyramidal": {2, 3, 4},
        "interneuron": {2, 3},
    }
    for ct, valid_sub_labels in subtree_targets.items():
        all_train_morphs = (
            train_morphs_by_ct.get("pyramidal", [])
            + train_morphs_by_ct.get("interneuron", [])
        )
        m = _train_subtree_owner_for_cell_type(
            all_train_morphs, ct, valid_sub_labels, seed=42
        )
        if m is not None:
            subtree_owner_models_by_cell_type[ct] = m

    # Owner probability maps per file, keyed by filepath for reuse in branch
    # feature augmentation. Computed once for both pyramidal and interneuron
    # (whichever have a subtree owner model trained).
    owner_maps_by_file: dict[str, dict[int, dict[str, float | int]]] = {}
    for ct, owner_model in subtree_owner_models_by_cell_type.items():
        for f in train_files.get(ct, []):
            owner_maps_by_file[str(f)] = _predict_subtree_owner_map(
                str(f), ct, owner_model
            )

    for ct in ["pyramidal", "interneuron"]:
        if ct not in train_files or not train_files[ct]:
            continue
        valid_neurite = NEURITE_LABELS.get(ct, {2, 3})
        use_owner = ct in subtree_owner_models_by_cell_type

        # Build (features, label, node_count_weight) rows
        X_list, y_list, w_list = [], [], []
        for morph in train_morphs_by_ct.get(ct, []):
            owner_map = owner_maps_by_file.get(morph.file_path, {}) if use_owner else {}
            for br in morph.branches:
                feat = _branch_feature_with_owner(br, owner_map)
                for lbl, count in br.gt_label_counts.items():
                    if lbl in valid_neurite and count > 0:
                        X_list.append(feat)
                        y_list.append(lbl)
                        w_list.append(float(count))

        if not X_list:
            continue
        y_arr = np.array(y_list)
        w_arr = np.array(w_list)
        X_arr = np.stack(X_list)

        if len(set(y_arr.tolist())) < 2:
            # Only one class → register as default label
            default_labels_by_cell_type[ct] = int(y_arr[0])
            continue

        sw = _node_balanced_weights(y_arr, w_arr)

        # Isotonic-calibrated fit (item 3). Falls back to uncalibrated if
        # the split is infeasible (too few samples / classes).
        pipeline = _fit_calibrated_pipeline(X_arr, y_arr, sw, seed=42)
        models_by_cell_type[ct] = pipeline

    # Back-compat alias for old pipeline.py readers.
    pyramidal_subtree_owner_model = subtree_owner_models_by_cell_type.get("pyramidal")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({
            "models_by_cell_type": models_by_cell_type,
            "default_labels_by_cell_type": default_labels_by_cell_type,
            "subtree_owner_models_by_cell_type": subtree_owner_models_by_cell_type,
            "pyramidal_subtree_owner_model": pyramidal_subtree_owner_model,
            "feature_names": list(AUGMENTED_BRANCH_FEATURE_NAMES),
            "label_names": LABEL_NAMES,
            "calibration": "isotonic_prefit_0.15",
        }, f)


def _evaluate_file(
    nodes: list[SWCNode],
    cell_type_gt: str,
    stage1_model: Path,
    stage2_model: Path,
) -> tuple[
    list[int],      # gt labels
    list[int],      # stage2-only labels
    list[int],      # stage2+3 labels
    str,            # predicted cell type
]:
    """Run the pipeline on one file and return ground truth + predictions."""
    # Ground truth
    gt_labels = [nd.type for nd in nodes]

    # Run full pipeline
    result = run_pipeline_on_nodes(nodes, "", stage1_model, stage2_model)

    # Also get Stage 2-only labels (before refinement)
    # Re-extract to get the pre-refinement labels
    morph = extract_branches(nodes, result.stage1.cell_type, "")
    label_set = result.stage1.label_set
    neurite_labels = sorted(label_set - {1})

    s2_bundle = pickle.load(open(stage2_model, "rb"))
    # Per-cell-type dispatch (supports both new dict-of-models and old
    # single-model bundles for backward compat during transition).
    models = s2_bundle.get("models_by_cell_type")
    defaults = s2_bundle.get("default_labels_by_cell_type", {})
    if models is not None:
        ct_pred = result.stage1.cell_type
        model = models.get(ct_pred)
        default_label = defaults.get(ct_pred)
        if model is None and models:
            model = next(iter(models.values()))
    else:
        model = s2_bundle.get("model")
        default_label = None
    # Subtree-owner model dispatch: new bundles have a per-cell-type dict,
    # older bundles have a single pyramidal-only model (back-compat).
    subtree_models_by_ct = s2_bundle.get("subtree_owner_models_by_cell_type")
    ct_pred_s1 = result.stage1.cell_type
    if subtree_models_by_ct:
        subtree_owner_model = subtree_models_by_ct.get(ct_pred_s1)
    else:
        legacy = s2_bundle.get("pyramidal_subtree_owner_model")
        subtree_owner_model = legacy if ct_pred_s1 == "pyramidal" else None

    subtree_owner_map: dict = {}
    if subtree_owner_model is not None:
        from hybrid.subtree_features import extract_primary_subtrees
        subtrees = extract_primary_subtrees(nodes, ct_pred_s1)
        if subtrees:
            X_sub = np.stack([sub.features for sub in subtrees])
            probs = subtree_owner_model.predict_proba(X_sub)
            classes = subtree_owner_model.classes_
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
                subtree_owner_map[sub.root_idx] = info

    s2_labels = [1 if i in set(morph.soma_indices) else 0 for i in range(len(nodes))]
    if model is not None:
        for br in morph.branches:
            X = _branch_feature_with_owner(br, subtree_owner_map).reshape(1, -1)
            probs = model.predict_proba(X)[0]
            classes = model.classes_
            valid_probs = {}
            for cls, prob in zip(classes, probs):
                if int(cls) in neurite_labels:
                    valid_probs[int(cls)] = float(prob)
            if valid_probs:
                total = sum(valid_probs.values())
                if total > 0:
                    valid_probs = {k: v / total for k, v in valid_probs.items()}
                best = max(valid_probs, key=lambda k: valid_probs[k])
            else:
                best = neurite_labels[0] if neurite_labels else 3
            for idx in br.node_indices:
                s2_labels[idx] = best
    else:
        fb = default_label if default_label is not None else (
            neurite_labels[0] if neurite_labels else 3
        )
        for br in morph.branches:
            for idx in br.node_indices:
                s2_labels[idx] = fb

    for i in range(len(nodes)):
        if s2_labels[i] == 0:
            s2_labels[i] = neurite_labels[0] if neurite_labels else 3

    s23_labels = result.node_labels
    return gt_labels, s2_labels, s23_labels, result.stage1.cell_type


def _compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    valid_labels: set[int],
) -> dict:
    """Compute per-node accuracy, macro-F1, balanced accuracy, and per-class stats.

    Each node counts once (pure per-node classification). Macro-averaged
    metrics give each class equal weight regardless of support, so the
    axon class (which dominates node counts in pyramidal cells) does not
    dominate the headline score. Soma-excluded macro metrics further
    strip out the trivially-correct soma class so the macro score
    reflects the harder neurite-discrimination task.
    """
    # Filter to valid labels only (by ground truth)
    pairs = [(gt, pred) for gt, pred in zip(y_true, y_pred) if gt in valid_labels]
    if not pairs:
        return {
            "accuracy": 0.0, "n": 0,
            "macro_f1": 0.0, "macro_precision": 0.0, "macro_recall": 0.0,
            "balanced_accuracy": 0.0,
            "neurite_macro_f1": 0.0, "neurite_balanced_accuracy": 0.0,
            "per_label": {}, "confusion": {},
        }

    n = len(pairs)
    correct = sum(1 for g, p in pairs if g == p)
    accuracy = correct / n

    all_labels = sorted(valid_labels)
    per_label: dict[str, dict] = {}
    # Track per-label f1/precision/recall in label order for macro aggregation
    label_f1: dict[int, float] = {}
    label_prec: dict[int, float] = {}
    label_rec: dict[int, float] = {}
    label_support: dict[int, int] = {}
    for lbl in all_labels:
        tp = sum(1 for g, p in pairs if g == lbl and p == lbl)
        fp = sum(1 for g, p in pairs if g != lbl and p == lbl)
        fn = sum(1 for g, p in pairs if g == lbl and p != lbl)
        support = sum(1 for g, _ in pairs if g == lbl)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        label_f1[lbl] = f1
        label_prec[lbl] = prec
        label_rec[lbl] = rec
        label_support[lbl] = support
        per_label[LABEL_NAMES.get(lbl, str(lbl))] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    # Macro averages over classes PRESENT in the ground truth (support > 0).
    # Absent classes would force 0.0 into the mean and misrepresent the task.
    present = [lbl for lbl in all_labels if label_support[lbl] > 0]
    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    macro_f1 = _mean([label_f1[l] for l in present])
    macro_prec = _mean([label_prec[l] for l in present])
    macro_rec = _mean([label_rec[l] for l in present])
    balanced_acc = macro_rec  # balanced accuracy == mean per-class recall

    # Neurite-only (soma excluded) — strips the trivial type==1 class
    # that every method gets right by a hard rule.
    neurite_present = [l for l in present if l != 1]
    neurite_macro_f1 = _mean([label_f1[l] for l in neurite_present])
    neurite_balanced_acc = _mean([label_rec[l] for l in neurite_present])

    confusion: dict[str, dict[str, int]] = {}
    for gt_l in all_labels:
        gt_name = LABEL_NAMES.get(gt_l, str(gt_l))
        confusion[gt_name] = {}
        for pred_l in all_labels:
            pred_name = LABEL_NAMES.get(pred_l, str(pred_l))
            confusion[gt_name][pred_name] = sum(1 for g, p in pairs if g == gt_l and p == pred_l)

    return {
        "accuracy": round(accuracy, 4),
        "n": n,
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "neurite_macro_f1": round(neurite_macro_f1, 4),
        "neurite_balanced_accuracy": round(neurite_balanced_acc, 4),
        "per_label": per_label,
        "confusion": confusion,
    }


def _print_metrics(title: str, metrics: dict) -> None:
    # HEADLINE (node-level, axon-bias-free): neurite-macro-F1.
    # Accuracy is node-weighted so axon dominates — kept for reference,
    # but NOT the primary number.
    print(f"\n  {title}")
    print(f"    HEADLINE  neurite-macro-F1  = {metrics.get('neurite_macro_f1', 0.0):.4f}  "
          f"← per-class equal weight, soma excluded")
    print(f"              neurite-bal-acc   = {metrics.get('neurite_balanced_accuracy', 0.0):.4f}")
    print(f"    (ref)     macro-F1          = {metrics.get('macro_f1', 0.0):.4f}  "
          f"(includes trivially-100%-correct soma)")
    print(f"    (ref)     balanced-accuracy = {metrics.get('balanced_accuracy', 0.0):.4f}")
    print(f"    (ref)     accuracy          = {metrics['accuracy']:.4f}  "
          f"({metrics['n']} nodes — axon-inflated, do not cite as headline)")
    if metrics["per_label"]:
        print(f"    {'Label':<16} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
        print(f"    {'-'*48}")
        for name, m in metrics["per_label"].items():
            print(f"    {name:<16} {m['precision']:>8.4f} {m['recall']:>8.4f} "
                  f"{m['f1']:>8.4f} {m['support']:>8d}")
    if metrics.get("confusion"):
        labels = list(metrics["confusion"].keys())
        print(f"    {'GT \\ Pred':<16}", end="")
        for l in labels:
            print(f" {l:>14}", end="")
        print()
        for gt_l in labels:
            print(f"    {gt_l:<16}", end="")
            for pred_l in labels:
                print(f" {metrics['confusion'][gt_l].get(pred_l, 0):>14d}", end="")
            print()


def _summarize_file_scores(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n_files": 0,
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "min": 0.0,
            "worst_files": [],
        }

    scores = np.array([float(r["neurite_macro_f1"]) for r in rows], dtype=np.float64)
    worst_rows = sorted(rows, key=lambda r: (float(r["neurite_macro_f1"]), -int(r["n_nodes"])))[:5]
    return {
        "n_files": len(rows),
        "mean": round(float(np.mean(scores)), 4),
        "median": round(float(np.median(scores)), 4),
        "p10": round(float(np.percentile(scores, 10)), 4),
        "min": round(float(np.min(scores)), 4),
        "worst_files": worst_rows,
    }


def _print_file_score_summary(title: str, summary: dict) -> None:
    print(f"\n  {title}")
    print(f"    files={summary['n_files']}  mean={summary['mean']:.4f}  median={summary['median']:.4f}  p10={summary['p10']:.4f}  min={summary['min']:.4f}")
    if summary["worst_files"]:
        print("    5 worst files:")
        for row in summary["worst_files"]:
            print(f"      {row['path']}  score={row['neurite_macro_f1']:.4f}  nodes={row['n_nodes']}")


def evaluate(
    data_dir: Path,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Full evaluation with fresh train/test split and no leakage."""
    # Collect files
    files_by_type: dict[str, list[Path]] = {}
    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        ct = subdir.name.lower()
        if ct not in VALID_LABELS:
            continue
        swc_root = subdir / "swc" if (subdir / "swc").is_dir() else subdir
        files = sorted(swc_root.glob("*.swc"))
        if files:
            files_by_type[ct] = files

    print(f"Dataset: {sum(len(f) for f in files_by_type.values())} files")
    for ct, files in sorted(files_by_type.items()):
        print(f"  {ct}: {len(files)} files")

    # Split (stable hash-based — same file → same split regardless of dataset size)
    train_files, test_files = _file_level_split(files_by_type, test_size, seed)
    print(f"\nSplit (test_size={test_size}, seed={seed}, hash-bucket):")
    for ct in sorted(train_files.keys()):
        print(f"  {ct}: {len(train_files[ct])} train, {len(test_files[ct])} test")

    # Persist the test split so A/B runs can confirm overlap.
    split_path = Path(__file__).parent / "models" / "eval_split.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump({
            "seed": seed,
            "test_size": test_size,
            "data_dir": str(data_dir),
            "test_files": {ct: sorted(p.name for p in files) for ct, files in test_files.items()},
        }, f, indent=2)
    print(f"  test split written to {split_path}")

    # Train models on train split only
    eval_dir = Path(__file__).parent / "models" / "eval_tmp"
    eval_dir.mkdir(parents=True, exist_ok=True)
    s1_model = eval_dir / "s1_eval.pkl"
    s2_model = eval_dir / "s2_eval.pkl"

    print("\nTraining Stage 1 on train split...")
    t0 = time.time()
    _train_stage1(train_files, s1_model)
    print(f"  done in {time.time() - t0:.1f}s")

    print("Training Stage 2 on train split...")
    t0 = time.time()
    _train_stage2(train_files, s2_model)
    print(f"  done in {time.time() - t0:.1f}s")

    # Evaluate on test files
    print("\nEvaluating on test files...")
    all_gt: dict[str, list[int]] = defaultdict(list)
    all_s2: dict[str, list[int]] = defaultdict(list)
    all_s23: dict[str, list[int]] = defaultdict(list)
    file_scores_s2: dict[str, list[dict]] = defaultdict(list)
    file_scores_s23: dict[str, list[dict]] = defaultdict(list)
    cell_type_correct = 0
    cell_type_total = 0
    cell_type_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for ct, files in sorted(test_files.items()):
        for fi, f in enumerate(files):
            try:
                nodes = parse_swc(f)
                if not nodes:
                    continue
                gt, s2, s23, pred_ct = _evaluate_file(nodes, ct, s1_model, s2_model)
                all_gt[ct].extend(gt)
                all_s2[ct].extend(s2)
                all_s23[ct].extend(s23)

                file_valid = [lbl for lbl in gt if lbl != 1]
                if len(file_valid) >= 50:
                    file_metrics_s2 = _compute_metrics(gt, s2, VALID_LABELS.get(ct, {1, 2, 3}))
                    file_metrics_s23 = _compute_metrics(gt, s23, VALID_LABELS.get(ct, {1, 2, 3}))
                    row_base = {
                        "path": str(f),
                        "cell_type": ct,
                        "n_nodes": len(file_valid),
                    }
                    file_scores_s2[ct].append({**row_base, "neurite_macro_f1": float(file_metrics_s2.get("neurite_macro_f1", 0.0))})
                    file_scores_s23[ct].append({**row_base, "neurite_macro_f1": float(file_metrics_s23.get("neurite_macro_f1", 0.0))})

                cell_type_total += 1
                if pred_ct == ct:
                    cell_type_correct += 1
                cell_type_confusion[ct][pred_ct] += 1

            except Exception as e:
                print(f"  skip {f.name}: {e}")

            if (fi + 1) % 50 == 0:
                print(f"  {ct}: {fi + 1}/{len(files)} files processed")

    # Print results
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS (file-level holdout, no data leakage)")
    print(f"{'='*70}")

    # Stage 1 accuracy
    print(f"\n--- Stage 1: Cell-Type Detection ---")
    print(f"  Accuracy: {cell_type_correct}/{cell_type_total} "
          f"({cell_type_correct/max(1,cell_type_total):.4f})")
    for ct in sorted(cell_type_confusion.keys()):
        preds = cell_type_confusion[ct]
        print(f"  {ct}: {dict(preds)}")

    # Per cell-type Stage 2 vs Stage 2+3
    results: dict[str, dict] = {}
    for ct in sorted(all_gt.keys()):
        valid = VALID_LABELS.get(ct, {1, 2, 3})
        gt = all_gt[ct]
        s2 = all_s2[ct]
        s23 = all_s23[ct]

        s2_metrics = _compute_metrics(gt, s2, valid)
        s23_metrics = _compute_metrics(gt, s23, valid)

        print(f"\n{'='*70}")
        print(f"Cell type: {ct.upper()}")
        print(f"{'='*70}")
        _print_metrics("Stage 2 only (ML)", s2_metrics)
        _print_metrics("Stage 2+3 (ML + Refinement)", s23_metrics)

        acc_delta = s23_metrics["accuracy"] - s2_metrics["accuracy"]
        macro_delta = s23_metrics["macro_f1"] - s2_metrics["macro_f1"]
        neurite_delta = s23_metrics["neurite_macro_f1"] - s2_metrics["neurite_macro_f1"]
        print(f"\n  Stage-3 delta: accuracy {acc_delta:+.4f}  "
              f"macro-F1 {macro_delta:+.4f}  neurite-macro-F1 {neurite_delta:+.4f}")

        file_summary_s2 = _summarize_file_scores(file_scores_s2.get(ct, []))
        file_summary_s23 = _summarize_file_scores(file_scores_s23.get(ct, []))
        _print_file_score_summary("Per-file neurite-macro-F1 (Stage 2)", file_summary_s2)
        _print_file_score_summary("Per-file neurite-macro-F1 (Stage 2+3)", file_summary_s23)

        results[ct] = {
            "stage2": s2_metrics,
            "stage23": s23_metrics,
            "per_file_stage2": file_summary_s2,
            "per_file_stage23": file_summary_s23,
            "improvement_accuracy": round(acc_delta, 4),
            "improvement_macro_f1": round(macro_delta, 4),
            "improvement_neurite_macro_f1": round(neurite_delta, 4),
        }

    # Overall
    all_gt_flat = []
    all_s2_flat = []
    all_s23_flat = []
    for ct in all_gt:
        all_gt_flat.extend(all_gt[ct])
        all_s2_flat.extend(all_s2[ct])
        all_s23_flat.extend(all_s23[ct])

    overall_s2 = _compute_metrics(all_gt_flat, all_s2_flat, {1, 2, 3, 4})
    overall_s23 = _compute_metrics(all_gt_flat, all_s23_flat, {1, 2, 3, 4})
    overall_file_summary_s2 = _summarize_file_scores([row for rows in file_scores_s2.values() for row in rows])
    overall_file_summary_s23 = _summarize_file_scores([row for rows in file_scores_s23.values() for row in rows])

    print(f"\n{'='*70}")
    print("OVERALL (all cell types combined)")
    print(f"{'='*70}")
    _print_metrics("Stage 2 only", overall_s2)
    _print_metrics("Stage 2+3 (with refinement)", overall_s23)
    _print_file_score_summary("Overall per-file neurite-macro-F1 (Stage 2)", overall_file_summary_s2)
    _print_file_score_summary("Overall per-file neurite-macro-F1 (Stage 2+3)", overall_file_summary_s23)
    acc_delta = overall_s23["accuracy"] - overall_s2["accuracy"]
    macro_delta = overall_s23["macro_f1"] - overall_s2["macro_f1"]
    neurite_delta = overall_s23["neurite_macro_f1"] - overall_s2["neurite_macro_f1"]
    print(f"\n  Overall Stage-3 delta (HEADLINE = neurite-macro-F1): "
          f"{neurite_delta:+.4f}")
    print(f"  (ref)  macro-F1 Δ {macro_delta:+.4f}   accuracy Δ {acc_delta:+.4f}")

    print(f"\n{'#'*70}")
    print(f"# FINAL HEADLINE NUMBERS (pure per-node, axon-bias-free)")
    print(f"#   Stage 2 only        neurite-macro-F1 = {overall_s2['neurite_macro_f1']:.4f}")
    print(f"#   Stage 2+3 refined   neurite-macro-F1 = {overall_s23['neurite_macro_f1']:.4f}")
    print(f"#   Cell-type accuracy  = {cell_type_correct/max(1,cell_type_total):.4f}")
    print(f"{'#'*70}")

    # Save results
    output = {
        "cell_type_accuracy": round(cell_type_correct / max(1, cell_type_total), 4),
        "per_cell_type": results,
        "overall_stage2": overall_s2,
        "overall_stage23": overall_s23,
        "overall_per_file_stage2": overall_file_summary_s2,
        "overall_per_file_stage23": overall_file_summary_s23,
        "overall_improvement_accuracy": round(acc_delta, 4),
        "overall_improvement_macro_f1": round(macro_delta, 4),
        "overall_improvement_neurite_macro_f1": round(neurite_delta, 4),
    }
    output_path = Path(__file__).parent / "models" / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Full per-file CSV — one row per test file, both Stage 2 and Stage 2+3
    # scores, sorted worst-first. Use this to find the long tail of bad files.
    csv_path = Path(__file__).parent / "models" / "per_file_scores.csv"
    s23_by_path = {row["path"]: row for rows in file_scores_s23.values() for row in rows}
    all_rows = []
    for ct, rows in file_scores_s2.items():
        for r in rows:
            s23 = s23_by_path.get(r["path"], {})
            all_rows.append((
                r["path"], ct, r["n_nodes"],
                float(r["neurite_macro_f1"]),
                float(s23.get("neurite_macro_f1", 0.0)),
            ))
    all_rows.sort(key=lambda x: (x[4], x[3]))  # worst Stage 2+3 first
    with open(csv_path, "w") as f:
        f.write("path,cell_type,n_nodes,neurite_macro_f1_stage2,neurite_macro_f1_stage23\n")
        for row in all_rows:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]:.4f},{row[4]:.4f}\n")
    print(f"Per-file CSV saved to {csv_path}")

    # Keep eval models around so follow-up diagnostic scripts can reuse them
    # without retraining. Delete hybrid/models/eval_tmp/ manually when done.
    print(f"Eval models kept at {eval_dir} — delete manually when no longer needed.")

    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid pipeline with file-level train/test split"
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "benchmark_pyramidal_interneuron_v1_qc_diag_pruned")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: {args.data_dir} not found")
        return 1

    evaluate(args.data_dir, args.test_size, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
