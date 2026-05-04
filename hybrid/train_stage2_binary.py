"""Train a binary axon-vs-dendrite classifier (Stage 2 B1).

Part of the proposed two-binary-stage architecture (your suggestion):

    Tier A: per-subtree axon/basal/apical classifier (existing)
        ↓ feeds 7 augmented features into B1
    Tier B1 (this script): binary axon-vs-dendrite per branch
        ↓ dendrite branches forwarded to B2
    Tier B2: basal-vs-apical (the existing GNN — paper/gnn_apical_basal.py)
        ↓
    Stage 3: topology refinement (unchanged)

The original Tier B was 3-class (axon / basal / apical) and had to balance
two decisions in one model. Splitting into B1+B2 lets each be tuned
independently. In particular, B1 can use an *asymmetric* cost — penalize
"dendrite called axon" more heavily than "axon called dendrite" — which
directly attacks the 18% raw apical→axon failure mode that caps Stage 2's
neurite F1.

This script:
    - Reuses the same training data as `hybrid.train_stage2` (file split
      determined by hybrid/models/eval_split.json — never touches test files).
    - Reuses Tier A's subtree-owner predictions for the 7 augmented features.
    - Maps SWC labels: axon (2) -> 1, basal/apical (3/4) -> 0.
    - Applies an asymmetric multiplier (`--dendrite-weight-mult`, default 2.0)
      on top of the existing node-count + class-balance weighting.
    - Trains one model per cell type.
    - Saves to hybrid/models/branch_classifier_axon_dendrite.pkl with the
      same bundle structure used by the production pipeline (so pipeline.py
      can load it with the existing `_load_stage2_bundle` helper).

Usage:
    python -m hybrid.train_stage2_binary
    python -m hybrid.train_stage2_binary --dendrite-weight-mult 3.0
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from hybrid.branch_features import MorphologyBranches, extract_branches
from hybrid.features import parse_swc
from hybrid.train_stage2 import (
    AUGMENTED_BRANCH_FEATURE_NAMES,
    LABEL_NAMES,
    VALID_LABELS,
    _build_pipeline,
    _morphologies_to_per_node_arrays,
    _node_balanced_weights,
    _predict_subtree_owner_map,
    _train_subtree_owner_for_cell_type,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data" / "benchmark_pyramidal_interneuron_v1_qc_diag_pruned"
DEFAULT_MODEL_PATH = ROOT / "hybrid" / "models" / "branch_classifier_axon_dendrite.pkl"
EVAL_SPLIT_PATH = ROOT / "hybrid" / "models" / "eval_split.json"

# Binary-class encoding the model emits.
DENDRITE_CLASS = 0  # SWC labels {3, 4} collapse to this
AXON_CLASS = 1      # SWC label 2


def _collect_morphs(data_dir: Path, eval_split_path: Path) -> tuple[list[MorphologyBranches], list[str]]:
    """Build MorphologyBranches for every TRAIN file in eval_split.json.

    Returns (train_morphs, skipped_reasons). Excludes the 328 held-out test
    files so the binary classifier never sees them — the same protocol the
    production train_stage2 uses inside `evaluate.py`.
    """
    import json
    with eval_split_path.open() as f:
        split = json.load(f)
    test_basenames = {
        Path(p).name
        for ct_files in split.get("test_files", {}).values()
        for p in ct_files
    }
    train_morphs: list[MorphologyBranches] = []
    for ct in ("pyramidal", "interneuron"):
        ct_dir = data_dir / ct
        if not ct_dir.is_dir():
            continue
        for swc in sorted(ct_dir.rglob("*.swc")):
            if swc.name in test_basenames:
                continue
            try:
                nodes = parse_swc(swc)
            except Exception:
                continue
            if not nodes:
                continue
            mb = extract_branches(nodes, ct, str(swc))
            if mb.branches:
                train_morphs.append(mb)
    return train_morphs, []


def _binarize(y: np.ndarray) -> np.ndarray:
    """Map SWC labels to {DENDRITE_CLASS, AXON_CLASS}."""
    out = np.where(y == 2, AXON_CLASS, DENDRITE_CLASS)
    return out.astype(np.int64)


def _evaluate_binary(model, X, y_bin, w, label: str) -> None:
    """Print a small diagnostic for this binary head."""
    if len(X) == 0:
        print(f"  [{label}] empty")
        return
    pred = model.predict(X)
    n = len(y_bin)
    tp = int(np.sum((pred == AXON_CLASS) & (y_bin == AXON_CLASS)))
    fp = int(np.sum((pred == AXON_CLASS) & (y_bin == DENDRITE_CLASS)))
    tn = int(np.sum((pred == DENDRITE_CLASS) & (y_bin == DENDRITE_CLASS)))
    fn = int(np.sum((pred == DENDRITE_CLASS) & (y_bin == AXON_CLASS)))
    # Branch-row-level recalls
    axon_recall = tp / max(tp + fn, 1)
    dend_recall = tn / max(tn + fp, 1)
    # Node-weighted (using w, the per-row node count)
    w_tp = float(np.sum(w[(pred == AXON_CLASS) & (y_bin == AXON_CLASS)]))
    w_fp = float(np.sum(w[(pred == AXON_CLASS) & (y_bin == DENDRITE_CLASS)]))
    w_tn = float(np.sum(w[(pred == DENDRITE_CLASS) & (y_bin == DENDRITE_CLASS)]))
    w_fn = float(np.sum(w[(pred == DENDRITE_CLASS) & (y_bin == AXON_CLASS)]))
    print(f"  [{label}] n_rows={n}  branches:")
    print(f"     axon recall  = {axon_recall:.4f}  dendrite recall = {dend_recall:.4f}")
    print(f"     node-weighted:")
    print(f"       axon recall     = {w_tp / max(w_tp + w_fn, 1):.4f}")
    print(f"       dendrite recall = {w_tn / max(w_tn + w_fp, 1):.4f}  "
          f"(this is the metric we want UP — it's 1 - apical→axon - basal→axon)")


def train(
    data_dir: Path,
    output_path: Path,
    eval_split_path: Path,
    dendrite_weight_mult: float = 2.0,
    seed: int = 42,
) -> dict:
    print(f"Collecting morphologies from {data_dir} ...")
    print(f"  excluding test files listed in {eval_split_path}")
    t0 = time.time()
    train_morphs, _ = _collect_morphs(data_dir, eval_split_path)
    print(f"  {len(train_morphs)} train morphologies loaded ({time.time()-t0:.1f}s)")

    # --- Tier A (subtree-owner) — same as train_stage2.py ---
    subtree_targets = {"pyramidal": {2, 3, 4}, "interneuron": {2, 3}}
    subtree_owner_models_by_cell_type: dict[str, object] = {}
    for ct, valid_sub_labels in subtree_targets.items():
        print(f"\nTraining Tier A subtree-owner for {ct} ...")
        m = _train_subtree_owner_for_cell_type(
            train_morphs, ct, valid_sub_labels, seed,
        )
        if m is not None:
            subtree_owner_models_by_cell_type[ct] = m

    # Build owner maps for every train morph (B1 features depend on Tier A).
    owner_maps_train: dict[str, dict[int, dict[str, float | int]]] = {}
    for m in train_morphs:
        if m.cell_type in subtree_owner_models_by_cell_type:
            owner_maps_train[m.file_path] = _predict_subtree_owner_map(
                m.file_path, m.cell_type,
                subtree_owner_models_by_cell_type[m.cell_type],
            )

    # --- Tier B1 (binary axon-vs-dendrite) per cell type ---
    models_by_cell_type: dict[str, object] = {}
    per_cell_type_results: dict[str, dict] = {}

    for ct in ("pyramidal", "interneuron"):
        print(f"\n{'='*70}\nCell type: {ct.upper()}  (binary B1)\n{'='*70}")
        ct_train = [m for m in train_morphs if m.cell_type == ct]
        if not ct_train:
            print(f"  no training data for {ct}, skipping")
            continue

        valid_neurite = VALID_LABELS[ct] - {1}
        use_owner = ct in subtree_owner_models_by_cell_type
        X_tr, y_tr, w_tr, _ = _morphologies_to_per_node_arrays(
            ct_train,
            valid_neurite,
            owner_maps_train if use_owner else None,
        )
        if len(X_tr) == 0:
            print(f"  no training rows for {ct}; skipping")
            continue

        # Binarize labels: axon (2) -> 1, basal/apical (3, 4) -> 0
        y_bin = _binarize(y_tr)
        n_axon = int(np.sum(y_bin == AXON_CLASS))
        n_dend = int(np.sum(y_bin == DENDRITE_CLASS))
        print(f"  Train rows: {len(X_tr)}  (weighted nodes: {int(w_tr.sum())})")
        print(f"    by binary class:  axon={n_axon}  dendrite={n_dend}")
        print(f"    by raw SWC class: " + ", ".join(
            f"{LABEL_NAMES.get(lbl, str(lbl))}={int(np.sum(y_tr == lbl))}"
            for lbl in sorted(set(y_tr.tolist()))
        ))

        if len(set(y_bin.tolist())) < 2:
            print(f"  only one binary class — cannot train")
            continue

        # Sample weights:
        #   1. Node-count + class-balance (existing helper).
        #   2. Asymmetric multiplier on dendrite rows (push the boundary
        #      to favor dendrite recall, attacking the apical-as-axon failure).
        sw = _node_balanced_weights(y_bin, w_tr)
        if dendrite_weight_mult != 1.0:
            sw = sw.copy()
            sw[y_bin == DENDRITE_CLASS] *= dendrite_weight_mult
            print(f"  Applied asymmetric weight: dendrite rows × {dendrite_weight_mult:.2f}")

        print(f"  Training B1 (binary)...")
        t0 = time.time()
        pipe = _build_pipeline(seed)
        pipe.fit(X_tr, y_bin, clf__sample_weight=sw)
        print(f"    done in {time.time() - t0:.1f}s")

        _evaluate_binary(pipe, X_tr, y_bin, w_tr, label=f"{ct} TRAIN")

        models_by_cell_type[ct] = pipe
        per_cell_type_results[ct] = {
            "n_train_rows": int(len(X_tr)),
            "n_axon_rows": int(n_axon),
            "n_dendrite_rows": int(n_dend),
            "weighted_nodes_total": int(w_tr.sum()),
            "dendrite_weight_mult": float(dendrite_weight_mult),
        }

    # --- Save bundle (mirrors train_stage2.py format so pipeline can load it) ---
    bundle = {
        "kind": "axon_dendrite_binary",
        "models_by_cell_type": models_by_cell_type,
        "subtree_owner_models_by_cell_type": subtree_owner_models_by_cell_type,
        "feature_names": list(AUGMENTED_BRANCH_FEATURE_NAMES),
        "branch_feature_names": list(AUGMENTED_BRANCH_FEATURE_NAMES),
        "label_names": {DENDRITE_CLASS: "dendrite", AXON_CLASS: "axon"},
        "label_swc_mapping": {DENDRITE_CLASS: 3, AXON_CLASS: 2},
        "calibration": "none_yet",
        "dendrite_weight_mult": float(dendrite_weight_mult),
        "seed": seed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved B1 bundle to {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)")
    return {
        "models": list(models_by_cell_type.keys()),
        "per_cell_type": per_cell_type_results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--eval-split", type=Path, default=EVAL_SPLIT_PATH)
    ap.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument(
        "--dendrite-weight-mult", type=float, default=2.0,
        help="Multiplier on dendrite rows in sample_weight (on top of node-balanced "
             "weighting). 1.0 = symmetric, 2.0 = punish dendrite-as-axon 2x more, etc.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(
        args.data_dir, args.out, args.eval_split,
        dendrite_weight_mult=args.dendrite_weight_mult,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
