"""End-to-end hybrid auto-labeling pipeline.

Chains Stage 1 (cell-type detection) → Stage 2 (branch classification)
→ Stage 3 (topology refinement) into a single function call.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import SWCNode, parse_swc
from .cell_type_detector import (
    CellTypeResult,
    detect_cell_type_from_nodes,
    DEFAULT_MODEL_PATH as STAGE1_MODEL,
)
from .branch_features import (
    extract_branches,
)
from .subtree_features import extract_primary_subtrees
from .stage3_refine import RefinementResult, refine

STAGE2_MODEL = Path(__file__).parent / "models" / "branch_classifier.pkl"


@dataclass
class PipelineResult:
    """Full pipeline output."""
    stage1: CellTypeResult
    stage3: RefinementResult
    node_labels: list[int]          # final per-node labels
    node_confidences: list[float]   # per-node ML confidence


def _load_stage2_bundle(model_path: Path) -> dict:
    """Load the Stage 2 bundle, supporting both old (single model) and
    new (per-cell-type models) formats for backward compatibility."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data


def _select_stage2_model(
    bundle: dict,
    cell_type: str,
) -> tuple[object | None, int | None]:
    """Pick the right Stage 2 model (or default label) for a cell type.

    Returns (model, default_label). Exactly one of them is non-None:
      - (model, None): use model.predict_proba on branch features
      - (None, label): no model trained for this cell type; assign the
                       single default label to all non-soma branches
    """
    models = bundle.get("models_by_cell_type")
    defaults = bundle.get("default_labels_by_cell_type", {})

    if models is not None:
        if cell_type in models:
            return models[cell_type], None
        if cell_type in defaults:
            return None, int(defaults[cell_type])
        # Fallback: use any trained model (e.g. "other" cell type)
        if models:
            return next(iter(models.values())), None
        return None, 3  # final fallback: mark as generic dendrite
    # Old-format single model
    return bundle.get("model"), None


def run_pipeline(
    swc_path: str | Path,
    stage1_model: str | Path | None = None,
    stage2_model: str | Path | None = None,
) -> PipelineResult:
    """Run the full 3-stage hybrid pipeline on an SWC file.

    Args:
        swc_path: path to SWC file
        stage1_model: path to Stage 1 model (optional, uses default)
        stage2_model: path to Stage 2 model (optional, uses default)

    Returns:
        PipelineResult with per-node labels and metadata.
    """
    nodes = parse_swc(swc_path)
    return run_pipeline_on_nodes(nodes, str(swc_path), stage1_model, stage2_model)


def run_pipeline_on_nodes(
    nodes: list[SWCNode],
    file_path: str = "",
    stage1_model: str | Path | None = None,
    stage2_model: str | Path | None = None,
) -> PipelineResult:
    """Run the full pipeline on pre-parsed nodes."""
    n = len(nodes)

    # --- Stage 1: Cell-type detection ---
    s1_result = detect_cell_type_from_nodes(nodes, stage1_model)
    cell_type = s1_result.cell_type
    label_set = s1_result.label_set

    # --- Stage 2: Branch classification ---
    s2_path = Path(stage2_model) if stage2_model else STAGE2_MODEL
    bundle = _load_stage2_bundle(s2_path)
    model, default_label = _select_stage2_model(bundle, cell_type)
    # Pick subtree-owner model: new bundles have a per-cell-type dict,
    # older bundles have a single pyramidal-only model.
    subtree_models_by_ct = bundle.get("subtree_owner_models_by_cell_type")
    if subtree_models_by_ct:
        subtree_owner_model = subtree_models_by_ct.get(cell_type)
    else:
        legacy = bundle.get("pyramidal_subtree_owner_model")
        subtree_owner_model = legacy if cell_type == "pyramidal" else None

    morph = extract_branches(nodes, cell_type, file_path)
    subtree_owner_map = _predict_subtree_owner_map(nodes, cell_type, subtree_owner_model)
    # Apical owner only exists for pyramidal cells (no apical in interneurons).
    apical_owner_root = _best_apical_owner(subtree_owner_map) if cell_type == "pyramidal" else None

    # Initialize per-node labels and confidences from the label-free proxy root.
    proxy_soma = set(morph.soma_indices)
    node_labels = [1 if i in proxy_soma else 0 for i in range(n)]
    node_confidences = [1.0 if i in proxy_soma else 0.0 for i in range(n)]

    neurite_labels = sorted(label_set - {1})

    if model is not None:
        # Classify branches via the per-cell-type model
        for br in morph.branches:
            X = _branch_feature_with_owner(br, subtree_owner_map).reshape(1, -1)
            probs = model.predict_proba(X)[0]
            classes = model.classes_

            # Find probabilities for valid labels only
            valid_probs: dict[int, float] = {}
            for cls, prob in zip(classes, probs):
                if int(cls) in neurite_labels:
                    valid_probs[int(cls)] = float(prob)

            if valid_probs:
                total = sum(valid_probs.values())
                if total > 0:
                    valid_probs = {k: v / total for k, v in valid_probs.items()}
                best_label = max(valid_probs, key=lambda k: valid_probs[k])
                best_conf = valid_probs[best_label]
            else:
                best_label = neurite_labels[0] if neurite_labels else 3
                best_conf = 0.5

            for node_idx in br.node_indices:
                node_labels[node_idx] = best_label
                node_confidences[node_idx] = best_conf
    else:
        # No model for this cell type (e.g. Purkinje with only dendrite
        # in training data). Assign the registered default label with
        # conservative confidence so Stage 3 can still override it.
        fallback = default_label if default_label is not None else (
            neurite_labels[0] if neurite_labels else 3
        )
        for br in morph.branches:
            for node_idx in br.node_indices:
                node_labels[node_idx] = fallback
                node_confidences[node_idx] = 0.5

    # Fill any unassigned non-soma nodes
    for i in range(n):
        if node_labels[i] == 0:
            node_labels[i] = neurite_labels[0] if neurite_labels else 3
            node_confidences[i] = 0.3

    # --- Stage 3: Topology refinement ---
    s3_result = refine(
        nodes,
        node_labels,
        node_confidences,
        s1_result,
        apical_owner_root=apical_owner_root,
    )
    final_labels = [rl.label for rl in s3_result.labels]

    return PipelineResult(
        stage1=s1_result,
        stage3=s3_result,
        node_labels=final_labels,
        node_confidences=node_confidences,
    )


def _predict_subtree_owner_map(
    nodes: list[SWCNode],
    cell_type: str,
    subtree_owner_model: object | None,
) -> dict[int, dict[str, float | int]]:
    # Generalised (item 4): any cell type with a trained subtree-owner
    # model gets augmented features. Pyramidal → {axon, basal, apical},
    # interneuron → {axon, basal}, purkinje → usually none (single-class).
    if subtree_owner_model is None:
        return {}

    subtrees = extract_primary_subtrees(nodes, cell_type)
    if not subtrees:
        return {}

    X = np.stack([sub.features for sub in subtrees])
    probs = subtree_owner_model.predict_proba(X)
    classes = subtree_owner_model.classes_
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


def _branch_feature_with_owner(br, subtree_owner_map: dict[int, dict[str, float | int]]) -> np.ndarray:
    aug = np.zeros(7, dtype=np.float64)
    pr = getattr(br, "primary_root_idx", None)
    if pr is not None and pr in subtree_owner_map:
        info = subtree_owner_map[pr]
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


def _best_apical_owner(subtree_owner_map: dict[int, dict[str, float | int]]) -> int | None:
    best_root: int | None = None
    best_prob = 0.0
    for root_idx, info in subtree_owner_map.items():
        prob = float(info.get("prob_4", 0.0))
        if prob > best_prob:
            best_prob = prob
            best_root = root_idx
    if best_root is None or best_prob < 0.45:
        return None
    return best_root
