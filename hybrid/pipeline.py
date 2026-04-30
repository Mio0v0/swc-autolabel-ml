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
    CELL_TYPES,
    CELL_TYPE_LABEL_SETS,
    CellTypeResult,
    detect_cell_type_from_nodes,
    DEFAULT_MODEL_PATH as STAGE1_MODEL,
)

# Soft handoff: when Stage 1's confidence falls below this threshold the
# pipeline runs Stage 2+3 for BOTH cell types and picks whichever produces
# higher mean per-node confidence. This recovers borderline files (e.g.
# slice-flat pyramidals, tall interneurons) that would otherwise be
# locked into the wrong Stage 2 branch by a hard Stage-1 label.
DEFAULT_SOFT_HANDOFF_THRESHOLD = 0.65
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
        # Fallback: use any trained model if a direct cell-type match is absent.
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
    soft_handoff_threshold: float = DEFAULT_SOFT_HANDOFF_THRESHOLD,
) -> PipelineResult:
    """Run the full pipeline on pre-parsed nodes.

    When Stage 1's predicted-class probability falls below
    ``soft_handoff_threshold`` and the Stage 2 bundle has models for
    multiple cell types, the pipeline runs Stage 2+3 for *both* cell
    types and picks whichever produces higher mean per-node confidence.
    This recovers borderline files (e.g. slice-flat pyramidals or tall
    interneurons) that would otherwise be locked into the wrong Stage 2
    branch by a hard Stage-1 label.

    Pass ``soft_handoff_threshold=0.0`` to disable the soft handoff and
    use the original hard-cascade behaviour.
    """
    # --- Stage 1: Cell-type detection ---
    s1_result = detect_cell_type_from_nodes(nodes, stage1_model)

    s2_path = Path(stage2_model) if stage2_model else STAGE2_MODEL
    bundle = _load_stage2_bundle(s2_path)

    # --- Soft handoff: try both cell types when Stage 1 is uncertain ---
    chosen_ct = s1_result.cell_type
    soft_handoff_used = False
    can_handoff = (
        s1_result.confidence < soft_handoff_threshold
        and bundle.get("models_by_cell_type") is not None
        and len(bundle.get("models_by_cell_type") or {}) > 1
    )
    if can_handoff:
        candidates: list[tuple[str, dict]] = []
        for ct in CELL_TYPES:
            try:
                trial = _run_stage23(nodes, file_path, ct, bundle, s1_result)
                candidates.append((ct, trial))
            except Exception:
                # Don't let a per-branch trial failure crash the whole pipeline;
                # fall back to the hard Stage-1 prediction below.
                continue
        if candidates:
            best_ct, best_trial = max(
                candidates,
                key=lambda kv: kv[1]["mean_neurite_conf"],
            )
            chosen_ct = best_ct
            chosen = best_trial
            soft_handoff_used = (best_ct != s1_result.cell_type)
        else:
            chosen = _run_stage23(nodes, file_path, s1_result.cell_type, bundle, s1_result)
    else:
        chosen = _run_stage23(nodes, file_path, s1_result.cell_type, bundle, s1_result)

    # If the soft handoff overrode the Stage 1 label, propagate the new
    # cell type into the returned s1_result so downstream consumers see
    # the resolved type and the canonical label set for it.
    if chosen_ct != s1_result.cell_type:
        s1_result = CellTypeResult(
            cell_type=chosen_ct,
            confidence=float(s1_result.probabilities.get(chosen_ct, s1_result.confidence)),
            probabilities=dict(s1_result.probabilities),
            label_set=set(CELL_TYPE_LABEL_SETS.get(chosen_ct, {1, 2, 3})),
            structure_flags={**s1_result.structure_flags, "soft_handoff": True},
            features=dict(s1_result.features),
        )
    elif soft_handoff_used:
        # Same prediction, but record that the soft handoff was exercised.
        s1_result.structure_flags["soft_handoff"] = True

    return PipelineResult(
        stage1=s1_result,
        stage3=chosen["s3_result"],
        node_labels=chosen["final_labels"],
        node_confidences=chosen["node_confidences"],
    )


def _run_stage23(
    nodes: list[SWCNode],
    file_path: str,
    cell_type: str,
    bundle: dict,
    s1_result: CellTypeResult,
) -> dict:
    """Run Stage 2 + Stage 3 for a *given* cell type.

    Used both for the normal hard-cascade case and as the trial routine
    inside the soft-handoff branch. Returns a dict with:
        - final_labels: list[int]
        - node_confidences: list[float]
        - s3_result: RefinementResult
        - mean_neurite_conf: float (used to compare trials)
        - cell_type: str
    """
    n = len(nodes)
    label_set = set(CELL_TYPE_LABEL_SETS.get(cell_type, {1, 2, 3}))
    neurite_labels = sorted(label_set - {1})

    # --- Stage 2 model selection ---
    model, default_label = _select_stage2_model(bundle, cell_type)
    subtree_models_by_ct = bundle.get("subtree_owner_models_by_cell_type")
    if subtree_models_by_ct:
        subtree_owner_model = subtree_models_by_ct.get(cell_type)
    else:
        legacy = bundle.get("pyramidal_subtree_owner_model")
        subtree_owner_model = legacy if cell_type == "pyramidal" else None

    morph = extract_branches(nodes, cell_type, file_path)
    subtree_owner_map = _predict_subtree_owner_map(nodes, cell_type, subtree_owner_model)
    apical_owner_root = _best_apical_owner(subtree_owner_map) if cell_type == "pyramidal" else None

    proxy_soma = set(morph.soma_indices)
    node_labels = [1 if i in proxy_soma else 0 for i in range(n)]
    node_confidences = [1.0 if i in proxy_soma else 0.0 for i in range(n)]

    branch_confs: list[float] = []
    if model is not None:
        for br in morph.branches:
            X = _branch_feature_with_owner(br, subtree_owner_map).reshape(1, -1)
            probs = model.predict_proba(X)[0]
            classes = model.classes_

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

            branch_confs.append(best_conf)
            for node_idx in br.node_indices:
                node_labels[node_idx] = best_label
                node_confidences[node_idx] = best_conf
    else:
        fallback = default_label if default_label is not None else (
            neurite_labels[0] if neurite_labels else 3
        )
        for br in morph.branches:
            for node_idx in br.node_indices:
                node_labels[node_idx] = fallback
                node_confidences[node_idx] = 0.5
        # Conservative confidence so the soft-handoff comparison doesn't
        # spuriously prefer the no-model branch.
        branch_confs.append(0.5)

    for i in range(n):
        if node_labels[i] == 0:
            node_labels[i] = neurite_labels[0] if neurite_labels else 3
            node_confidences[i] = 0.3

    # Build a trial Stage-1 result with the candidate cell type so
    # Stage-3 refinement uses the matching label set.
    trial_s1 = CellTypeResult(
        cell_type=cell_type,
        confidence=float(s1_result.probabilities.get(cell_type, s1_result.confidence)),
        probabilities=dict(s1_result.probabilities),
        label_set=label_set,
        structure_flags=dict(s1_result.structure_flags),
        features=dict(s1_result.features),
    )

    s3_result = refine(
        nodes,
        node_labels,
        node_confidences,
        trial_s1,
        apical_owner_root=apical_owner_root,
    )
    final_labels = [rl.label for rl in s3_result.labels]

    mean_neurite_conf = float(np.mean(branch_confs)) if branch_confs else 0.0

    return {
        "final_labels": final_labels,
        "node_confidences": node_confidences,
        "s3_result": s3_result,
        "mean_neurite_conf": mean_neurite_conf,
        "cell_type": cell_type,
    }


def _predict_subtree_owner_map(
    nodes: list[SWCNode],
    cell_type: str,
    subtree_owner_model: object | None,
) -> dict[int, dict[str, float | int]]:
    # Any cell type with a trained subtree-owner model gets augmented
    # features. In the current benchmark: pyramidal → {axon, basal, apical},
    # interneuron → {axon, basal}.
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
