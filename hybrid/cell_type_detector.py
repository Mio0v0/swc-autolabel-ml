"""Stage 1: Cell-type and structure detection.

Classifies whole morphologies into the current benchmark cell types and determines:
- cell_type: pyramidal or interneuron
- label_set: which SWC type codes to assign (e.g. {1,2,3,4} or {1,2,3})
- structure_flags: purely morphology-derived properties

This stage must not depend on ground-truth SWC type labels. Any trained
model or heuristic that uses label-distribution features is considered
leaky and is rejected.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

from .features import (
    FEATURE_NAMES,
    SWCNode,
    extract_feature_vector,
    extract_global_features,
    parse_swc,
)

# ---------------------------------------------------------------------------
# Cell type definitions
# ---------------------------------------------------------------------------

CELL_TYPES = [
    "pyramidal",
    "interneuron",
]

# What label sets each cell type typically uses
CELL_TYPE_LABEL_SETS: dict[str, set[int]] = {
    "pyramidal": {1, 2, 3, 4},       # soma, axon, basal, apical
    "interneuron": {1, 2, 3},         # soma, axon, dendrite (basal only)
}

LEAKY_STAGE1_FEATURES = {
    "frac_type_1",
    "frac_type_2",
    "frac_type_3",
    "frac_type_4",
    "n_unique_types",
}


@dataclass
class CellTypeResult:
    """Output of the cell-type detection stage."""
    cell_type: str                          # predicted cell type
    confidence: float                       # classifier confidence (max probability)
    probabilities: dict[str, float]         # per-class probabilities
    label_set: set[int]                     # SWC types to assign
    structure_flags: dict[str, bool]        # detected structural properties
    features: dict[str, float]             # extracted global features (for debugging)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["label_set"] = sorted(self.label_set)
        return d


# ---------------------------------------------------------------------------
# Structure flag detection (rule-based, runs after cell type is known)
# ---------------------------------------------------------------------------

def _detect_structure_flags(
    features: dict[str, float],
    cell_type: str,
) -> dict[str, bool]:
    """Detect structural properties from global features."""
    flags: dict[str, bool] = {}

    # Multiple disconnected roots in the SWC topology
    flags["multi_root"] = features.get("n_root_nodes", 1) > 1

    # Z-asymmetry can indicate apical presence for pyramidal cells
    z_asym = features.get("z_asymmetry", 0.0)
    z_span = features.get("z_span", 0.0)
    flags["z_asymmetric"] = abs(z_asym) > 0.3 and z_span > 100.0
    flags["apical_candidate"] = cell_type == "pyramidal" and abs(z_asym) > 0.2 and z_span > 150.0

    # Large morphology (many nodes)
    flags["large_morphology"] = features.get("n_nodes", 0) > 5000

    # High branching complexity
    flags["high_branching"] = features.get("branch_point_ratio", 0) > 0.15

    return flags


# ---------------------------------------------------------------------------
# Heuristic classifier (fallback when no trained model available)
# ---------------------------------------------------------------------------

def _heuristic_classify(features: dict[str, float]) -> tuple[str, dict[str, float]]:
    """Rule-based cell-type classification as fallback.

    Returns (predicted_type, probability_dict).
    This is a reasonable heuristic but should be replaced by a trained model
    once sufficient training data is available.
    """
    scores: dict[str, float] = {ct: 0.0 for ct in CELL_TYPES}

    n_primary = features.get("n_primary_subtrees", 0)
    z_asym = features.get("z_asymmetry", 0.0)
    z_span = features.get("z_span", 0.0)
    branch_ratio = features.get("branch_point_ratio", 0.0)
    max_strahler = features.get("max_strahler", 0)
    n_nodes = features.get("n_nodes", 0)
    mean_radius = features.get("mean_radius", 0.0)
    max_subtree_size = features.get("max_subtree_size", 0)
    subtree_size_std = features.get("subtree_size_std", 0)

    # --- Pyramidal indicators ---
    # Multiple subtrees, moderate branching, often z-asymmetric
    if n_primary >= 3:
        scores["pyramidal"] += 0.15
    if z_span > 200 and abs(z_asym) > 0.2:
        scores["pyramidal"] += 0.20
    if max_subtree_size > 1000 and subtree_size_std > 200:
        scores["pyramidal"] += 0.20
    if 0.05 < branch_ratio < 0.18:
        scores["pyramidal"] += 0.10
    # Subtree size asymmetry (one big apical vs several small basals)
    if n_primary >= 3 and subtree_size_std > 200:
        scores["pyramidal"] += 0.10

    # --- Interneuron indicators ---
    # Typically more symmetric, no apical, often denser branching
    if abs(z_asym) < 0.15 and z_span < 250:
        scores["interneuron"] += 0.20
    if abs(z_asym) < 0.15:
        scores["interneuron"] += 0.15
    if branch_ratio > 0.10:
        scores["interneuron"] += 0.10
    if n_primary >= 3 and subtree_size_std < 300:
        scores["interneuron"] += 0.10
    if n_nodes < 3000:
        scores["interneuron"] += 0.05

    # Normalize to probabilities
    total = sum(scores.values())
    if total <= 0:
        scores = {ct: 1.0 for ct in CELL_TYPES}
        total = float(len(CELL_TYPES))
    probs = {ct: s / total for ct, s in scores.items()}

    best = max(probs, key=lambda ct: probs[ct])
    return best, probs


# ---------------------------------------------------------------------------
# Trained model classifier
# ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).parent / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "cell_type_classifier.pkl"
DEFAULT_META_PATH = MODEL_DIR / "cell_type_meta.json"


class CellTypeClassifier:
    """Wraps a trained sklearn model for cell-type prediction."""

    def __init__(self, model: Any = None, feature_names: list[str] | None = None):
        self.model = model
        self.feature_names = feature_names or list(FEATURE_NAMES)
        self._classes: list[str] = list(CELL_TYPES)

    @classmethod
    def load(cls, model_path: str | Path = DEFAULT_MODEL_PATH) -> "CellTypeClassifier":
        """Load a trained model from disk."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                "Run hybrid/train_stage1.py first to train a model, "
                "or the heuristic fallback will be used."
            )
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.model = data["model"]
        obj._classes = data.get("classes", list(CELL_TYPES))
        obj.feature_names = data.get("feature_names", list(FEATURE_NAMES))
        if any(name in LEAKY_STAGE1_FEATURES for name in obj.feature_names):
            raise ValueError(
                f"Leaky Stage 1 model at {model_path}: feature_names contain SWC type-derived fields. Retrain hybrid/train_stage1.py."
            )
        if obj.feature_names != list(FEATURE_NAMES):
            raise ValueError(
                f"Stage 1 model at {model_path} has stale feature schema. Retrain hybrid/train_stage1.py."
            )
        return obj

    def save(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        """Save the trained model to disk."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "classes": self._classes,
                "feature_names": self.feature_names,
            }, f)

    def predict(self, feature_vector: np.ndarray) -> tuple[str, dict[str, float]]:
        """Predict cell type from a feature vector.

        Returns (cell_type, probability_dict).
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load() first.")

        X = feature_vector.reshape(1, -1)
        probs = self.model.predict_proba(X)[0]
        prob_dict = {c: float(p) for c, p in zip(self._classes, probs)}
        best = self._classes[int(np.argmax(probs))]
        return best, prob_dict


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

def detect_cell_type(
    swc_path: str | Path,
    model_path: str | Path | None = None,
) -> CellTypeResult:
    """Run Stage 1 cell-type detection on an SWC file.

    If a trained model exists, uses it. Otherwise falls back to heuristics.
    """
    nodes = parse_swc(swc_path)
    if not nodes:
        return CellTypeResult(
            cell_type="interneuron",
            confidence=0.0,
            probabilities={ct: 1.0 / len(CELL_TYPES) for ct in CELL_TYPES},
            label_set={1, 2, 3},
            structure_flags={},
            features={},
        )

    features = extract_global_features(nodes)
    feature_vec = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)

    # Try trained model first, fall back to heuristics
    mpath = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    try:
        classifier = CellTypeClassifier.load(mpath)
        cell_type, probs = classifier.predict(feature_vec)
    except (FileNotFoundError, Exception):
        cell_type, probs = _heuristic_classify(features)

    confidence = probs.get(cell_type, 0.0)
    label_set = set(CELL_TYPE_LABEL_SETS.get(cell_type, {1, 2, 3}))

    # Refine label set based on structure detection
    flags = _detect_structure_flags(features, cell_type)

    # If pyramidal but no evidence of apical, downgrade to 3-class
    if cell_type == "pyramidal" and not flags.get("apical_candidate") and not flags.get("z_asymmetric"):
        # Check if the second-best is interneuron with close probability
        if probs.get("interneuron", 0.0) > probs.get("pyramidal", 0.0) * 0.7:
            label_set = {1, 2, 3}

    return CellTypeResult(
        cell_type=cell_type,
        confidence=confidence,
        probabilities=probs,
        label_set=label_set,
        structure_flags=flags,
        features=features,
    )


def detect_cell_type_from_nodes(
    nodes: list[SWCNode],
    model_path: str | Path | None = None,
) -> CellTypeResult:
    """Run Stage 1 on pre-parsed nodes (avoids re-reading file)."""
    if not nodes:
        return CellTypeResult(
            cell_type="interneuron",
            confidence=0.0,
            probabilities={ct: 1.0 / len(CELL_TYPES) for ct in CELL_TYPES},
            label_set={1, 2, 3},
            structure_flags={},
            features={},
        )

    features = extract_global_features(nodes)
    feature_vec = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)

    mpath = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    try:
        classifier = CellTypeClassifier.load(mpath)
        cell_type, probs = classifier.predict(feature_vec)
    except (FileNotFoundError, Exception):
        cell_type, probs = _heuristic_classify(features)

    confidence = probs.get(cell_type, 0.0)
    label_set = set(CELL_TYPE_LABEL_SETS.get(cell_type, {1, 2, 3}))
    flags = _detect_structure_flags(features, cell_type)

    if cell_type == "pyramidal" and not flags.get("apical_candidate") and not flags.get("z_asymmetric"):
        if probs.get("interneuron", 0.0) > probs.get("pyramidal", 0.0) * 0.7:
            label_set = {1, 2, 3}

    return CellTypeResult(
        cell_type=cell_type,
        confidence=confidence,
        probabilities=probs,
        label_set=label_set,
        structure_flags=flags,
        features=features,
    )
