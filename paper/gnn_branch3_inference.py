"""Inference helpers for the optional 3-class pyramidal branch rescue head."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from hybrid.branch_features import MorphologyBranches
from paper.branch3_gate import GATE_FEATURE_NAMES, build_gate_features
from paper.gnn_apical_basal import ApicalBasalSAGE, FeatureScaler
from paper.gnn_branch3_rescue import (
    BRANCH3_FEATURE_NAMES,
    CLASS_TO_LABEL,
    _branch_feature_vector,
)
from paper.gnn_dataset import _build_edge_index


@dataclass
class Branch3State:
    model: ApicalBasalSAGE
    scaler: FeatureScaler
    feature_names: tuple[str, ...]
    device: torch.device
    metadata: dict
    gate: dict | None = None


def load_branch3(
    path: Path | str,
    device: torch.device | None = None,
    gate_path: Path | str | None = None,
) -> Branch3State:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = payload["model_config"]
    model = ApicalBasalSAGE(
        in_dim=cfg["in_dim"],
        hidden=cfg["hidden"],
        n_classes=cfg.get("n_classes", 3),
        dropout=cfg["dropout"],
        n_layers=cfg.get("n_layers", 2),
        gnn_type=cfg.get("gnn_type", "sage"),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    gate = None
    env_gate = os.environ.get("SWCAL_BRANCH3_GATE_PATH")
    if gate_path is None and env_gate:
        gate_path = env_gate
    if gate_path is not None:
        gate = joblib.load(Path(gate_path))
        if tuple(gate.get("feature_names", [])) != tuple(GATE_FEATURE_NAMES):
            raise ValueError(f"Branch3 gate feature schema mismatch: {gate_path}")

    return Branch3State(
        model=model,
        scaler=FeatureScaler.from_state(payload["scaler"]),
        feature_names=tuple(payload.get("feature_names", BRANCH3_FEATURE_NAMES)),
        device=device,
        metadata={
            "train_config": payload.get("train_config"),
            "cv_summary": payload.get("cv_summary"),
            "test_metrics": payload.get("test_metrics"),
            "final_epochs": payload.get("final_epochs"),
        },
        gate=gate,
    )


@torch.no_grad()
def score_morphology(
    state: Branch3State,
    morph: MorphologyBranches,
    subtree_owner_map: dict[int, dict[str, float | int]],
    current_labels: Sequence[int],
    current_confidences: Sequence[float],
) -> dict[int, tuple[int, float, dict[int, float], float | None]]:
    """Return {branch_id: (SWC_label, confidence, probabilities_by_label, gate_score)}."""
    if not morph.branches:
        return {}

    raw_x = np.stack(
        [
            _branch_feature_vector(br, subtree_owner_map, current_labels[i], current_confidences[i])
            for i, br in enumerate(morph.branches)
        ]
    ).astype(np.float32)
    if tuple(state.feature_names) != tuple(BRANCH3_FEATURE_NAMES):
        # The checkpoint records the exact feature schema. Current code only
        # supports the canonical schema because the features are generated from
        # local branch/state objects, not selected by name.
        raise ValueError("Branch3 checkpoint feature schema does not match current code.")
    x = (raw_x - state.scaler.mean) / state.scaler.std
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_t = torch.from_numpy(x).to(state.device)
    edge_index_t = _build_edge_index(morph.branches).to(state.device)
    logits = state.model(x_t, edge_index_t)
    probs = F.softmax(logits, dim=1).cpu().numpy()
    pred_class = probs.argmax(axis=1)
    pred_conf = probs.max(axis=1)
    gate_scores: dict[int, float] = {}
    if state.gate is not None:
        current_class = np.asarray(
            [
                0 if int(lbl) == 2 else 1 if int(lbl) == 3 else 2
                for lbl in current_labels
            ],
            dtype=np.int64,
        )
        changed = pred_class != current_class
        if np.any(changed):
            X_gate = build_gate_features(
                raw_x[changed],
                probs[changed],
                current_class[changed],
                pred_class[changed],
            )
            model = state.gate["model"]
            scores = model.predict_proba(X_gate)[:, 1]
            for idx, score in zip(np.flatnonzero(changed), scores):
                gate_scores[int(idx)] = float(score)

    out: dict[int, tuple[int, float, dict[int, float], float | None]] = {}
    for i, br in enumerate(morph.branches):
        label = CLASS_TO_LABEL[int(pred_class[i])]
        by_label = {
            CLASS_TO_LABEL[c]: float(probs[i, c])
            for c in range(probs.shape[1])
            if c in CLASS_TO_LABEL
        }
        out[br.branch_id] = (label, float(pred_conf[i]), by_label, gate_scores.get(i))
    return out
