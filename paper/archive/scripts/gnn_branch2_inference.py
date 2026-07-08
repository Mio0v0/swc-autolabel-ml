"""Inference helpers for the interneuron 2-class branch rescue head (Branch2)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from hybrid.branch_features import MorphologyBranches
from paper.gnn_apical_basal import ApicalBasalSAGE, FeatureScaler
from paper.gnn_branch2_rescue import (
    BRANCH2_FEATURE_NAMES,
    CLASS_TO_LABEL,
    _branch_feature_vector,
)
from paper.gnn_dataset import _build_edge_index


@dataclass
class Branch2State:
    model: ApicalBasalSAGE
    scaler: FeatureScaler
    feature_names: tuple[str, ...]
    device: torch.device
    metadata: dict


def load_branch2(
    path: Path | str,
    device: torch.device | None = None,
) -> Branch2State:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = payload["model_config"]
    model = ApicalBasalSAGE(
        in_dim=cfg["in_dim"],
        hidden=cfg["hidden"],
        n_classes=cfg.get("n_classes", 2),
        dropout=cfg["dropout"],
        n_layers=cfg.get("n_layers", 2),
        gnn_type=cfg.get("gnn_type", "sage"),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return Branch2State(
        model=model,
        scaler=FeatureScaler.from_state(payload["scaler"]),
        feature_names=tuple(payload.get("feature_names", BRANCH2_FEATURE_NAMES)),
        device=device,
        metadata={
            "train_config": payload.get("train_config"),
            "cv_summary": payload.get("cv_summary"),
            "test_metrics": payload.get("test_metrics"),
            "final_epochs": payload.get("final_epochs"),
        },
    )


@torch.no_grad()
def score_morphology(
    state: Branch2State,
    morph: MorphologyBranches,
    subtree_owner_map: dict[int, dict[str, float | int]],
) -> dict[int, tuple[int, float, dict[int, float]]]:
    """Return {branch_id: (SWC_label, confidence, probabilities_by_label)}."""
    if not morph.branches:
        return {}

    raw_x = np.stack(
        [_branch_feature_vector(br, subtree_owner_map) for br in morph.branches]
    ).astype(np.float32)
    if tuple(state.feature_names) != tuple(BRANCH2_FEATURE_NAMES):
        raise ValueError("Branch2 checkpoint feature schema does not match current code.")
    x = (raw_x - state.scaler.mean) / state.scaler.std
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_t = torch.from_numpy(x).to(state.device)
    edge_index_t = _build_edge_index(morph.branches).to(state.device)
    logits = state.model(x_t, edge_index_t)
    probs = F.softmax(logits, dim=1).cpu().numpy()
    pred_class = probs.argmax(axis=1)
    pred_conf = probs.max(axis=1)

    out: dict[int, tuple[int, float, dict[int, float]]] = {}
    for i, br in enumerate(morph.branches):
        label = CLASS_TO_LABEL[int(pred_class[i])]
        by_label = {
            CLASS_TO_LABEL[c]: float(probs[i, c])
            for c in range(probs.shape[1])
            if c in CLASS_TO_LABEL
        }
        out[br.branch_id] = (label, float(pred_conf[i]), by_label)
    return out
