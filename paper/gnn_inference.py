"""Inference shim for the apical-vs-basal GraphSAGE head.

Used by the production pipeline (`hybrid.pipeline.run_pipeline_on_nodes`)
when `use_gnn=True`. Stays in `paper/` rather than `hybrid/` so the core
pipeline keeps no torch dependency unless the GNN is actually requested.

API:
    state = load_gnn(path, device=None)            -> GNNState
    preds = score_morphology(state, mb)            -> dict[branch_id -> (label, conf)]
        - mb is a `hybrid.branch_features.MorphologyBranches`.
        - Returned labels use the SWC type-column convention:
              3 = basal/dendrite, 4 = apical
        - Returned conf is the softmax probability of the chosen class.
        - Only branches with `n_branches >= 1` of the input get an entry;
          callers should treat any missing branch_id as "GNN abstains".

The model and scaler are loaded once into a `GNNState` and reused across
files (the pipeline calls `score_morphology` per cell). The Data graphs
are built on the fly via `paper.gnn_dataset.morphology_to_data`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from hybrid.branch_features import MorphologyBranches
from paper.gnn_apical_basal import (
    ApicalBasalSAGE,
    FeatureScaler,
    DEFAULT_CKPT_PATH,
)
from paper.gnn_dataset import (
    APICAL_LABEL,
    BASAL_LABEL,
    CLASS_APICAL,
    CLASS_BASAL,
    morphology_to_data,
)


@dataclass
class GNNState:
    model: ApicalBasalSAGE
    scaler: FeatureScaler
    feature_names: tuple[str, ...]
    device: torch.device
    metadata: dict


def load_gnn(
    path: Path | str = DEFAULT_CKPT_PATH,
    device: Optional[torch.device] = None,
) -> GNNState:
    """Load a checkpoint produced by `paper.gnn_apical_basal.save_checkpoint`."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = payload["model_config"]
    model = ApicalBasalSAGE(
        in_dim=cfg["in_dim"],
        hidden=cfg["hidden"],
        n_classes=cfg["n_classes"],
        dropout=cfg["dropout"],
        n_layers=cfg.get("n_layers", 2),
        gnn_type=cfg.get("gnn_type", "sage"),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    scaler = FeatureScaler.from_state(payload["scaler"])
    feature_names = tuple(payload["feature_names"])
    return GNNState(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
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
    state: GNNState,
    mb: MorphologyBranches,
) -> dict[int, tuple[int, float]]:
    """Return GNN predictions for every branch in `mb`.

    Output:
        { branch_id: (predicted_swc_label, confidence) }
    where predicted_swc_label in {BASAL_LABEL=3, APICAL_LABEL=4} and
    confidence is the softmax probability of the chosen class.

    The whole branch graph is fed forward (axon/soma branches included as
    message-passing context); only the branches in `mb.branches` are
    returned. Caller decides which Stage 2 predictions to override
    (typically only branches Stage 2 classified as dendrite).
    """
    if not mb.branches:
        return {}

    # Build a Data graph using the same feature subset the model was trained on.
    data = morphology_to_data(
        mb,
        feature_names=state.feature_names,
        only_pyramidal_dendrites=False,  # we want ALL branches as context
    )
    # Apply the training-time z-score scaler.
    x = data.x.numpy()
    x = (x - state.scaler.mean) / state.scaler.std
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_t = torch.from_numpy(x.astype(np.float32)).to(state.device)
    edge_index_t = data.edge_index.to(state.device)

    logits = state.model(x_t, edge_index_t)  # [N, 2]
    probs = F.softmax(logits, dim=1).cpu().numpy()
    pred_class = probs.argmax(axis=1)
    pred_conf = probs.max(axis=1)

    out: dict[int, tuple[int, float]] = {}
    for i, br in enumerate(mb.branches):
        cls = int(pred_class[i])
        if cls == CLASS_APICAL:
            label = APICAL_LABEL
        elif cls == CLASS_BASAL:
            label = BASAL_LABEL
        else:
            # Future-proofing if model ever has a 3rd class
            continue
        out[br.branch_id] = (label, float(pred_conf[i]))
    return out
