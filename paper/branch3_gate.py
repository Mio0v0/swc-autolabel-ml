"""Feature builder for the Branch3 accept/abstain gate."""
from __future__ import annotations

import numpy as np

from paper.gnn_branch3_rescue import BRANCH3_FEATURE_NAMES


TRANSITIONS: tuple[tuple[int, int], ...] = tuple(
    (cur, pred)
    for cur in range(3)
    for pred in range(3)
    if cur != pred
)

GATE_BASE_FEATURE_NAMES: tuple[str, ...] = (
    "p_axon",
    "p_basal",
    "p_apical",
    "p_pred",
    "p_current",
    "p_pred_minus_current",
    "p_top1_minus_top2",
    "p_entropy",
    "current_class",
    "pred_class",
    "current_is_axon",
    "current_is_basal",
    "current_is_apical",
    "pred_is_axon",
    "pred_is_basal",
    "pred_is_apical",
) + tuple(f"transition_{cur}_to_{pred}" for cur, pred in TRANSITIONS)

GATE_FEATURE_NAMES: tuple[str, ...] = (
    GATE_BASE_FEATURE_NAMES
    + tuple(f"branch3_x__{name}" for name in BRANCH3_FEATURE_NAMES)
)


def build_gate_features(
    raw_x: np.ndarray,
    probs: np.ndarray,
    current_class: np.ndarray,
    pred_class: np.ndarray,
) -> np.ndarray:
    """Build deployment-safe gate features.

    Parameters are arrays for the same branch rows:
    - raw_x: unscaled Branch3 input feature matrix
    - probs: Branch3 softmax probabilities, columns [axon, basal, apical]
    - current_class: current pipeline class ids [0,1,2]
    - pred_class: Branch3 argmax class ids [0,1,2]
    """
    raw_x = np.asarray(raw_x, dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32)
    current_class = np.asarray(current_class, dtype=np.int64)
    pred_class = np.asarray(pred_class, dtype=np.int64)
    n = int(probs.shape[0])
    if n == 0:
        return np.zeros((0, len(GATE_FEATURE_NAMES)), dtype=np.float32)

    rows: list[np.ndarray] = []
    p_pred = probs[np.arange(n), pred_class]
    p_current = probs[np.arange(n), current_class]
    top2 = np.sort(probs, axis=1)[:, -2:]
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-9, 1.0)), axis=1)

    base = [
        probs[:, 0],
        probs[:, 1],
        probs[:, 2],
        p_pred,
        p_current,
        p_pred - p_current,
        top2[:, 1] - top2[:, 0],
        entropy,
        current_class.astype(np.float32),
        pred_class.astype(np.float32),
    ]
    for cls in range(3):
        base.append((current_class == cls).astype(np.float32))
    for cls in range(3):
        base.append((pred_class == cls).astype(np.float32))
    for cur, pred in TRANSITIONS:
        base.append(((current_class == cur) & (pred_class == pred)).astype(np.float32))

    rows.extend(base)
    rows.append(raw_x.T)
    # ``rows`` contains 1D arrays plus a [D, N] matrix as its last item.
    head = np.stack(rows[:-1], axis=1)
    out = np.concatenate([head, rows[-1].T], axis=1).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
