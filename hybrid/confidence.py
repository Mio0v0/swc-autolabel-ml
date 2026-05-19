"""Confidence aggregation + two-level flagging for the auto-labeling pipeline.

The hybrid pipeline already produces per-node confidences (Stage 2 RF
predict_proba, optionally overridden by the Stage 3 GNN apical/basal
softmax). This module turns those raw confidences into:

  1. per-branch confidence    — mean confidence over the nodes in a branch
  2. per-cell aggregates       — mean / median / fraction of low-confidence nodes
  3. two-level flagging        — per-branch and per-cell flags against
                                 calibrated thresholds

The thresholds themselves are loaded from a JSON config that the
calibration script (paper/_calibrate_flags.py) produces from a held-out
validation split.

Typical usage:
    from hybrid.confidence import (
        ConfidenceConfig, summarize_confidence, apply_two_level_flag,
    )
    from hybrid.pipeline import run_pipeline_on_nodes

    result = run_pipeline_on_nodes(nodes, ...)
    cfg = ConfidenceConfig.load(path)           # or .default()
    summary = summarize_confidence(nodes, result, cfg)
    flag = apply_two_level_flag(summary, cfg)
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from .features import SWCNode

# Node SWC type → conceptual class name
LABEL_NAMES = {1: "soma", 2: "axon", 3: "basal/dendrite", 4: "apical"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class ConfidenceConfig:
    """Calibrated thresholds for the two-level flag.

    Defaults are conservative starting points; the calibration script
    overwrites them with values that target a specific precision on a
    held-out validation set.
    """

    # Per-NODE confidence below which a node is "uncertain"
    node_low_threshold: float = 0.70
    # Per-BRANCH flag fires if branch mean confidence < this
    branch_flag_threshold: float = 0.75
    # Per-CELL flag fires if EITHER condition is true:
    #   - cell mean confidence < cell_mean_threshold, OR
    #   - fraction of low-confidence nodes > cell_low_fraction_threshold
    cell_mean_threshold: float = 0.85
    cell_low_fraction_threshold: float = 0.20
    # Stage 1 cell-type-confidence under which the whole cell is flagged
    # (this catches the "ambiguous cell type" case)
    stage1_low_threshold: float = 0.60

    # Provenance: which calibration run produced this config?
    calibration_id: str = ""
    target_precision: float = 0.0
    note: str = ""

    @classmethod
    def default(cls) -> "ConfidenceConfig":
        return cls(note="default (uncalibrated)")

    @classmethod
    def load(cls, path: Path | str) -> "ConfidenceConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-branch + per-cell summary
# ---------------------------------------------------------------------------
@dataclass
class BranchConfidence:
    """Confidence summary for one branch.

    A 'branch' here is the contiguous run of nodes between branch points
    or between a branch point and a tip. We segment by walking parent
    links and breaking at any node that has != 1 child or has type != its
    parent's type.
    """

    branch_id: int
    node_indices: list[int]
    n_nodes: int
    predicted_label: int                # majority label in this branch
    mean_confidence: float
    min_confidence: float
    n_low_confidence_nodes: int         # below ConfidenceConfig.node_low_threshold
    flag: bool = False                  # filled in by apply_two_level_flag


@dataclass
class CellConfidence:
    """Cell-level confidence summary."""

    n_nodes: int
    mean_node_confidence: float
    median_node_confidence: float
    fraction_low_confidence: float
    stage1_cell_type: str
    stage1_confidence: float
    # Per-class aggregates
    per_class_mean_confidence: dict[str, float] = field(default_factory=dict)
    per_class_node_count: dict[str, int] = field(default_factory=dict)
    # Filled in by apply_two_level_flag
    flag: bool = False
    flag_reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Branch segmentation (linear runs between bifurcations / type changes)
# ---------------------------------------------------------------------------
def _segment_branches(
    nodes: list[SWCNode], labels: list[int]
) -> list[list[int]]:
    """Return a list of branches, each a list of node *indices* into ``nodes``.

    A branch breaks at:
      - any bifurcation (parent has > 1 child), OR
      - any label change (the label of a node differs from its parent's), OR
      - the root.

    This matches how we'd want a human reviewer to scan: each segment is
    homogeneous in label and bounded by structural features.
    """
    n = len(nodes)
    id_to_idx = {nd.id: i for i, nd in enumerate(nodes)}
    children: list[list[int]] = [[] for _ in range(n)]
    parent_idx: list[int] = [-1] * n
    for i, nd in enumerate(nodes):
        if nd.parent != -1 and nd.parent in id_to_idx:
            p = id_to_idx[nd.parent]
            parent_idx[i] = p
            children[p].append(i)

    # A node starts a NEW branch if:
    #   it is a root, OR
    #   its parent has >1 child (it's at a bifurcation), OR
    #   its label differs from parent's label.
    starts = [False] * n
    for i in range(n):
        p = parent_idx[i]
        if p < 0:
            starts[i] = True
        elif len(children[p]) > 1:
            starts[i] = True
        elif labels[i] != labels[p]:
            starts[i] = True

    # Each node is assigned a branch_id = the start node it descends from
    # without crossing another start.
    branch_id_of: list[int] = [-1] * n
    # Walk in topo order: roots first. We have parent_idx; iterate
    # nodes that come AFTER their parents by simple stable order
    # (SWC files conventionally list parents first; this holds in our
    # corpus). For safety, do a parent-first walk explicitly.
    order: list[int] = []
    seen = [False] * n
    # BFS from roots
    queue = [i for i in range(n) if parent_idx[i] < 0]
    while queue:
        nq = []
        for i in queue:
            if seen[i]:
                continue
            seen[i] = True
            order.append(i)
            nq.extend(children[i])
        queue = nq

    for i in order:
        if starts[i]:
            branch_id_of[i] = i
        else:
            branch_id_of[i] = branch_id_of[parent_idx[i]]

    # Group node indices by branch_id
    by_branch: dict[int, list[int]] = {}
    for i in range(n):
        by_branch.setdefault(branch_id_of[i], []).append(i)
    # Return as list, sorted by smallest node index (stable across runs)
    return [by_branch[k] for k in sorted(by_branch)]


# ---------------------------------------------------------------------------
# Build the summary
# ---------------------------------------------------------------------------
def summarize_confidence(
    nodes: list[SWCNode],
    labels: Iterable[int],
    confidences: Iterable[float],
    stage1_cell_type: str,
    stage1_confidence: float,
    cfg: ConfidenceConfig | None = None,
) -> tuple[list[BranchConfidence], CellConfidence]:
    """Compute per-branch + per-cell confidence summaries.

    Args:
        nodes: list of SWCNode (parsed from the input file)
        labels: per-node predicted labels (same length as nodes)
        confidences: per-node ML confidence (0..1), same length as nodes
        stage1_cell_type: e.g. "pyramidal" / "interneuron"
        stage1_confidence: Stage 1's cell-type-prediction probability
        cfg: ConfidenceConfig (defaults if None)

    Returns:
        (list[BranchConfidence], CellConfidence)
    """
    cfg = cfg or ConfidenceConfig.default()
    labels = list(labels)
    confidences = [float(c) for c in confidences]
    assert len(labels) == len(nodes), "labels length != nodes length"
    assert len(confidences) == len(nodes), "confidences length != nodes length"

    # Per-cell aggregates
    if not confidences:
        cell = CellConfidence(
            n_nodes=0, mean_node_confidence=0.0, median_node_confidence=0.0,
            fraction_low_confidence=0.0,
            stage1_cell_type=stage1_cell_type, stage1_confidence=stage1_confidence,
        )
        return [], cell

    mean_conf = sum(confidences) / len(confidences)
    median_conf = statistics.median(confidences)
    n_low = sum(1 for c in confidences if c < cfg.node_low_threshold)
    frac_low = n_low / len(confidences)

    # Per-class breakdown
    per_class_mean: dict[str, float] = {}
    per_class_count: dict[str, int] = {}
    for cls_id, cls_name in LABEL_NAMES.items():
        idxs = [i for i, lab in enumerate(labels) if lab == cls_id]
        per_class_count[cls_name] = len(idxs)
        if idxs:
            per_class_mean[cls_name] = sum(confidences[i] for i in idxs) / len(idxs)

    cell = CellConfidence(
        n_nodes=len(nodes),
        mean_node_confidence=mean_conf,
        median_node_confidence=median_conf,
        fraction_low_confidence=frac_low,
        stage1_cell_type=stage1_cell_type,
        stage1_confidence=stage1_confidence,
        per_class_mean_confidence=per_class_mean,
        per_class_node_count=per_class_count,
    )

    # Per-branch aggregates
    segs = _segment_branches(nodes, labels)
    branches: list[BranchConfidence] = []
    for branch_idx, node_idxs in enumerate(segs):
        if not node_idxs:
            continue
        branch_labels = [labels[i] for i in node_idxs]
        # Majority label
        from collections import Counter
        majority = Counter(branch_labels).most_common(1)[0][0]
        branch_confs = [confidences[i] for i in node_idxs]
        mean_c = sum(branch_confs) / len(branch_confs)
        min_c = min(branch_confs)
        n_low_branch = sum(1 for c in branch_confs if c < cfg.node_low_threshold)
        branches.append(BranchConfidence(
            branch_id=branch_idx,
            node_indices=node_idxs,
            n_nodes=len(node_idxs),
            predicted_label=int(majority),
            mean_confidence=mean_c,
            min_confidence=min_c,
            n_low_confidence_nodes=n_low_branch,
        ))

    return branches, cell


# ---------------------------------------------------------------------------
# Two-level flag
# ---------------------------------------------------------------------------
def apply_two_level_flag(
    branches: list[BranchConfidence],
    cell: CellConfidence,
    cfg: ConfidenceConfig | None = None,
) -> tuple[list[BranchConfidence], CellConfidence]:
    """Set the .flag fields on branches and on the cell.

    Per-branch:
        flag = (branch.mean_confidence < cfg.branch_flag_threshold)
    Per-cell:
        flag = (cell.mean_node_confidence < cfg.cell_mean_threshold)
            OR (cell.fraction_low_confidence > cfg.cell_low_fraction_threshold)
            OR (cell.stage1_confidence < cfg.stage1_low_threshold)
    """
    cfg = cfg or ConfidenceConfig.default()

    for b in branches:
        b.flag = b.mean_confidence < cfg.branch_flag_threshold

    reasons: list[str] = []
    if cell.mean_node_confidence < cfg.cell_mean_threshold:
        reasons.append(f"mean_conf<{cfg.cell_mean_threshold:.2f}")
    if cell.fraction_low_confidence > cfg.cell_low_fraction_threshold:
        reasons.append(f"frac_low>{cfg.cell_low_fraction_threshold:.2f}")
    if cell.stage1_confidence < cfg.stage1_low_threshold:
        reasons.append(f"stage1_conf<{cfg.stage1_low_threshold:.2f}")

    cell.flag = bool(reasons)
    cell.flag_reasons = reasons
    return branches, cell


# ---------------------------------------------------------------------------
# JSON report (for the CLI wrapper)
# ---------------------------------------------------------------------------
def report_to_dict(
    branches: list[BranchConfidence],
    cell: CellConfidence,
    cfg: ConfidenceConfig,
) -> dict:
    """Compact JSON-serializable report for inclusion in the CLI output."""
    return {
        "cell": {
            "n_nodes":                cell.n_nodes,
            "mean_node_confidence":   round(cell.mean_node_confidence, 4),
            "median_node_confidence": round(cell.median_node_confidence, 4),
            "fraction_low_confidence": round(cell.fraction_low_confidence, 4),
            "stage1_cell_type":       cell.stage1_cell_type,
            "stage1_confidence":      round(cell.stage1_confidence, 4),
            "per_class_mean_confidence": {
                k: round(v, 4) for k, v in cell.per_class_mean_confidence.items()
            },
            "per_class_node_count": cell.per_class_node_count,
            "flag":                  cell.flag,
            "flag_reasons":          cell.flag_reasons,
        },
        "branches_flagged": [
            {
                "branch_id":         b.branch_id,
                "predicted_label":   b.predicted_label,
                "predicted_name":    LABEL_NAMES.get(b.predicted_label, str(b.predicted_label)),
                "n_nodes":           b.n_nodes,
                "mean_confidence":   round(b.mean_confidence, 4),
                "min_confidence":    round(b.min_confidence, 4),
                "node_indices":      b.node_indices,
            }
            for b in branches if b.flag
        ],
        "n_branches_total":   len(branches),
        "n_branches_flagged": sum(1 for b in branches if b.flag),
        "config": asdict(cfg),
    }
