"""Stage 3: Topology-aware refinement.

Applies structural constraints and label propagation to clean up
Stage 2 branch classifications. This is purely rule-based and uses
the cell type from Stage 1 to select which constraints apply.

Key refinements:
1. Primary subtree voting — majority label within each soma-child subtree
2. Single-axon / single-apical constraints (pyramidal only)
3. Parent-child label propagation (confidence-weighted smoothing)
4. Island flipping — small isolated segments adopt neighbor labels
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .features import SWCNode, parse_swc
from .cell_type_detector import CellTypeResult


@dataclass
class RefinedLabel:
    """Per-node label assignment after refinement."""
    node_id: int
    label: int
    confidence: float           # ML confidence for this label
    was_refined: bool           # True if changed by Stage 3


@dataclass
class RefinementResult:
    """Output of Stage 3."""
    labels: list[RefinedLabel]  # one per node
    n_refined: int              # how many labels were changed
    subtree_labels: dict[int, int]  # soma_child_id -> assigned label
    cell_type: str


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------

def _build_tree(nodes: list[SWCNode]) -> tuple[
    dict[int, int],         # id -> index
    list[int | None],       # parent_idx
    list[list[int]],        # children
    list[int],              # roots
]:
    id_to_idx = {n.id: i for i, n in enumerate(nodes)}
    n = len(nodes)
    parent_idx: list[int | None] = [None] * n
    children: list[list[int]] = [[] for _ in range(n)]
    roots: list[int] = []
    for i, nd in enumerate(nodes):
        pidx = id_to_idx.get(nd.parent)
        parent_idx[i] = pidx
        if pidx is not None:
            children[pidx].append(i)
        if nd.parent == -1 or pidx is None:
            roots.append(i)
    return id_to_idx, parent_idx, children, roots


def _euclidean(a: SWCNode, b: SWCNode) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _select_proxy_root(
    nodes: list[SWCNode],
    children: list[list[int]],
    roots: list[int],
) -> int:
    if not nodes:
        return 0
    candidate_roots = roots or [0]
    return max(
        candidate_roots,
        key=lambda idx: (nodes[idx].radius, len(children[idx]), -idx),
    )


def _subtree_indices(root: int, children: list[list[int]]) -> list[int]:
    """Get all node indices in the subtree rooted at root."""
    result: list[int] = []
    stack = [root]
    while stack:
        idx = stack.pop()
        result.append(idx)
        stack.extend(children[idx])
    return result


def _path_length_from(
    start: int,
    nodes: list[SWCNode],
    parent_idx: list[int | None],
    children: list[list[int]],
) -> dict[int, float]:
    """BFS path length from start to all descendants."""
    dist: dict[int, float] = {start: 0.0}
    queue = [start]
    while queue:
        idx = queue.pop(0)
        for ci in children[idx]:
            d = dist[idx] + _euclidean(nodes[idx], nodes[ci])
            dist[ci] = d
            queue.append(ci)
    return dist


# ---------------------------------------------------------------------------
# Refinement strategies
# ---------------------------------------------------------------------------

def _primary_subtree_voting(
    nodes: list[SWCNode],
    labels: list[int],
    confidences: list[float],
    parent_idx: list[int | None],
    children: list[list[int]],
    soma_indices: set[int],
    label_set: set[int],
) -> tuple[list[int], dict[int, int]]:
    """Assign each primary subtree a single label via confidence-weighted voting.

    For each soma child, collect all non-soma nodes in its subtree,
    compute confidence-weighted label votes, and assign the winner
    to the entire subtree.
    """
    out = list(labels)
    subtree_labels: dict[int, int] = {}

    # Find primary children of soma
    primary_roots: list[int] = []
    for si in soma_indices:
        for ci in children[si]:
            if ci not in soma_indices:
                primary_roots.append(ci)

    neurite_labels = label_set - {1}
    if not neurite_labels:
        return out, subtree_labels

    for pr in primary_roots:
        subtree = _subtree_indices(pr, children)
        if not subtree:
            continue

        # Confidence-weighted voting (exclude soma nodes)
        votes: dict[int, float] = {lbl: 0.0 for lbl in neurite_labels}
        for idx in subtree:
            lbl = labels[idx]
            if lbl in neurite_labels:
                votes[lbl] += confidences[idx]

        if not any(v > 0 for v in votes.values()):
            continue

        winner = max(neurite_labels, key=lambda lbl: votes.get(lbl, 0.0))
        subtree_labels[pr] = winner

        # Apply to all nodes in subtree
        for idx in subtree:
            if labels[idx] in neurite_labels:
                out[idx] = winner

    return out, subtree_labels


def _enforce_single_class(
    subtree_labels: dict[int, int],
    nodes: list[SWCNode],
    children: list[list[int]],
    labels: list[int],
    confidences: list[float],
    target_class: int,
    fallback_class: int,
) -> tuple[dict[int, int], list[int]]:
    """Enforce that at most one primary subtree has target_class.

    The subtree with highest total confidence for target_class wins.
    Others are reassigned to fallback_class.
    """
    out_labels = list(labels)
    out_subtrees = dict(subtree_labels)

    owners = [pr for pr, lbl in subtree_labels.items() if lbl == target_class]
    if len(owners) <= 1:
        return out_subtrees, out_labels

    # Pick the best one by total confidence for this class
    best_pr = None
    best_score = -1.0
    for pr in owners:
        subtree = _subtree_indices(pr, children)
        score = sum(confidences[idx] for idx in subtree if labels[idx] == target_class)
        if score > best_score:
            best_score = score
            best_pr = pr

    # Reassign losers
    for pr in owners:
        if pr == best_pr:
            continue
        out_subtrees[pr] = fallback_class
        subtree = _subtree_indices(pr, children)
        for idx in subtree:
            if out_labels[idx] == target_class:
                out_labels[idx] = fallback_class

    return out_subtrees, out_labels


def _parent_child_smoothing(
    nodes: list[SWCNode],
    labels: list[int],
    confidences: list[float],
    parent_idx: list[int | None],
    children: list[list[int]],
    soma_indices: set[int],
    label_set: set[int],
    iterations: int = 3,
    parent_weight: float = 0.25,
    child_weight: float = 0.15,
    confidence_threshold: float = 0.7,
) -> list[int]:
    """Smooth labels by propagating from parent/children.

    Only overrides nodes whose ML confidence is below the threshold.
    High-confidence predictions are kept.
    """
    neurite_labels = sorted(label_set - {1})
    if len(neurite_labels) <= 1:
        return labels

    out = list(labels)
    n = len(nodes)

    for _ in range(iterations):
        new_labels = list(out)
        for i in range(n):
            if i in soma_indices or out[i] == 1:
                continue
            if confidences[i] >= confidence_threshold:
                continue

            # Count neighbor labels
            votes: dict[int, float] = {lbl: 0.0 for lbl in neurite_labels}
            # Self vote (weighted by confidence)
            if out[i] in votes:
                votes[out[i]] += confidences[i]

            # Parent vote
            pidx = parent_idx[i]
            if pidx is not None and pidx not in soma_indices and out[pidx] in votes:
                votes[out[pidx]] += parent_weight

            # Children votes
            for ci in children[i]:
                if ci not in soma_indices and out[ci] in votes:
                    votes[out[ci]] += child_weight

            winner = max(neurite_labels, key=lambda lbl: votes.get(lbl, 0.0))
            new_labels[i] = winner

        out = new_labels

    return out


def _island_flipping(
    nodes: list[SWCNode],
    labels: list[int],
    confidences: list[float],
    parent_idx: list[int | None],
    children: list[list[int]],
    soma_indices: set[int],
    label_set: set[int],
    max_island_size: int = 15,
    confidence_threshold: float = 0.75,
) -> list[int]:
    """Flip small isolated label islands to match their surroundings.

    An island is a contiguous group of nodes with the same label,
    surrounded by a different label. If the island is small and
    low-confidence, flip it.
    """
    neurite_labels = label_set - {1}
    if len(neurite_labels) <= 1:
        return labels

    out = list(labels)
    n = len(nodes)
    visited = [False] * n

    for i in range(n):
        if visited[i] or i in soma_indices or out[i] == 1:
            continue

        # BFS to find contiguous region with same label
        lbl = out[i]
        region: list[int] = []
        queue = [i]
        neighbors_outside: list[int] = []

        while queue:
            idx = queue.pop(0)
            if visited[idx]:
                continue
            if idx in soma_indices or out[idx] != lbl:
                if out[idx] != lbl and idx not in soma_indices:
                    neighbors_outside.append(idx)
                continue
            visited[idx] = True
            region.append(idx)
            # Add parent and children to explore
            pidx = parent_idx[idx]
            if pidx is not None:
                queue.append(pidx)
            queue.extend(children[idx])

        if len(region) > max_island_size or not neighbors_outside:
            continue

        # Check if average confidence is low
        avg_conf = sum(confidences[idx] for idx in region) / len(region)
        if avg_conf >= confidence_threshold:
            continue

        # Flip to the most common neighbor label
        neighbor_labels = [out[idx] for idx in neighbors_outside if out[idx] in neurite_labels]
        if not neighbor_labels:
            continue
        counts = Counter(neighbor_labels)
        flip_to = counts.most_common(1)[0][0]
        if flip_to == lbl:
            continue

        for idx in region:
            out[idx] = flip_to

    return out


def _constrain_apical_to_owner_subtree(
    labels: list[int],
    confidences: list[float],
    children: list[list[int]],
    soma_indices: set[int],
    apical_owner_root: int | None,
    apical_label: int = 4,
    fallback_label: int = 3,
    preserve_confidence: float = 0.93,
) -> list[int]:
    """Mostly constrain apical labels to the preferred apical subtree.

    Nodes outside the preferred subtree keep apical only if the branch
    classifier was extremely confident. This keeps the constraint soft
    enough to avoid obvious false negatives while still cleaning up most
    stray apical islands in basal subtrees.
    """
    if apical_owner_root is None:
        return labels

    owner_nodes = set(_subtree_indices(apical_owner_root, children))
    out = list(labels)
    for idx, lbl in enumerate(labels):
        if idx in soma_indices:
            continue
        if lbl == apical_label and idx not in owner_nodes and confidences[idx] < preserve_confidence:
            out[idx] = fallback_label
    return out


def _strip_spurious_soma(
    labels: list[int],
    soma_indices: set[int],
    neurite_labels: list[int],
) -> list[int]:
    """Guard: no non-soma node may carry label 1.

    Any label-1 on a non-soma node is remapped to the first neurite label
    (or 3 as a final fallback). Prevents Stage 3 from ever emitting a
    false-positive soma prediction.
    """
    out = list(labels)
    fallback = neurite_labels[0] if neurite_labels else 3
    for i in range(len(out)):
        if i not in soma_indices and out[i] == 1:
            out[i] = fallback
    return out


# ---------------------------------------------------------------------------
# Main refinement function
# ---------------------------------------------------------------------------

def refine(
    nodes: list[SWCNode],
    ml_labels: list[int],
    ml_confidences: list[float],
    cell_type_result: CellTypeResult,
    apical_owner_root: int | None = None,
) -> RefinementResult:
    """Apply Stage 3 topology-aware refinement.

    Args:
        nodes: parsed SWC nodes
        ml_labels: per-node labels from Stage 2 (1=soma, 2=axon, 3=basal, 4=apical)
        ml_confidences: per-node confidence from Stage 2 ML model
        cell_type_result: output from Stage 1

    Returns:
        RefinementResult with refined labels.
    """
    n = len(nodes)
    if n == 0:
        return RefinementResult([], 0, {}, cell_type_result.cell_type)

    id_to_idx, parent_idx, children, roots = _build_tree(nodes)
    label_set = cell_type_result.label_set
    cell_type = cell_type_result.cell_type

    # Label-free proxy soma/root anchor
    soma_indices = {_select_proxy_root(nodes, children, roots)}

    # Start with ML labels, set soma nodes to 1
    labels = list(ml_labels)
    for si in soma_indices:
        labels[si] = 1

    # Mask out labels not in the label set
    neurite_labels = sorted(label_set - {1})
    for i in range(n):
        if labels[i] != 1 and labels[i] not in neurite_labels:
            # Map to closest valid label
            if neurite_labels:
                labels[i] = neurite_labels[0]  # default to first neurite type

    subtree_labels: dict[int, int] = {}

    # GUARD (item 2): If the label set has ≤1 neurite class (e.g. Purkinje
    # with {1, 3}), refinement cannot do anything meaningful — there is no
    # class to flip between. Return early to disconnect Stage 3 behavior
    # from any Stage 1 misclassification that might route this morphology
    # through an aggressive refinement branch.
    if len(neurite_labels) <= 1:
        # Still enforce the non-soma-label-1 guard (item 1) for safety
        labels = _strip_spurious_soma(labels, soma_indices, neurite_labels)
        refined_labels_early = [
            RefinedLabel(
                node_id=nodes[i].id,
                label=labels[i],
                confidence=ml_confidences[i],
                was_refined=labels[i] != ml_labels[i],
            )
            for i in range(n)
        ]
        return RefinementResult(
            labels=refined_labels_early,
            n_refined=sum(1 for i in range(n) if labels[i] != ml_labels[i]),
            subtree_labels={},
            cell_type=cell_type,
        )

    if cell_type == "purkinje":
        # Purkinje: let the model's predictions pass through with gentle
        # smoothing only. Previously this branch hard-coded every non-soma
        # node to dendrite (type 3), which bypassed the model entirely and
        # destroyed any axon labels that might come from real data. Now
        # that Stage 2 has a per-cell-type model, we trust its output. If
        # only one neurite class is in label_set, _parent_child_smoothing
        # short-circuits, which is correct.
        labels = _parent_child_smoothing(
            nodes, labels, ml_confidences, parent_idx, children,
            soma_indices, label_set,
            iterations=2, confidence_threshold=0.60,
        )
        n_refined = sum(1 for i in range(n) if labels[i] != ml_labels[i])

    elif cell_type == "interneuron":
        # Interneuron: NO subtree voting — axon can emerge from within
        # dendrite subtrees, so forcing whole subtrees to one label hurts.
        # Only gentle smoothing and small island flipping.
        labels = _parent_child_smoothing(
            nodes, labels, ml_confidences, parent_idx, children,
            soma_indices, label_set,
            iterations=2, parent_weight=0.15, child_weight=0.10,
            confidence_threshold=0.55,
        )
        labels = _island_flipping(
            nodes, labels, ml_confidences, parent_idx, children,
            soma_indices, label_set,
            max_island_size=8, confidence_threshold=0.60,
        )
        n_refined = sum(1 for i in range(n) if labels[i] != ml_labels[i])

    elif cell_type == "pyramidal":
        # Pyramidal: full refinement pipeline
        # 1. Primary subtree voting
        labels, subtree_labels = _primary_subtree_voting(
            nodes, labels, ml_confidences, parent_idx, children,
            soma_indices, label_set,
        )

        # 2. Single-axon constraint
        if 2 in label_set:
            subtree_labels, labels = _enforce_single_class(
                subtree_labels, nodes, children, labels, ml_confidences,
                target_class=2, fallback_class=3,
            )

        # 3. Single-apical constraint
        if 4 in label_set:
            subtree_labels, labels = _enforce_single_class(
                subtree_labels, nodes, children, labels, ml_confidences,
                target_class=4, fallback_class=3,
            )

        labels = _constrain_apical_to_owner_subtree(
            labels,
            ml_confidences,
            children,
            soma_indices,
            apical_owner_root,
        )

        # 4. Parent-child smoothing
        labels = _parent_child_smoothing(
            nodes, labels, ml_confidences, parent_idx, children,
            soma_indices, label_set,
            iterations=3, confidence_threshold=0.75,
        )

        # 5. Island flipping
        labels = _island_flipping(
            nodes, labels, ml_confidences, parent_idx, children,
            soma_indices, label_set,
            max_island_size=15, confidence_threshold=0.70,
        )

        labels = _constrain_apical_to_owner_subtree(
            labels,
            ml_confidences,
            children,
            soma_indices,
            apical_owner_root,
        )

        n_refined = sum(1 for i in range(n) if labels[i] != ml_labels[i])

    else:
        # Other/unknown: basic smoothing only
        labels = _parent_child_smoothing(
            nodes, labels, ml_confidences, parent_idx, children,
            soma_indices, label_set,
            iterations=2, confidence_threshold=0.60,
        )
        n_refined = sum(1 for i in range(n) if labels[i] != ml_labels[i])

    labels = _strip_spurious_soma(labels, soma_indices, neurite_labels)

    # Build result
    refined_labels = []
    for i in range(n):
        refined_labels.append(RefinedLabel(
            node_id=nodes[i].id,
            label=labels[i],
            confidence=ml_confidences[i],
            was_refined=labels[i] != ml_labels[i],
        ))

    return RefinementResult(
        labels=refined_labels,
        n_refined=n_refined,
        subtree_labels=subtree_labels,
        cell_type=cell_type,
    )
