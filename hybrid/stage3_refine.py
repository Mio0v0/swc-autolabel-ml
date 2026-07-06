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


def _pick_apical_by_principal_axis(
    subtree_labels: dict[int, int],
    nodes: list[SWCNode],
    children: list[list[int]],
    soma_indices: set[int],
    labels: list[int],
    apical_label: int = 4,
    fallback_label: int = 3,
) -> tuple[dict[int, int], list[int]]:
    """Enforce ≤1 apical primary subtree, choosing the winner by the cell's
    principal axis (PC1 of all neurite nodes) rather than by confidence sum.

    Rationale: in cells with rotated coordinate frames or sideways apicals,
    the model can confidently mislabel the wrong subtree as apical. PC1
    captures the cell's actual elongation direction, so the subtree whose
    mean position projects highest along PC1 is the true apical trunk.

    Falls back to a no-op if PC1 is degenerate or only one apical subtree
    exists.
    """
    out_labels = list(labels)
    out_subtrees = dict(subtree_labels)

    apical_owners = [pr for pr, lbl in subtree_labels.items() if lbl == apical_label]
    if len(apical_owners) <= 1:
        return out_subtrees, out_labels

    # Soma center (proxy)
    if soma_indices:
        sx = float(np.mean([nodes[i].x for i in soma_indices]))
        sy = float(np.mean([nodes[i].y for i in soma_indices]))
        sz = float(np.mean([nodes[i].z for i in soma_indices]))
    else:
        sx, sy, sz = nodes[0].x, nodes[0].y, nodes[0].z

    # PC1 of neurite cloud, signed away from soma
    n = len(nodes)
    neurite_idx = [i for i in range(n) if i not in soma_indices]
    if len(neurite_idx) < 5:
        return out_subtrees, out_labels
    pts = np.array(
        [[nodes[i].x - sx, nodes[i].y - sy, nodes[i].z - sz] for i in neurite_idx],
        dtype=np.float64,
    )
    try:
        cov = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order_eig = np.argsort(eigvals)[::-1]
        pc1 = eigvecs[:, order_eig[0]]
        if float(np.dot(pc1, pts.mean(axis=0))) < 0.0:
            pc1 = -pc1
    except np.linalg.LinAlgError:
        return out_subtrees, out_labels

    # Score each apical-labeled primary by mean PC1 projection of its subtree
    best_pr = None
    best_score = -float("inf")
    for pr in apical_owners:
        sub = _subtree_indices(pr, children)
        if not sub:
            continue
        projs = [
            (nodes[i].x - sx) * pc1[0]
            + (nodes[i].y - sy) * pc1[1]
            + (nodes[i].z - sz) * pc1[2]
            for i in sub
        ]
        score = float(np.mean(projs))
        if score > best_score:
            best_score = score
            best_pr = pr

    if best_pr is None:
        return out_subtrees, out_labels

    # Reassign losers to fallback (basal)
    for pr in apical_owners:
        if pr == best_pr:
            continue
        out_subtrees[pr] = fallback_label
        sub = _subtree_indices(pr, children)
        for idx in sub:
            if out_labels[idx] == apical_label:
                out_labels[idx] = fallback_label

    return out_subtrees, out_labels


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

    Override `max_island_size` via env var `SWCAL_MAX_ISLAND_SIZE`
    (used by the P10 iteration to test more aggressive cleanup).
    """
    import os as _os
    env_size = _os.environ.get("SWCAL_MAX_ISLAND_SIZE")
    if env_size is not None:
        try:
            max_island_size = int(env_size)
        except ValueError:
            pass
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


def _soft_subtree_majority(
    nodes: list[SWCNode],
    labels: list[int],
    confidences: list[float],
    children: list[list[int]],
    soma_indices: set[int],
    label_set: set[int],
    high_conf: float | None = None,
    low_conf: float | None = None,
    min_margin: float | None = None,
) -> list[int]:
    """Soft subtree-majority propagation (interneuron-safe).

    Unlike pyramidal "hard" subtree voting (which forces every node in a
    primary subtree to one label), interneurons allow axons to emerge from
    within dendritic subtrees — so we cannot simply majority-vote. Instead:

    1. For each primary subtree, compute confidence-weighted votes using
       ONLY nodes with ml_confidence ≥ high_conf.
    2. If the winning label owns ≥ min_margin of the high-confidence mass,
       propagate it ONLY to low-confidence (< low_conf) nodes in the subtree.
    3. High-confidence nodes are never touched.

    This is a conservative "fill in the unsure parts" operation.

    Thresholds default to (high_conf=0.70, low_conf=0.55, min_margin=0.60) but
    can be overridden via SWCAL_SOFT_HIGH_CONF / SWCAL_SOFT_LOW_CONF /
    SWCAL_SOFT_MIN_MARGIN env vars for boundary-smoothing parameter sweeps.
    """
    import os as _os
    if high_conf is None:
        high_conf = float(_os.environ.get("SWCAL_SOFT_HIGH_CONF", 0.70))
    if low_conf is None:
        low_conf = float(_os.environ.get("SWCAL_SOFT_LOW_CONF", 0.55))
    if min_margin is None:
        min_margin = float(_os.environ.get("SWCAL_SOFT_MIN_MARGIN", 0.60))

    neurite_labels = sorted(label_set - {1})
    if len(neurite_labels) <= 1:
        return labels

    out = list(labels)
    primary_roots: list[int] = []
    for si in soma_indices:
        for ci in children[si]:
            if ci not in soma_indices:
                primary_roots.append(ci)

    for pr in primary_roots:
        subtree = _subtree_indices(pr, children)
        if len(subtree) < 4:
            continue  # too small to vote

        # Confidence-weighted votes over HIGH-confidence nodes only
        hi_votes: dict[int, float] = {lbl: 0.0 for lbl in neurite_labels}
        hi_total = 0.0
        for idx in subtree:
            if confidences[idx] < high_conf:
                continue
            lbl = out[idx]
            if lbl in hi_votes:
                hi_votes[lbl] += confidences[idx]
                hi_total += confidences[idx]

        if hi_total <= 0:
            continue

        winner = max(neurite_labels, key=lambda lbl: hi_votes.get(lbl, 0.0))
        winner_share = hi_votes[winner] / hi_total

        if winner_share < min_margin:
            continue  # subtree is genuinely mixed — leave it alone

        # Flip low-confidence nodes in this subtree to the winner
        for idx in subtree:
            if confidences[idx] < low_conf and out[idx] != winner and out[idx] in neurite_labels:
                out[idx] = winner

    return out


def _interneuron_axon_bias(
    nodes: list[SWCNode],
    labels: list[int],
    confidences: list[float],
    children: list[list[int]],
    soma_indices: set[int],
    label_set: set[int],
    low_conf: float = 0.60,
    min_subtree_nodes: int = 20,
    thin_radius_percentile: float = 0.25,
) -> list[int]:
    """Bias the thinnest-longest primary subtree toward axon (interneuron).

    Interneuron axons are characterized by a distinctive geometric signature:
    thin (small radius) and long (high path length). This rule identifies
    the primary subtree most axon-like by that signature and gently flips
    low-confidence non-axon nodes in it toward axon (2).

    It does NOT touch high-confidence nodes, so if the ML model is certain
    a thin branch is dendritic, the rule defers.
    """
    if 2 not in label_set:
        return labels

    out = list(labels)
    primary_roots: list[int] = []
    for si in soma_indices:
        for ci in children[si]:
            if ci not in soma_indices:
                primary_roots.append(ci)

    if len(primary_roots) < 2:
        return out  # nothing to compare against

    # Score each primary by an axon-likeness metric: low mean radius + high
    # max path length. Normalize both within this cell so the rule is
    # scale-invariant.
    primary_stats: list[tuple[int, float, float, int]] = []  # (pr, mean_r, max_path, size)
    for pr in primary_roots:
        subtree = _subtree_indices(pr, children)
        if len(subtree) < min_subtree_nodes:
            continue
        radii = [nodes[i].radius for i in subtree]
        if not radii:
            continue
        mean_r = float(np.mean(radii))
        # Path length from pr to the deepest node
        dist = _path_length_from(pr, nodes, [None] * len(nodes), children)
        max_path = max(dist.values()) if dist else 0.0
        primary_stats.append((pr, mean_r, max_path, len(subtree)))

    if len(primary_stats) < 2:
        return out

    min_r = min(s[1] for s in primary_stats)
    max_r = max(s[1] for s in primary_stats)
    max_path = max(s[2] for s in primary_stats)
    if max_r - min_r < 1e-9 or max_path < 1e-9:
        return out

    # axon-score = 0.6 * (1 - normalized_radius) + 0.4 * normalized_path
    best_pr = None
    best_score = -1.0
    for pr, mean_r, path, _size in primary_stats:
        r_score = 1.0 - (mean_r - min_r) / (max_r - min_r)
        p_score = path / max_path
        score = 0.6 * r_score + 0.4 * p_score
        if score > best_score:
            best_score = score
            best_pr = pr

    # Only act if the candidate is notably thinner than average — otherwise
    # there isn't a clear axon-like primary to bias toward.
    avg_r = float(np.mean([s[1] for s in primary_stats]))
    best_mean_r = next(s[1] for s in primary_stats if s[0] == best_pr)
    if best_mean_r > avg_r * (1.0 - thin_radius_percentile):
        return out  # not distinctly thin → don't force

    # Bias low-confidence non-axon nodes in the winning subtree toward axon
    axon_subtree = _subtree_indices(best_pr, children)
    for idx in axon_subtree:
        if idx in soma_indices:
            continue
        if out[idx] != 2 and confidences[idx] < low_conf and out[idx] != 1:
            out[idx] = 2

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


def _branch_neighbor_smoothing(
    nodes: list[SWCNode],
    labels: list[int],
    parent_idx: list[int | None],
    children: list[list[int]],
    soma_indices: set[int],
    label_set: set[int],
) -> list[int]:
    """Post-process: flip a branch's label if its parent branch AND at least
    one child branch both predict a DIFFERENT class than this branch.

    A "branch" here is a contiguous run of nodes with the same label between
    bifurcations. The function uses Stage 3's already-segmented branches via
    a simple node-level scan + parent/child references.

    Targets isolated mislabeled branches that survived earlier smoothing
    (e.g., a basal branch sandwiched between two apical branches).

    Env-gated by SWCAL_BRANCH_NEIGHBOR_SMOOTH=1 (off by default).
    """
    import os as _os
    if _os.environ.get("SWCAL_BRANCH_NEIGHBOR_SMOOTH") != "1":
        return labels

    out = list(labels)
    neurite_labels = label_set - {1}
    if len(neurite_labels) <= 1:
        return out

    # Group nodes into "branch segments" — contiguous run with same label
    # between bifurcations.
    n = len(nodes)
    branch_id = [-1] * n
    cur_id = 0
    for i in range(n):
        if i in soma_indices:
            continue
        p = parent_idx[i] if isinstance(parent_idx[i], int) else None
        # New branch if parent is soma, parent has >1 children, or parent has
        # a different label than this node
        if (p is None or p in soma_indices
                or len(children[p]) > 1
                or out[i] != out[p]):
            branch_id[i] = cur_id
            cur_id += 1
        else:
            branch_id[i] = branch_id[p]

    # For each branch, find its parent branch + child branches
    branch_label: dict[int, int] = {}
    branch_parent: dict[int, int | None] = {}
    branch_children: dict[int, list[int]] = {}
    branch_nodes: dict[int, list[int]] = {}
    for i in range(n):
        b = branch_id[i]
        if b < 0:
            continue
        branch_label[b] = out[i]
        branch_nodes.setdefault(b, []).append(i)
        p = parent_idx[i]
        if isinstance(p, int) and p >= 0 and p not in soma_indices:
            pb = branch_id[p]
            if pb >= 0 and pb != b:
                branch_parent.setdefault(b, pb)
                branch_children.setdefault(pb, []).append(b)

    # Now flip branches whose parent + at least one child agree on a
    # different label
    flipped = 0
    for b, lbl in branch_label.items():
        pb = branch_parent.get(b)
        if pb is None:
            continue
        parent_lbl = branch_label.get(pb)
        if parent_lbl is None or parent_lbl == lbl:
            continue
        # Need at least one child branch agreeing with parent (not this branch)
        kids = branch_children.get(b, [])
        if not any(branch_label.get(k) == parent_lbl for k in kids):
            continue
        # Both parent + a child agree on a different label → flip this branch
        for idx in branch_nodes[b]:
            out[idx] = parent_lbl
            flipped += 1

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
        # Interneuron: no HARD subtree voting (axon can emerge from within
        # dendrite subtrees), but soft rules help close the Stage-3 gap.
        #
        # Rule 1 — thin-primary axon bias: identify the primary subtree
        # most likely to be the axon (thinnest mean radius + longest path)
        # and gently flip low-confidence non-axon nodes in it toward axon.
        labels = _interneuron_axon_bias(
            nodes, labels, ml_confidences, children,
            soma_indices, label_set,
            low_conf=0.60,
        )

        # Rule 2 — soft subtree majority: for each primary subtree, if
        # high-confidence nodes strongly agree on one label, propagate that
        # label only to LOW-confidence nodes in the same subtree. High
        # confidence predictions are preserved.
        labels = _soft_subtree_majority(
            nodes, labels, ml_confidences, children,
            soma_indices, label_set,
            high_conf=0.70, low_conf=0.55, min_margin=0.60,
        )

        # Rule 3 — gentle parent-child smoothing and small-island flipping
        # (unchanged).
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

        # 3. Single-apical constraint — PCA-ranked picker first (uses cell's
        # own principal axis, robust to rotated coordinate frames), then the
        # confidence-based enforcer as a safety net.
        if 4 in label_set:
            subtree_labels, labels = _pick_apical_by_principal_axis(
                subtree_labels, nodes, children, soma_indices, labels,
                apical_label=4, fallback_label=3,
            )
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

    # Optional: branch-level neighbor smoothing (env-gated; off by default).
    # Targets isolated mislabeled branches between consistently-labeled
    # parent and child branches.
    labels = _branch_neighbor_smoothing(
        nodes, labels, parent_idx, children, soma_indices, label_set,
    )
    n_refined = sum(1 for i in range(n) if labels[i] != ml_labels[i])

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
