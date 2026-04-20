"""Branch-level feature extraction for Stage 2.

Extracts per-branch geometric and topological features from an SWC
morphology. These features are label-agnostic at inference time: they
do not use the input SWC type column to locate soma or decide branch
structure. A proxy root anchor is inferred from topology and geometry.

Each branch segment is a linear chain of nodes between bifurcation
points (or between a bifurcation and a terminal).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import SWCNode, parse_swc

# ---------------------------------------------------------------------------
# Branch feature names (label-free — no leakage)
# ---------------------------------------------------------------------------

BRANCH_FEATURE_NAMES: list[str] = [
    # Branch geometry
    "path_length",              # total path length of this branch
    "radial_extent",            # max Euclidean distance from anchor
    "n_nodes",                  # number of nodes in this branch
    "mean_radius",              # mean node radius
    "std_radius",               # std of node radii
    "min_radius",               # minimum radius
    "max_radius",               # maximum radius
    "taper_ratio",              # terminal/proximal radius ratio
    "persistence",              # Euclidean/path ratio (straightness)
    "up_alignment",             # z-direction alignment (0=down, 1=up)
    "branchiness",              # fraction of bifurcation nodes
    "symmetry",                 # child radius symmetry at anchor

    # Position relative to soma
    "root_path_dist",           # path distance from soma to branch midpoint
    "root_radial_dist",         # Euclidean distance from soma to branch midpoint
    "z_rel_soma",               # mean z relative to soma center
    "z_span",                   # z range within this branch

    # Subtree context
    "subtree_size",             # total nodes in the subtree rooted at this branch
    "subtree_depth",            # max depth from this branch to leaf
    "subtree_max_path",         # max path length in subtree
    "branch_order",             # bifurcation count from soma to this branch

    # Parent/sibling context
    "parent_radius",            # radius at the anchor (parent) node
    "radius_ratio_to_parent",   # mean_radius / parent_radius
    "n_siblings",               # number of sibling branches at the anchor
    "is_primary",               # 1.0 if direct child of soma, else 0.0

    # Cell-level normalized features (relative to cell stats)
    "path_length_rel",          # path_length / max_path_in_cell
    "radial_extent_rel",        # radial_extent / max_radial_in_cell
    "radius_rel",               # mean_radius / cell_mean_radius
    "subtree_size_rel",         # subtree_size / total_nodes

    # Cell type encoding (from Stage 1)
    "is_pyramidal",             # 1.0 if pyramidal
    "is_interneuron",           # 1.0 if interneuron
    "is_purkinje",              # 1.0 if purkinje

    # Primary-subtree rank features (help apical vs basal: apical is the
    # single dominant z-aligned / longest subtree)
    "is_longest_subtree",       # 1.0 if this branch's primary subtree is the longest in the cell
    "subtree_path_ratio",       # this subtree's max path / max over all primary subtrees
    "subtree_z_ratio",          # this subtree's z_span / max over all primary subtrees
    "subtree_z_rank",           # normalized rank by z_span among primary subtrees (1.0=max)
    "subtree_max_radial_rank",  # normalized rank by max soma radial reach among primary subtrees

    # Axon initial segment / thin-long signatures (help axon vs dendrite
    # for interneurons where axons emerge abruptly and remain thin)
    "starts_at_soma",           # 1.0 if anchor node is a soma node
    "radius_drop_at_anchor",    # (anchor_r - first_nodes_mean_r) / anchor_r
    "thin_fraction",            # fraction of branch nodes with radius < 0.5 µm
    "min_radius_in_subtree",    # min radius across all descendants (thin chain signature)

    # Multi-scale local radius / straightness
    "proximal_mean_radius",     # mean radius over first 1/3 of branch
    "distal_mean_radius",       # mean radius over last 1/3 of branch
    "proximal_persistence",     # straightness of first 1/3
    "distal_persistence",       # straightness of last 1/3
]


@dataclass
class BranchData:
    """Per-branch data: features + ground truth label."""
    branch_id: int
    node_indices: list[int]     # indices into the node list
    anchor_idx: int             # parent/anchor node index
    features: np.ndarray        # feature vector
    gt_label: int               # ground truth majority label (from SWC type column)
    gt_label_counts: dict[int, int]  # per-class node counts
    n_nodes: int
    primary_root_idx: int | None


@dataclass
class MorphologyBranches:
    """All branches extracted from one morphology."""
    file_path: str
    cell_type: str              # from Stage 1 or ground truth folder
    branches: list[BranchData]
    soma_indices: list[int]     # proxy soma/root anchor indices
    total_nodes: int


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------

def _build_tree(nodes: list[SWCNode]) -> tuple[
    dict[int, int],         # id -> index
    list[int | None],       # parent_idx per node
    list[list[int]],        # children per node
    list[int],              # root indices
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


def _bfs_order(roots: list[int], children: list[list[int]], n: int) -> list[int]:
    order: list[int] = []
    visited: set[int] = set()
    queue = list(roots)
    while queue:
        idx = queue.pop(0)
        if idx in visited:
            continue
        visited.add(idx)
        order.append(idx)
        queue.extend(sorted(children[idx]))
    return order


def _euclidean(a: SWCNode, b: SWCNode) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _select_proxy_root(
    nodes: list[SWCNode],
    children: list[list[int]],
    roots: list[int],
) -> int:
    """Pick a label-free root anchor that approximates the soma location."""
    if not nodes:
        return 0
    candidate_roots = roots or [0]
    return max(
        candidate_roots,
        key=lambda idx: (nodes[idx].radius, len(children[idx]), -idx),
    )


# ---------------------------------------------------------------------------
# Branch partitioning
# ---------------------------------------------------------------------------

def _partition_branches(
    nodes: list[SWCNode],
    parent_idx: list[int | None],
    children: list[list[int]],
    soma_indices: set[int],
    max_chunk_path: float = 180.0,
) -> list[tuple[int, list[int]]]:
    """Partition morphology into (anchor_idx, [node_indices]) branch segments.

    A branch is a linear chain from a bifurcation/root to the next
    bifurcation or terminal. Long segments are chunked.
    """
    n = len(nodes)
    assigned: list[bool] = [False] * n
    branches: list[tuple[int, list[int]]] = []

    # Find starting points: soma children
    starts: list[tuple[int, int]] = []  # (anchor, start_node)
    for si in sorted(soma_indices):
        assigned[si] = True
        for ci in sorted(children[si]):
            if ci not in soma_indices:
                starts.append((si, ci))

    # If no soma, use roots
    if not starts:
        roots = [i for i, p in enumerate(parent_idx) if p is None]
        for ri in roots:
            for ci in sorted(children[ri]):
                starts.append((ri, ci))
            if not children[ri]:
                starts.append((ri, ri))

    pending = list(starts)
    while pending:
        anchor, start = pending.pop(0)
        if assigned[start]:
            continue

        # Trace linear segment
        segment: list[int] = []
        cur = start
        path_len = 0.0
        while not assigned[cur]:
            assigned[cur] = True
            segment.append(cur)

            # Track path length for chunking
            if len(segment) > 1:
                path_len += _euclidean(nodes[segment[-2]], nodes[cur])
                if path_len > max_chunk_path:
                    break

            kids = [c for c in children[cur] if not assigned[c]]
            if len(kids) == 1:
                cur = kids[0]
            else:
                # Terminal or bifurcation — end this segment
                for ci in sorted(kids):
                    pending.append((cur, ci))
                break

        if segment:
            branches.append((anchor, segment))

    return branches


# ---------------------------------------------------------------------------
# Per-branch feature computation
# ---------------------------------------------------------------------------

def _subtree_stats(
    root: int,
    children: list[list[int]],
    nodes: list[SWCNode],
    parent_idx: list[int | None],
) -> tuple[int, int, float]:
    """Compute (subtree_size, subtree_depth, subtree_max_path)."""
    stack: list[tuple[int, int, float]] = [(root, 0, 0.0)]
    size = 0
    max_depth = 0
    max_path = 0.0
    while stack:
        idx, depth, path = stack.pop()
        size += 1
        max_depth = max(max_depth, depth)
        max_path = max(max_path, path)
        for ci in children[idx]:
            seg_len = _euclidean(nodes[idx], nodes[ci])
            stack.append((ci, depth + 1, path + seg_len))
    return size, max_depth, max_path


def extract_branches(
    nodes: list[SWCNode],
    cell_type: str,
    file_path: str = "",
) -> MorphologyBranches:
    """Extract all branch segments and their features from a morphology.

    Args:
        nodes: parsed SWC nodes
        cell_type: cell type string (from Stage 1 or ground truth)
        file_path: source file path (for tracking)

    Returns:
        MorphologyBranches with per-branch features and labels.
    """
    if not nodes:
        return MorphologyBranches(file_path, cell_type, [], [], 0)

    id_to_idx, parent_idx, children, roots = _build_tree(nodes)
    n = len(nodes)
    order = _bfs_order(roots, children, n)

    # Label-free proxy soma/root anchor
    proxy_root = _select_proxy_root(nodes, children, roots)
    soma_indices = {proxy_root}

    # Soma center
    if soma_indices:
        soma_x = np.mean([nodes[i].x for i in soma_indices])
        soma_y = np.mean([nodes[i].y for i in soma_indices])
        soma_z = np.mean([nodes[i].z for i in soma_indices])
    else:
        soma_x = nodes[0].x
        soma_y = nodes[0].y
        soma_z = nodes[0].z

    # Path from root for all nodes
    path_from_root = [0.0] * n
    branch_order_arr = [0] * n
    for i in order:
        pidx = parent_idx[i]
        if pidx is not None:
            path_from_root[i] = path_from_root[pidx] + _euclidean(nodes[pidx], nodes[i])
            branch_order_arr[i] = branch_order_arr[pidx] + (1 if len(children[pidx]) > 1 else 0)

    # Cell-level stats for normalization
    all_radii = [nd.radius for i, nd in enumerate(nodes) if i not in soma_indices]
    if not all_radii:
        all_radii = [nd.radius for nd in nodes]
    cell_mean_radius = float(np.mean(all_radii)) if all_radii else 1.0
    max_path_in_cell = max(path_from_root) if path_from_root else 1.0
    max_radial_in_cell = max(
        math.sqrt((nd.x - soma_x)**2 + (nd.y - soma_y)**2 + (nd.z - soma_z)**2)
        for nd in nodes
    ) if nodes else 1.0

    # Primary children of soma (for is_primary feature)
    primary_set: set[int] = set()
    for si in soma_indices:
        for ci in children[si]:
            if ci not in soma_indices:
                primary_set.add(ci)

    # Cell type encoding
    is_pyramidal = 1.0 if cell_type == "pyramidal" else 0.0
    is_interneuron = 1.0 if cell_type == "interneuron" else 0.0
    is_purkinje = 1.0 if cell_type == "purkinje" else 0.0

    # --- Primary-subtree stats (for rank features: apical vs basal) ---
    # For each primary subtree root, compute max path, z_span, min_radius,
    # and a per-node map telling which primary root each node belongs to.
    primary_root_of = [-1] * n
    subtree_max_path_by_root: dict[int, float] = {}
    subtree_z_span_by_root: dict[int, float] = {}
    subtree_min_radius_by_root: dict[int, float] = {}
    subtree_max_radial_by_root: dict[int, float] = {}
    for pr in primary_set:
        stack = [pr]
        sub_nodes: list[int] = []
        while stack:
            idx = stack.pop()
            sub_nodes.append(idx)
            primary_root_of[idx] = pr
            for ci in children[idx]:
                stack.append(ci)
        if not sub_nodes:
            continue
        subtree_max_path_by_root[pr] = max(path_from_root[i] for i in sub_nodes)
        z_vals = [nodes[i].z for i in sub_nodes]
        subtree_z_span_by_root[pr] = (max(z_vals) - min(z_vals)) if z_vals else 0.0
        subtree_min_radius_by_root[pr] = min(nodes[i].radius for i in sub_nodes)
        subtree_max_radial_by_root[pr] = max(
            math.sqrt(
                (nodes[i].x - soma_x) ** 2 +
                (nodes[i].y - soma_y) ** 2 +
                (nodes[i].z - soma_z) ** 2
            )
            for i in sub_nodes
        )

    max_subtree_path_across = max(subtree_max_path_by_root.values(), default=1.0)
    max_subtree_z_across = max(subtree_z_span_by_root.values(), default=1.0)

    # z-span rank: sort primary roots by z_span, assign normalized rank (1.0 = largest)
    z_rank_of_root: dict[int, float] = {}
    sorted_roots_by_z = sorted(
        subtree_z_span_by_root.items(), key=lambda kv: kv[1], reverse=True
    )
    n_primary = len(sorted_roots_by_z)
    for rank, (pr, _) in enumerate(sorted_roots_by_z):
        # rank 0 is largest → 1.0; last → ~0
        z_rank_of_root[pr] = 1.0 - (rank / max(1, n_primary - 1)) if n_primary > 1 else 1.0

    radial_rank_of_root: dict[int, float] = {}
    sorted_roots_by_radial = sorted(
        subtree_max_radial_by_root.items(), key=lambda kv: kv[1], reverse=True
    )
    n_primary_radial = len(sorted_roots_by_radial)
    for rank, (pr, _) in enumerate(sorted_roots_by_radial):
        radial_rank_of_root[pr] = 1.0 - (rank / max(1, n_primary_radial - 1)) if n_primary_radial > 1 else 1.0

    # Partition into branches
    raw_branches = _partition_branches(nodes, parent_idx, children, soma_indices)

    branches: list[BranchData] = []
    for bid, (anchor, segment) in enumerate(raw_branches):
        if not segment:
            continue

        seg_nodes = [nodes[i] for i in segment]
        anchor_node = nodes[anchor]

        # --- Branch geometry ---
        path_length = 0.0
        for k in range(1, len(segment)):
            path_length += _euclidean(nodes[segment[k - 1]], nodes[segment[k]])

        radial_extent = max(
            _euclidean(anchor_node, nodes[i]) for i in segment
        ) if segment else 0.0

        radii = [nd.radius for nd in seg_nodes]
        mean_rad = float(np.mean(radii))
        std_rad = float(np.std(radii)) if len(radii) > 1 else 0.0
        min_rad = float(min(radii))
        max_rad = float(max(radii))

        # Taper ratio
        win = min(3, len(radii))
        prox_r = float(np.mean(radii[:win])) if radii else 1.0
        dist_r = float(np.mean(radii[-win:])) if radii else 1.0
        taper = dist_r / prox_r if prox_r > 1e-9 else 1.0

        # Persistence (straightness)
        euclid_dist = _euclidean(nodes[segment[0]], nodes[segment[-1]]) if len(segment) > 1 else 0.0
        persistence = euclid_dist / path_length if path_length > 1e-9 else 0.5

        # Up alignment
        if len(segment) > 1:
            dz = nodes[segment[-1]].z - nodes[segment[0]].z
            dist_3d = euclid_dist
            up_align = (dz / dist_3d + 1.0) * 0.5 if dist_3d > 1e-9 else 0.5
            up_align = max(0.0, min(1.0, up_align))
        else:
            up_align = 0.5

        # Branchiness
        bif_count = sum(1 for i in segment if len(children[i]) > 1)
        branchiness = bif_count / max(1, len(segment))

        # Symmetry at anchor
        anchor_kids = children[anchor]
        if len(anchor_kids) > 1:
            kid_radii = [nodes[ci].radius for ci in anchor_kids]
            med = float(np.median(kid_radii))
            if med > 1e-9:
                mad = float(np.mean([abs(r - med) for r in kid_radii]))
                symmetry = 1.0 - min(1.0, mad / med)
            else:
                symmetry = 0.5
        else:
            symmetry = 0.5

        # --- Position relative to soma ---
        mid_idx = segment[len(segment) // 2]
        root_path_dist = path_from_root[mid_idx]
        root_radial_dist = math.sqrt(
            (nodes[mid_idx].x - soma_x)**2 +
            (nodes[mid_idx].y - soma_y)**2 +
            (nodes[mid_idx].z - soma_z)**2
        )
        z_rel = float(np.mean([nd.z - soma_z for nd in seg_nodes]))
        z_sp = max(nd.z for nd in seg_nodes) - min(nd.z for nd in seg_nodes) if seg_nodes else 0.0

        # --- Subtree context ---
        sub_size, sub_depth, sub_max_path = _subtree_stats(
            segment[0], children, nodes, parent_idx
        )
        b_order = branch_order_arr[segment[0]]

        # --- Parent/sibling context ---
        parent_rad = anchor_node.radius
        rad_ratio = mean_rad / parent_rad if parent_rad > 1e-9 else 1.0
        n_siblings = len(anchor_kids) - 1 if anchor in soma_indices else len(anchor_kids)
        is_primary = 1.0 if segment[0] in primary_set else 0.0

        # --- Cell-relative normalization ---
        path_rel = path_length / max_path_in_cell if max_path_in_cell > 1e-9 else 0.0
        radial_rel = radial_extent / max_radial_in_cell if max_radial_in_cell > 1e-9 else 0.0
        radius_rel = mean_rad / cell_mean_radius if cell_mean_radius > 1e-9 else 1.0
        subtree_rel = sub_size / max(1, n)

        # --- Ground truth label (majority vote, excluding soma) ---
        label_counts: dict[int, int] = {}
        for i in segment:
            t = nodes[i].type
            if t != 1:  # ground-truth target extraction only
                label_counts[t] = label_counts.get(t, 0) + 1

        if label_counts:
            gt_label = max(label_counts, key=lambda t: label_counts[t])
        else:
            gt_label = nodes[segment[0]].type  # fallback

        # --- Primary-subtree rank features ---
        br_primary_root = primary_root_of[segment[0]]
        if br_primary_root >= 0 and br_primary_root in subtree_max_path_by_root:
            this_sub_max_p = subtree_max_path_by_root[br_primary_root]
            this_sub_z = subtree_z_span_by_root[br_primary_root]
            this_sub_min_r = subtree_min_radius_by_root[br_primary_root]
            is_longest_sub = 1.0 if this_sub_max_p >= max_subtree_path_across - 1e-9 else 0.0
            sub_path_ratio = (
                this_sub_max_p / max_subtree_path_across
                if max_subtree_path_across > 1e-9 else 1.0
            )
            sub_z_ratio = (
                this_sub_z / max_subtree_z_across
                if max_subtree_z_across > 1e-9 else 0.0
            )
            sub_z_rank = z_rank_of_root.get(br_primary_root, 0.0)
            sub_radial_rank = radial_rank_of_root.get(br_primary_root, 0.0)
        else:
            is_longest_sub = 0.0
            sub_path_ratio = 0.0
            sub_z_ratio = 0.0
            sub_z_rank = 0.0
            sub_radial_rank = 0.0
            this_sub_min_r = mean_rad

        # --- Axon initial segment / thin-long signatures ---
        starts_at_soma = 1.0 if anchor in soma_indices else 0.0
        first_win = min(3, len(segment))
        first_mean_r = float(np.mean([nodes[segment[k]].radius for k in range(first_win)]))
        anchor_r = anchor_node.radius
        radius_drop = (
            (anchor_r - first_mean_r) / anchor_r if anchor_r > 1e-9 else 0.0
        )
        thin_nodes = sum(1 for r in radii if r < 0.5)
        thin_frac = thin_nodes / max(1, len(radii))

        # --- Multi-scale proximal / distal features ---
        seg_len = len(segment)
        third = max(1, seg_len // 3)
        proximal_radii = [nodes[segment[k]].radius for k in range(third)]
        distal_radii = [nodes[segment[k]].radius for k in range(seg_len - third, seg_len)]
        prox_mean_r = float(np.mean(proximal_radii)) if proximal_radii else mean_rad
        dist_mean_r = float(np.mean(distal_radii)) if distal_radii else mean_rad

        if seg_len >= 3:
            # Proximal persistence
            prox_path = 0.0
            for k in range(1, third):
                prox_path += _euclidean(nodes[segment[k - 1]], nodes[segment[k]])
            prox_eu = _euclidean(nodes[segment[0]], nodes[segment[third - 1]])
            prox_persist = prox_eu / prox_path if prox_path > 1e-9 else persistence

            # Distal persistence
            dist_path = 0.0
            for k in range(seg_len - third + 1, seg_len):
                dist_path += _euclidean(nodes[segment[k - 1]], nodes[segment[k]])
            dist_eu = _euclidean(nodes[segment[seg_len - third]], nodes[segment[-1]])
            dist_persist = dist_eu / dist_path if dist_path > 1e-9 else persistence
        else:
            prox_persist = persistence
            dist_persist = persistence

        # Build feature vector
        fv = np.array([
            path_length,
            radial_extent,
            float(len(segment)),
            mean_rad,
            std_rad,
            min_rad,
            max_rad,
            taper,
            persistence,
            up_align,
            branchiness,
            symmetry,

            root_path_dist,
            root_radial_dist,
            z_rel,
            z_sp,

            float(sub_size),
            float(sub_depth),
            sub_max_path,
            float(b_order),

            parent_rad,
            rad_ratio,
            float(n_siblings),
            is_primary,

            path_rel,
            radial_rel,
            radius_rel,
            subtree_rel,

            is_pyramidal,
            is_interneuron,
            is_purkinje,

            # Primary-subtree rank features
            is_longest_sub,
            sub_path_ratio,
            sub_z_ratio,
            sub_z_rank,
            sub_radial_rank,

            # Axon initial segment / thin-long signatures
            starts_at_soma,
            radius_drop,
            thin_frac,
            this_sub_min_r,

            # Multi-scale proximal / distal
            prox_mean_r,
            dist_mean_r,
            prox_persist,
            dist_persist,
        ], dtype=np.float64)

        branches.append(BranchData(
            branch_id=bid,
            node_indices=segment,
            anchor_idx=anchor,
            features=fv,
            gt_label=gt_label,
            gt_label_counts=label_counts,
            n_nodes=len(segment),
            primary_root_idx=br_primary_root if br_primary_root >= 0 else None,
        ))

    return MorphologyBranches(
        file_path=file_path,
        cell_type=cell_type,
        branches=branches,
        soma_indices=sorted(soma_indices),
        total_nodes=n,
    )
