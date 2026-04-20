"""Primary-subtree feature extraction for pyramidal owner classification.

This module builds one feature vector per soma-child subtree. The target
task is subtree-owner prediction among {axon, basal/dendrite, apical}.
The features are label-free except for the training target, which is
derived from the existing SWC type column.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .features import SWCNode


PRIMARY_SUBTREE_FEATURE_NAMES: list[str] = [
    "subtree_size",
    "subtree_depth",
    "subtree_max_path",
    "subtree_max_radial",
    "subtree_mean_radius",
    "subtree_std_radius",
    "subtree_min_radius",
    "subtree_root_radius",
    "subtree_taper_ratio",
    "subtree_mean_z_rel",
    "subtree_z_span",
    "subtree_up_alignment",
    "subtree_branch_density",
    "subtree_terminal_density",
    "root_path_dist",
    "root_radial_dist",
    "size_ratio",
    "path_ratio",
    "z_ratio",
    "radius_ratio",
    "is_longest_subtree",
    "z_rank",
    "radius_rank",
    "n_primary_subtrees",
]


@dataclass
class PrimarySubtreeData:
    root_idx: int
    node_indices: list[int]
    features: np.ndarray
    gt_label: int
    gt_label_counts: dict[int, int]


def _build_tree(nodes: list[SWCNode]) -> tuple[list[int | None], list[list[int]], list[int]]:
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
    return parent_idx, children, roots


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


def _subtree_nodes(root_idx: int, children: list[list[int]]) -> list[int]:
    out: list[int] = []
    stack = [root_idx]
    while stack:
        idx = stack.pop()
        out.append(idx)
        stack.extend(children[idx])
    return out


def _bfs_order(roots: list[int], children: list[list[int]]) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    queue = list(roots)
    while queue:
        idx = queue.pop(0)
        if idx in seen:
            continue
        seen.add(idx)
        order.append(idx)
        queue.extend(sorted(children[idx]))
    return order


def extract_primary_subtrees(
    nodes: list[SWCNode],
    cell_type: str,
) -> list[PrimarySubtreeData]:
    """Extract one feature vector per primary subtree."""
    if not nodes:
        return []

    parent_idx, children, roots = _build_tree(nodes)
    n = len(nodes)

    proxy_root = _select_proxy_root(nodes, children, roots)
    soma_indices = {proxy_root}

    if soma_indices:
        soma_x = float(np.mean([nodes[i].x for i in soma_indices]))
        soma_y = float(np.mean([nodes[i].y for i in soma_indices]))
        soma_z = float(np.mean([nodes[i].z for i in soma_indices]))
    else:
        soma_x = nodes[0].x
        soma_y = nodes[0].y
        soma_z = nodes[0].z

    order = _bfs_order(roots, children)
    path_from_root = [0.0] * n
    for i in order:
        pidx = parent_idx[i]
        if pidx is not None:
            path_from_root[i] = path_from_root[pidx] + _euclidean(nodes[pidx], nodes[i])

    primary_roots: list[int] = []
    for si in sorted(soma_indices):
        for ci in children[si]:
            if ci not in soma_indices:
                primary_roots.append(ci)
    if not primary_roots and roots:
        for ri in roots:
            if ri not in soma_indices:
                primary_roots.append(ri)

    if not primary_roots:
        return []

    subtree_nodes_by_root = {pr: _subtree_nodes(pr, children) for pr in primary_roots}
    max_size = max((len(v) for v in subtree_nodes_by_root.values()), default=1)
    max_path = max(
        (max((path_from_root[i] - path_from_root[pr]) for i in idxs) for pr, idxs in subtree_nodes_by_root.items()),
        default=1.0,
    )
    max_z_span = max(
        ((max(nodes[i].z for i in idxs) - min(nodes[i].z for i in idxs)) for idxs in subtree_nodes_by_root.values()),
        default=1.0,
    )
    cell_mean_radius = float(np.mean([nd.radius for i, nd in enumerate(nodes) if i not in soma_indices] or [nd.radius for nd in nodes]))

    path_rank_pairs = []
    z_rank_pairs = []
    radius_rank_pairs = []
    for pr, idxs in subtree_nodes_by_root.items():
        path_rank_pairs.append((pr, max((path_from_root[i] - path_from_root[pr]) for i in idxs)))
        z_rank_pairs.append((pr, max(nodes[i].z for i in idxs) - min(nodes[i].z for i in idxs)))
        radius_rank_pairs.append((pr, nodes[pr].radius))

    z_rank_pairs.sort(key=lambda x: x[1], reverse=True)
    radius_rank_pairs.sort(key=lambda x: x[1], reverse=True)
    z_rank = {
        pr: (1.0 - rank / max(1, len(z_rank_pairs) - 1)) if len(z_rank_pairs) > 1 else 1.0
        for rank, (pr, _) in enumerate(z_rank_pairs)
    }
    radius_rank = {
        pr: (1.0 - rank / max(1, len(radius_rank_pairs) - 1)) if len(radius_rank_pairs) > 1 else 1.0
        for rank, (pr, _) in enumerate(radius_rank_pairs)
    }
    longest_root = max(path_rank_pairs, key=lambda x: x[1])[0] if path_rank_pairs else -1

    out: list[PrimarySubtreeData] = []
    for pr in primary_roots:
        idxs = subtree_nodes_by_root[pr]
        radii = [nodes[i].radius for i in idxs]
        path_vals = [path_from_root[i] - path_from_root[pr] for i in idxs]
        max_path_local = max(path_vals) if path_vals else 0.0
        z_vals = [nodes[i].z for i in idxs]
        z_span = (max(z_vals) - min(z_vals)) if z_vals else 0.0
        branch_points = 0
        terminals = 0
        leaves: list[int] = []
        for i in idxs:
            if len(children[i]) > 1:
                branch_points += 1
            if not children[i]:
                terminals += 1
                leaves.append(i)
        if leaves:
            best_leaf = max(leaves, key=lambda i: nodes[i].z - nodes[pr].z)
            vec_z = nodes[best_leaf].z - nodes[pr].z
            vec_norm = _euclidean(nodes[pr], nodes[best_leaf])
            up_align = (vec_z / vec_norm + 1.0) * 0.5 if vec_norm > 1e-9 else 0.5
        else:
            up_align = 0.5

        depth = 0
        stack = [(pr, 0)]
        while stack:
            idx, d = stack.pop()
            depth = max(depth, d)
            for ci in children[idx]:
                stack.append((ci, d + 1))

        mean_r = float(np.mean(radii)) if radii else 0.0
        root_r = float(nodes[pr].radius)
        taper = (
            float(np.mean([nodes[i].radius for i in leaves[:3]])) / root_r
            if leaves and root_r > 1e-9 else 1.0
        )
        root_path_dist = path_from_root[pr]
        root_radial_dist = math.sqrt(
            (nodes[pr].x - soma_x) ** 2 + (nodes[pr].y - soma_y) ** 2 + (nodes[pr].z - soma_z) ** 2
        )

        gt_counts: dict[int, int] = {}
        for i in idxs:
            lbl = nodes[i].type
            if lbl != 1:
                gt_counts[lbl] = gt_counts.get(lbl, 0) + 1
        gt_label = max(gt_counts, key=lambda lbl: gt_counts[lbl]) if gt_counts else 3

        fv = np.array([
            float(len(idxs)),
            float(depth),
            max_path_local,
            max(
                math.sqrt(
                    (nodes[i].x - nodes[pr].x) ** 2 +
                    (nodes[i].y - nodes[pr].y) ** 2 +
                    (nodes[i].z - nodes[pr].z) ** 2
                )
                for i in idxs
            ),
            mean_r,
            float(np.std(radii)) if len(radii) > 1 else 0.0,
            float(min(radii)) if radii else 0.0,
            root_r,
            taper,
            float(np.mean([nodes[i].z - soma_z for i in idxs])) if idxs else 0.0,
            z_span,
            max(0.0, min(1.0, up_align)),
            branch_points / max(1, len(idxs)),
            terminals / max(1, len(idxs)),
            root_path_dist,
            root_radial_dist,
            len(idxs) / max(1, max_size),
            max_path_local / max(1e-9, max_path),
            z_span / max(1e-9, max_z_span),
            mean_r / max(1e-9, cell_mean_radius),
            1.0 if pr == longest_root else 0.0,
            z_rank.get(pr, 0.0),
            radius_rank.get(pr, 0.0),
            float(len(primary_roots)),
        ], dtype=np.float64)

        out.append(PrimarySubtreeData(
            root_idx=pr,
            node_indices=idxs,
            features=fv,
            gt_label=gt_label,
            gt_label_counts=gt_counts,
        ))

    return out
