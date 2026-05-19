"""SWC type-normalization: replace custom / non-standard node types with
the dominant standard type of the branch the node belongs to.

Some labs (and some NeuroMorpho submitters) annotate sub-cellular features
with non-standard SWC type values, e.g.:
    type 10 — axon hillock / axon initial segment
    type 11 — spines
    type 12 — boutons / synapses
or any other custom value outside the standard {0, 1, 2, 3, 4, 5, 6, 7}.

These nodes are *part of* a real neurite (axon, dendrite, etc.) and should
be labeled as such for training purposes, not treated as a separate class.

The function ``normalize_custom_types`` rewrites every non-standard type
to the dominant standard type ({1, 2, 3, 4}) of the branch the node sits
on. A "branch" here is a contiguous run between bifurcations (or between
a bifurcation and the root). If no standard type is present in the branch,
we walk up the parent chain to inherit the nearest standard ancestor's
type. If no standard ancestor exists either, the node is mapped to 0
(undefined) as a last-resort fallback.

Topology (id, x, y, z, radius, parent) is preserved exactly. Only the
``type`` field is rewritten.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .features import SWCNode

STANDARD_NEURITE_TYPES = {1, 2, 3, 4}
# Anything outside this set is considered "custom" and gets rewritten.


def normalize_custom_types(nodes: list[SWCNode]) -> tuple[list[SWCNode], int]:
    """Return (normalized_nodes, n_rewritten).

    Rule:
      For each node whose type is not in {1, 2, 3, 4}:
        1. Find the branch the node belongs to (contiguous run between
           bifurcations).
        2. Replace its type with the most common standard type in that
           branch.
        3. If the branch contains no standard-type nodes at all, walk up
           parent links until a standard-type ancestor is found; use that.
        4. If neither succeeds, set type to 0.

    Standard types (1=soma, 2=axon, 3=basal/dendrite, 4=apical) are NEVER
    touched.
    """
    n = len(nodes)
    if n == 0:
        return nodes, 0

    # Build topology
    id_to_idx = {nd.id: i for i, nd in enumerate(nodes)}
    children: list[list[int]] = [[] for _ in range(n)]
    parent_idx: list[int] = [-1] * n
    for i, nd in enumerate(nodes):
        if nd.parent != -1 and nd.parent in id_to_idx:
            p = id_to_idx[nd.parent]
            children[p].append(i)
            parent_idx[i] = p

    # A node starts a new branch if:
    #   it has no parent (root), OR
    #   its parent has more than one child (bifurcation)
    # NOTE: we deliberately do NOT split on type change — custom-type
    # nodes are supposed to be ABSORBED into their branch.
    starts = [False] * n
    for i in range(n):
        p = parent_idx[i]
        if p < 0:
            starts[i] = True
        elif len(children[p]) > 1:
            starts[i] = True

    # Parent-first traversal so branch ids propagate correctly
    order: list[int] = []
    seen = [False] * n
    queue = [i for i in range(n) if parent_idx[i] < 0]
    while queue:
        nq: list[int] = []
        for i in queue:
            if seen[i]:
                continue
            seen[i] = True
            order.append(i)
            nq.extend(children[i])
        queue = nq

    # Assign each node to a branch id (= the start index it descends from)
    branch_id: list[int] = [-1] * n
    for i in order:
        if starts[i]:
            branch_id[i] = i
        else:
            branch_id[i] = branch_id[parent_idx[i]]

    # Group nodes by branch
    branch_to_idxs: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        branch_to_idxs[branch_id[i]].append(i)

    # Determine dominant standard type per branch
    branch_dominant: dict[int, int | None] = {}
    for bid, idxs in branch_to_idxs.items():
        std = [nodes[i].type for i in idxs if nodes[i].type in STANDARD_NEURITE_TYPES]
        branch_dominant[bid] = Counter(std).most_common(1)[0][0] if std else None

    # Rewrite custom types
    new_types = [nd.type for nd in nodes]
    n_rewritten = 0
    for i, nd in enumerate(nodes):
        if nd.type in STANDARD_NEURITE_TYPES:
            continue
        # First try branch dominant
        dom = branch_dominant[branch_id[i]]
        if dom is not None:
            new_types[i] = dom
            n_rewritten += 1
            continue
        # Else walk up the parent chain
        j = parent_idx[i]
        while j >= 0 and nodes[j].type not in STANDARD_NEURITE_TYPES:
            j = parent_idx[j]
        if j >= 0:
            new_types[i] = nodes[j].type
            n_rewritten += 1
        else:
            new_types[i] = 0
            n_rewritten += 1

    out = [
        SWCNode(
            id=nd.id, type=new_types[i],
            x=nd.x, y=nd.y, z=nd.z, radius=nd.radius, parent=nd.parent,
        )
        for i, nd in enumerate(nodes)
    ]
    return out, n_rewritten


def count_custom_types(nodes: list[SWCNode]) -> dict[int, int]:
    """Return a {type_value: count} dict of non-standard type occurrences."""
    out: dict[int, int] = {}
    for nd in nodes:
        if nd.type not in STANDARD_NEURITE_TYPES:
            out[nd.type] = out.get(nd.type, 0) + 1
    return out
