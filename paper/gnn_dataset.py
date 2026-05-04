"""GNN dataset builder for the apical-vs-basal head (Paper 1, §6 of CONTINUATION.md).

Converts a `hybrid.branch_features.MorphologyBranches` (the same per-branch
representation Stage 2 already operates on) into a `torch_geometric.data.Data`
object suitable for training a small GraphSAGE / GCN classifier whose only
job is to re-decide apical vs basal for pyramidal-dendrite branches.

Graph structure
---------------
- One node per branch (BranchData).
- Edges = parent-branch <-> child-branch derived from each branch's
  `anchor_idx` (the parent NODE index where the branch attaches). The branch
  whose `node_indices` contain that anchor is the parent branch. Branches
  whose anchor is the proxy soma have no branch-parent (they are primary
  branches and become roots of the branch graph).
- Edges are stored undirected: both (parent->child) and (child->parent) are
  present, since GraphSAGE/GCN/GAT propagate along the given edge list.
- Node features `x`: subset of `BRANCH_FEATURE_NAMES` (configurable via
  `feature_names=`). Default `DENDRITE_FEATURE_NAMES` drops the 3 cell-type
  one-hots and 7 explicit axon-vs-dendrite discriminators that are useless
  once we already know we're inside a pyramidal-dendrite subtree.
- Node labels `y`: 0=basal (SWC type 3), 1=apical (SWC type 4),
  -100=ignored (soma / axon / unlabeled / non-pyramidal). -100 is
  PyTorch's default `cross_entropy` ignore_index, so the standard loss can
  be used directly on `y`.
- `apical_basal_mask`: bool tensor, True only on branches that should
  contribute to loss / metrics. Provided in addition to `y` for explicit
  masking in eval / per-sample weighting.

Per-graph metadata stored on the Data object:
- `file_path`: source SWC path
- `cell_type`: "pyramidal" / "interneuron" (from input MorphologyBranches)
- `feature_names`: tuple of feature names, in column order
- `n_branches`, `n_apical_basal`: ints

Loaders:
- `morphology_to_data(mb, ...)`: one MorphologyBranches -> one PyG Data
- `build_dataset(data_dir, ...)`: walks data_dir/{pyramidal,interneuron}/,
  extracts branches via `hybrid.branch_features.extract_branches`, returns a
  list[Data]. Optional cache-to-disk via `cache_path=`.

Default-included files: only pyramidal cells (since the GNN's role is to
re-decide apical-vs-basal for pyramidal dendrites). To include interneuron
graphs for joint training / pretraining, pass `include_interneurons=True`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from hybrid.branch_features import (
    BRANCH_FEATURE_NAMES,
    BranchData,
    MorphologyBranches,
    extract_branches,
)
from hybrid.features import parse_swc

# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

# Features to drop from the GNN node-feature vector. Two groups:
#   1. Cell-type one-hots — irrelevant once we condition on pyramidal cells.
#   2. Axon-vs-dendrite discriminators — irrelevant once Stage 2 has
#      already classified the branch as a dendrite. Keeping them risks
#      teaching the GNN to use axon-shape signals that are zero / noise
#      across its training distribution (only basal/apical branches).
_FEATURES_TO_DROP_FOR_DENDRITE_GNN: tuple[str, ...] = (
    # 1. Cell-type one-hots
    "is_pyramidal",
    "is_interneuron",
    "is_purkinje",
    # 2. Axon-vs-dendrite discriminators
    "thin_fraction",
    "min_radius_in_subtree",
    "radius_drop_at_anchor",
    "starts_at_soma",
    "bifurcations_per_micron",
    "mean_internode_distance",
    "subtree_bifurcations_per_micron",
)

DENDRITE_FEATURE_NAMES: tuple[str, ...] = tuple(
    f for f in BRANCH_FEATURE_NAMES if f not in _FEATURES_TO_DROP_FOR_DENDRITE_GNN
)
"""Default feature subset for the apical-vs-basal GNN (51 features)."""


# SWC type-column codes (matches hybrid/train_stage2.py LABEL_NAMES).
SOMA_LABEL = 1
AXON_LABEL = 2
BASAL_LABEL = 3
APICAL_LABEL = 4

# GNN class indices (the model outputs 2 logits).
CLASS_BASAL = 0
CLASS_APICAL = 1
CLASS_IGNORE = -100  # PyTorch cross_entropy default ignore_index


# ---------------------------------------------------------------------------
# Single-cell conversion
# ---------------------------------------------------------------------------


def _branch_id_lookup(branches: Sequence[BranchData]) -> dict[int, int]:
    """Map each branch-owned node_idx -> branch_id (= index in branches list).

    BranchData.branch_id is set to the enumerate index inside
    `extract_branches`, so it equals the position in `MorphologyBranches.branches`.
    """
    node_to_branch: dict[int, int] = {}
    for br in branches:
        for ni in br.node_indices:
            node_to_branch[ni] = br.branch_id
    return node_to_branch


def _build_edge_index(branches: Sequence[BranchData]) -> torch.Tensor:
    """Return undirected branch->branch edges as a [2, E] long tensor."""
    node_to_branch = _branch_id_lookup(branches)
    src: list[int] = []
    dst: list[int] = []
    for br in branches:
        # The branch whose nodes contain this branch's anchor is its parent.
        # If the anchor is the proxy soma (not part of any branch), this
        # branch is a primary branch with no branch-parent.
        parent_bid = node_to_branch.get(br.anchor_idx)
        if parent_bid is None or parent_bid == br.branch_id:
            continue
        # Undirected: add both directions for symmetric message passing.
        src.append(parent_bid); dst.append(br.branch_id)
        src.append(br.branch_id); dst.append(parent_bid)
    if not src:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def _select_feature_columns(feature_names: Sequence[str]) -> tuple[list[int], tuple[str, ...]]:
    """Return (column indices into BRANCH_FEATURE_NAMES, names tuple)."""
    name_to_idx = {n: i for i, n in enumerate(BRANCH_FEATURE_NAMES)}
    missing = [n for n in feature_names if n not in name_to_idx]
    if missing:
        raise KeyError(f"Unknown branch feature names: {missing}")
    cols = [name_to_idx[n] for n in feature_names]
    return cols, tuple(feature_names)


def _build_labels(
    branches: Sequence[BranchData],
    cell_type: str,
    only_pyramidal_dendrites: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (y [N], apical_basal_mask [N])."""
    n = len(branches)
    y = torch.full((n,), CLASS_IGNORE, dtype=torch.long)
    mask = torch.zeros((n,), dtype=torch.bool)
    if only_pyramidal_dendrites and cell_type != "pyramidal":
        # Whole graph is masked out — useful only for unsupervised pretraining.
        return y, mask
    for i, br in enumerate(branches):
        if br.gt_label == BASAL_LABEL:
            y[i] = CLASS_BASAL
            mask[i] = True
        elif br.gt_label == APICAL_LABEL:
            y[i] = CLASS_APICAL
            mask[i] = True
        # All other labels (soma=1, axon=2, unknown) stay as ignore.
    return y, mask


def morphology_to_data(
    mb: MorphologyBranches,
    feature_names: Sequence[str] = DENDRITE_FEATURE_NAMES,
    only_pyramidal_dendrites: bool = True,
) -> Data:
    """Convert one `MorphologyBranches` into a PyG `Data` graph.

    Parameters
    ----------
    mb : MorphologyBranches
        From `hybrid.branch_features.extract_branches`.
    feature_names : sequence of str
        Column subset of `BRANCH_FEATURE_NAMES` to use as node features.
        Default `DENDRITE_FEATURE_NAMES` drops cell-type one-hots and the
        axon-vs-dendrite discriminators that are degenerate within the
        pyramidal-dendrite training distribution.
    only_pyramidal_dendrites : bool
        If True (default) and `mb.cell_type != "pyramidal"`, the whole
        graph is emitted with mask=False everywhere. This keeps interneuron
        graphs available for unsupervised pretraining without ever
        contributing them to supervised loss.

    Returns
    -------
    torch_geometric.data.Data with attributes:
        x : float32 [num_branches, num_features]
        edge_index : long [2, num_edges]   (undirected, both directions)
        y : long [num_branches]            (0=basal, 1=apical, -100=ignore)
        apical_basal_mask : bool [num_branches]
        file_path : str
        cell_type : str
        feature_names : tuple[str, ...]
        n_branches, n_apical_basal : int
    """
    cols, feat_names = _select_feature_columns(feature_names)

    # Node features
    if not mb.branches:
        x = torch.zeros((0, len(cols)), dtype=torch.float32)
    else:
        feat_matrix = np.stack([br.features for br in mb.branches], axis=0)
        x = torch.tensor(feat_matrix[:, cols], dtype=torch.float32)
        # Defensive: replace NaN/Inf (rare in practice but safe).
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Edges
    edge_index = _build_edge_index(mb.branches)

    # Labels + mask
    y, mask = _build_labels(mb.branches, mb.cell_type, only_pyramidal_dendrites)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.apical_basal_mask = mask
    # Stash metadata on the Data object — PyG keeps non-tensor attrs as-is.
    data.file_path = mb.file_path
    data.cell_type = mb.cell_type
    data.feature_names = feat_names
    data.n_branches = int(len(mb.branches))
    data.n_apical_basal = int(mask.sum().item())
    return data


# ---------------------------------------------------------------------------
# Multi-cell loader
# ---------------------------------------------------------------------------


@dataclass
class DatasetStats:
    n_files: int
    n_pyramidal: int
    n_interneuron: int
    n_branches_total: int
    n_apical_basal_total: int
    n_apical: int
    n_basal: int
    n_skipped_empty: int


def _iter_swc_files(data_dir: Path, cell_types: Iterable[str]) -> Iterable[tuple[str, Path]]:
    """Yield (cell_type, swc_path) for every .swc under data_dir/<ct>/."""
    for ct in cell_types:
        ct_dir = data_dir / ct
        if not ct_dir.is_dir():
            continue
        # The benchmark layout is data_dir/<ct>/swc/*.swc; some older layouts
        # are data_dir/<ct>/*.swc. Support both.
        candidates = list(ct_dir.rglob("*.swc"))
        for swc in sorted(candidates):
            yield ct, swc


def build_dataset(
    data_dir: Path | str,
    feature_names: Sequence[str] = DENDRITE_FEATURE_NAMES,
    include_interneurons: bool = False,
    only_pyramidal_dendrites: bool = True,
    file_filter: callable | None = None,
    progress: bool = True,
) -> tuple[list[Data], DatasetStats]:
    """Walk `data_dir/{pyramidal[, interneuron]}/**/*.swc`, build PyG graphs.

    Parameters
    ----------
    data_dir : path
        Root containing `pyramidal/` (and optionally `interneuron/`)
        subdirectories of SWC files. Matches the benchmark layout.
    feature_names : sequence of str
        Forwarded to `morphology_to_data`.
    include_interneurons : bool
        If True, also include interneuron files. Their graphs will have
        `apical_basal_mask=False` everywhere (since interneurons have no
        apical) — useful only for unsupervised pretraining.
    only_pyramidal_dendrites : bool
        Forwarded to `morphology_to_data`.
    file_filter : callable(Path) -> bool, optional
        If given, only include files where `file_filter(path)` is True.
        Use this to apply the eval split (held-out test set).
    progress : bool
        Print progress every 200 files.
    """
    data_dir = Path(data_dir)
    cell_types = ("pyramidal", "interneuron") if include_interneurons else ("pyramidal",)

    out: list[Data] = []
    stats = DatasetStats(0, 0, 0, 0, 0, 0, 0, 0)

    for i, (ct, swc_path) in enumerate(_iter_swc_files(data_dir, cell_types)):
        if file_filter is not None and not file_filter(swc_path):
            continue
        try:
            nodes = parse_swc(swc_path)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {swc_path.name}: parse error {e}")
            continue
        if not nodes:
            stats.n_skipped_empty += 1
            continue
        mb = extract_branches(nodes, cell_type=ct, file_path=str(swc_path))
        if not mb.branches:
            stats.n_skipped_empty += 1
            continue
        data = morphology_to_data(
            mb,
            feature_names=feature_names,
            only_pyramidal_dendrites=only_pyramidal_dendrites,
        )
        out.append(data)
        stats.n_files += 1
        if ct == "pyramidal":
            stats.n_pyramidal += 1
        else:
            stats.n_interneuron += 1
        stats.n_branches_total += data.n_branches
        stats.n_apical_basal_total += data.n_apical_basal
        stats.n_apical += int((data.y == CLASS_APICAL).sum().item())
        stats.n_basal += int((data.y == CLASS_BASAL).sum().item())
        if progress and (i + 1) % 200 == 0:
            print(
                f"  processed {i + 1} files, kept {stats.n_files}, "
                f"branches so far: {stats.n_branches_total} "
                f"(apical/basal labeled: {stats.n_apical_basal_total})"
            )

    return out, stats


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Smoke-test gnn_dataset.py on a real benchmark directory."
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/benchmark_pyramidal_interneuron_v1_qc_diag_pruned"),
    )
    ap.add_argument("--limit", type=int, default=10, help="files to process")
    ap.add_argument("--include-interneurons", action="store_true")
    args = ap.parse_args()

    print(f"Dendrite feature set: {len(DENDRITE_FEATURE_NAMES)} features")
    print(f"  dropped: {_FEATURES_TO_DROP_FOR_DENDRITE_GNN}")
    print()

    # Limited walker for smoke test.
    cell_types = ("pyramidal", "interneuron") if args.include_interneurons else ("pyramidal",)
    swcs = list(_iter_swc_files(args.data_dir, cell_types))[: args.limit]
    if not swcs:
        print(f"No SWC files found under {args.data_dir}")
        return 1

    print(f"Loading {len(swcs)} files...")
    n_with_apical = 0
    n_with_basal = 0
    for ct, p in swcs:
        nodes = parse_swc(p)
        mb = extract_branches(nodes, cell_type=ct, file_path=str(p))
        d = morphology_to_data(mb)
        a = int((d.y == CLASS_APICAL).sum().item())
        b = int((d.y == CLASS_BASAL).sum().item())
        i_ignored = int((d.y == CLASS_IGNORE).sum().item())
        n_with_apical += int(a > 0)
        n_with_basal += int(b > 0)
        print(
            f"  [{ct:11s}] {p.name:40s}  "
            f"branches={d.n_branches:4d}  edges={d.edge_index.shape[1]:4d}  "
            f"apical={a:3d}  basal={b:3d}  ignored={i_ignored:3d}"
        )
    print()
    print(f"Files with >=1 apical branch: {n_with_apical}/{len(swcs)}")
    print(f"Files with >=1 basal branch:  {n_with_basal}/{len(swcs)}")
    print(f"Feature dim: {d.x.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
