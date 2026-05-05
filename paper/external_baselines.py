#!/usr/bin/env python3
"""External-tool / SOTA-style baselines for the auto-typing paper.

Builds the "classical morphometric features + classifier" comparison rows
reviewers consistently demand when reading any new neuron-labeling
algorithm. Run on the same v9 test split as `paper.baselines` so the
numbers are directly comparable to v6 / v7 / v8 / v9.

Three baselines are implemented:

    neurom_rf           Per-branch NeuroM-style morphometric features
                        (length, radial extent, radii, branchiness,
                        partition asymmetry, taper, etc.) → RandomForest
                        → per-branch label → propagate to nodes.

    sholl_rf            Per-primary-subtree Sholl-derived features
                        (Sholl intersections at multiple radii, peak,
                        total branches, total length, principal-axis
                        orientation) → RandomForest → 3-class
                        apical/basal/axon → propagate to all nodes in
                        the subtree. Mirrors the architecture of
                        Emissah/Tecuatl/Ascoli (2026, bioRxiv) which is
                        the closest related contemporaneous work.

    sholl_mlp           Same Sholl features as above but with a small
                        PyTorch MLP head, again mirroring the Emissah
                        et al. setup more directly. Trained on CPU (the
                        feature dimensionality is small).

All three are implemented as drop-in `predict_fn(nodes, cell_type) ->
list[int]` predictors so they plug into the same `_aggregate_per_file`
harness `paper.baselines` already uses.

Outputs
-------
    paper/results/external_baselines_results.json
    paper/results/snapshots/external_baselines.json
    paper/results/snapshots/external_baselines_per_file.csv

Usage
-----
    python -m paper.external_baselines all
    python -m paper.external_baselines neurom_rf
    python -m paper.external_baselines sholl_rf
    python -m paper.external_baselines sholl_mlp
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Suppress sklearn pickle / torch_geometric chatter in baseline runs.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from hybrid.features import SWCNode, parse_swc                                # noqa: E402
from hybrid.cell_type_detector import CELL_TYPE_LABEL_SETS                    # noqa: E402
from hybrid.evaluate import _compute_metrics, _file_level_split, _summarize_file_scores  # noqa: E402

# Re-use the floor-baselines harness for consistent reporting.
from paper.baselines import _aggregate_per_file, _build_result               # noqa: E402

DEFAULT_DATA_DIR = ROOT / "data" / "v9_merged_dataset"
RESULTS_DIR = ROOT / "paper" / "results"
SNAPSHOT_DIR = RESULTS_DIR / "snapshots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Topology helpers (lightweight; replicate just enough of branch_features for
# the baselines to stand on their own without pulling in the full pipeline).
# =============================================================================

@dataclass
class _Topo:
    """Resolved tree topology for one cell."""
    nodes: list
    id_to_idx: dict[int, int]
    parent_idx: list  # list[int | None]
    children: list[list[int]]
    roots: list[int]


def _build_topo(nodes: list[SWCNode]) -> _Topo:
    id_to_idx = {n.id: i for i, n in enumerate(nodes)}
    n = len(nodes)
    parent_idx: list = [None] * n
    children: list[list[int]] = [[] for _ in range(n)]
    roots: list[int] = []
    for i, nd in enumerate(nodes):
        pidx = id_to_idx.get(nd.parent)
        parent_idx[i] = pidx
        if pidx is not None:
            children[pidx].append(i)
        if nd.parent == -1 or pidx is None:
            roots.append(i)
    return _Topo(nodes, id_to_idx, parent_idx, children, roots)


def _select_proxy_root(topo: _Topo) -> int:
    """Largest-radius root, falling back to the first root or 0."""
    if not topo.nodes:
        return 0
    candidate_roots = topo.roots or [0]
    return max(
        candidate_roots,
        key=lambda idx: (topo.nodes[idx].radius, len(topo.children[idx]), -idx),
    )


def _subtree_indices(topo: _Topo, root_idx: int) -> list[int]:
    """All node indices reachable from root_idx (inclusive) via children."""
    out: list[int] = []
    stack = [root_idx]
    seen: set[int] = set()
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
        stack.extend(topo.children[idx])
    return out


def _euclid(a: SWCNode, b: SWCNode) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


# =============================================================================
# BASELINE 1 — NeuroM-style per-branch features + RandomForest
# =============================================================================
#
# We split the morphology into linear branch segments anchored at roots /
# bifurcations (same as the pipeline's branch decomposition, but reimplemented
# here so the baseline doesn't depend on the proprietary feature extractor).
# For each branch we compute a feature vector that mirrors the kind of metrics
# NeuroM exposes per-section: length, max-radial, max-Z, mean-radius, taper,
# branchiness (downstream branch count), partition asymmetry, persistence
# (number of bifurcations along the branch), and parent-cell-type one-hot.
# A RandomForest is then trained on (features → ground-truth majority label).
# =============================================================================

def _branch_segments(topo: _Topo, proxy: int) -> list[list[int]]:
    """Return a list of branch segments. Each segment is a list of node
    indices forming a linear path from a bifurcation anchor (or root) to
    the next bifurcation / leaf, in BFS order."""
    segments: list[list[int]] = []
    visited: set[int] = set()
    visited.add(proxy)

    queue: list[int] = list(topo.children[proxy])
    while queue:
        start = queue.pop(0)
        if start in visited:
            continue
        seg: list[int] = []
        cur = start
        while cur is not None and cur not in visited:
            visited.add(cur)
            seg.append(cur)
            kids = topo.children[cur]
            if len(kids) != 1:
                # bifurcation or leaf — branch ends here
                queue.extend(kids)
                break
            cur = kids[0]
        if seg:
            segments.append(seg)
    return segments


def _branch_neurom_features(
    topo: _Topo,
    proxy: int,
    seg: list[int],
    cell_type: str,
) -> np.ndarray:
    """NeuroM-style 16-dim feature vector for a branch segment."""
    nodes = topo.nodes
    soma = nodes[proxy]
    head = nodes[seg[0]]
    tail = nodes[seg[-1]]

    # Path length (sum of segment Euclideans along the branch).
    path_len = 0.0
    for a, b in zip(seg[:-1], seg[1:]):
        path_len += _euclid(nodes[a], nodes[b])
    # Include the anchor → first-node hop so short segments aren't zero.
    parent_idx = topo.parent_idx[seg[0]]
    if parent_idx is not None:
        path_len += _euclid(nodes[parent_idx], nodes[seg[0]])

    # Radial / euclidean distances from soma.
    radial_head = _euclid(head, soma)
    radial_tail = _euclid(tail, soma)
    radial_max = max(radial_head, radial_tail)
    z_above = (tail.z - soma.z)

    # Radius statistics.
    radii = np.array([nodes[i].radius for i in seg], dtype=np.float64)
    mean_r = float(radii.mean())
    min_r = float(radii.min())
    max_r = float(radii.max())
    # Taper = (head_r - tail_r) / max(eps, head_r)
    head_r = float(nodes[seg[0]].radius)
    tail_r = float(nodes[seg[-1]].radius)
    taper = (head_r - tail_r) / max(1e-6, head_r)

    # Downstream branch count (proxy for "branchiness" / partition mass).
    sub = _subtree_indices(topo, seg[-1])
    n_subtree_nodes = len(sub)
    n_subtree_bifs = sum(1 for i in sub if len(topo.children[i]) >= 2)

    # Persistence proxy: max path-length from the branch tail down its subtree.
    sub_path = 0.0
    for i in sub:
        pi = topo.parent_idx[i]
        if pi is None:
            continue
        sub_path += _euclid(nodes[pi], nodes[i])

    # Partition asymmetry: |L - R| / (L + R) on the bigger sub-branches at tail.
    kids = topo.children[seg[-1]]
    if len(kids) >= 2:
        sizes = [len(_subtree_indices(topo, k)) for k in kids]
        sizes.sort(reverse=True)
        a, b = sizes[0], sizes[1]
        partition_asym = abs(a - b) / max(1, (a + b))
    else:
        partition_asym = 0.0

    # Cell type one-hot (2 features).
    is_pyr = 1.0 if cell_type == "pyramidal" else 0.0
    is_int = 1.0 if cell_type == "interneuron" else 0.0

    return np.array([
        path_len, radial_head, radial_tail, radial_max, z_above,
        mean_r, min_r, max_r, head_r, tail_r, taper,
        n_subtree_nodes, n_subtree_bifs, sub_path, partition_asym,
        is_pyr, is_int,
    ], dtype=np.float64)


def _branch_majority_label(topo: _Topo, seg: list[int]) -> int:
    """Majority neurite label among branch nodes (excluding soma)."""
    counts: dict[int, int] = {}
    for i in seg:
        t = int(topo.nodes[i].type)
        if t == 1:
            continue  # ignore soma nodes inside a non-soma branch
        counts[t] = counts.get(t, 0) + 1
    if not counts:
        return 3  # default basal
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _extract_branch_dataset(
    files_by_ct: dict[str, list[Path]],
    label: str,
):
    """Extract (X, y, file_owner) over a file collection."""
    X: list[np.ndarray] = []
    y: list[int] = []
    print(f"  [{label}] extracting branch features over {sum(len(v) for v in files_by_ct.values())} files...")
    for ct, files in files_by_ct.items():
        for f in files:
            try:
                nodes = parse_swc(f)
            except Exception:
                continue
            if not nodes:
                continue
            topo = _build_topo(nodes)
            proxy = _select_proxy_root(topo)
            for seg in _branch_segments(topo, proxy):
                X.append(_branch_neurom_features(topo, proxy, seg, ct))
                y.append(_branch_majority_label(topo, seg))
    return np.vstack(X) if X else np.zeros((0, 17)), np.array(y, dtype=np.int64)


def predict_neurom_rf(train_files: dict[str, list[Path]], seed: int = 42):
    """Train a RandomForest on per-branch NeuroM-style features. Returns a
    `predict_fn(nodes, cell_type) -> list[int]` predictor for nodes."""
    from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415
    Xtr, ytr = _extract_branch_dataset(train_files, "train")
    print(f"  fitting RandomForest on {Xtr.shape[0]} branches, {Xtr.shape[1]} features...")
    clf = RandomForestClassifier(
        n_estimators=400, max_depth=None, n_jobs=-1, random_state=seed,
        class_weight="balanced",
    )
    clf.fit(Xtr, ytr)

    def fn(nodes, cell_type):
        if not nodes:
            return []
        topo = _build_topo(nodes)
        proxy = _select_proxy_root(topo)
        out = [int(n.type) for n in nodes]  # default keeps original
        # Force soma label for the proxy and existing-soma nodes.
        for i, nd in enumerate(nodes):
            if nd.type == 1:
                out[i] = 1
        out[proxy] = 1
        # Predict per-branch then propagate to all branch nodes.
        valid = set(CELL_TYPE_LABEL_SETS.get(cell_type, {1, 2, 3}))
        for seg in _branch_segments(topo, proxy):
            if not seg:
                continue
            feats = _branch_neurom_features(topo, proxy, seg, cell_type)
            pred = int(clf.predict(feats[np.newaxis, :])[0])
            if pred not in valid:
                # Re-map disallowed apicals on interneurons to basal.
                pred = 3 if 3 in valid else next(iter(valid - {1}), 3)
            for i in seg:
                if topo.nodes[i].type == 1:
                    continue
                out[i] = pred
        return out

    return fn


# =============================================================================
# BASELINE 2 — Sholl + per-subtree features → RandomForest
# (Emissah/Ascoli 2026 style, but with RF instead of GCN for simplicity.)
# =============================================================================

SHOLL_RADII_UM = (10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0)


def _sholl_subtree_features(
    topo: _Topo,
    proxy: int,
    sub_root: int,
    cell_type: str,
) -> np.ndarray:
    """Compute Sholl + structural features for one primary subtree.

    Mirrors the descriptor set used by Emissah/Tecuatl/Ascoli (2026):
    Sholl intersections at fixed radii, max Sholl, peak radius, n_nodes,
    n_bifurcations, max radial distance, max root-to-tip path, total
    arbor length, principal axis orientation (PCA z-component), mean
    radius, plus a cell-type one-hot.
    """
    nodes = topo.nodes
    soma = nodes[proxy]
    sub = _subtree_indices(topo, sub_root)
    if not sub:
        return np.zeros(len(SHOLL_RADII_UM) + 14, dtype=np.float64)

    # Sholl intersections — count parent-child segments crossing each radius.
    parent_dists = np.array(
        [_euclid(nodes[i], soma) for i in sub], dtype=np.float64
    )
    intersections = []
    for r in SHOLL_RADII_UM:
        n_cross = 0
        for i in sub:
            pi = topo.parent_idx[i]
            if pi is None:
                continue
            d_parent = _euclid(nodes[pi], soma)
            d_child = _euclid(nodes[i], soma)
            if (d_parent < r <= d_child) or (d_child < r <= d_parent):
                n_cross += 1
        intersections.append(n_cross)

    max_sholl = float(max(intersections)) if intersections else 0.0
    peak_r_idx = int(np.argmax(intersections)) if intersections else 0
    peak_r = SHOLL_RADII_UM[peak_r_idx]

    n_sub_nodes = len(sub)
    n_bifs = sum(1 for i in sub if len(topo.children[i]) >= 2)
    max_radial = float(parent_dists.max()) if parent_dists.size else 0.0
    mean_radius = float(np.mean([nodes[i].radius for i in sub]))

    # Total arbor length.
    total_len = 0.0
    for i in sub:
        pi = topo.parent_idx[i]
        if pi is not None:
            total_len += _euclid(nodes[pi], nodes[i])

    # Max root-to-tip path (BFS expanding from sub_root).
    dist_along: dict[int, float] = {sub_root: 0.0}
    stack = [sub_root]
    visited_sub: set[int] = set()
    while stack:
        idx = stack.pop()
        if idx in visited_sub:
            continue
        visited_sub.add(idx)
        d_here = dist_along.get(idx, 0.0)
        for c in topo.children[idx]:
            dist_along[c] = d_here + _euclid(nodes[idx], nodes[c])
            stack.append(c)
    max_root_to_tip = float(max(dist_along.values())) if dist_along else 0.0

    # Principal axis (PCA z-component) over subtree node coordinates.
    coords = np.array([(nodes[i].x, nodes[i].y, nodes[i].z) for i in sub], dtype=np.float64)
    if coords.shape[0] >= 2:
        coords -= coords.mean(axis=0)
        try:
            _, _, vt = np.linalg.svd(coords, full_matrices=False)
            principal_axis = vt[0]  # 3D
        except np.linalg.LinAlgError:
            principal_axis = np.array([0.0, 0.0, 1.0])
    else:
        principal_axis = np.array([0.0, 0.0, 1.0])
    pa_x, pa_y, pa_z = principal_axis.tolist()

    # Z-extent above soma — strong apical signal in the literature.
    z_max = max((nodes[i].z - soma.z) for i in sub)
    z_min = min((nodes[i].z - soma.z) for i in sub)

    # Cell type one-hot.
    is_pyr = 1.0 if cell_type == "pyramidal" else 0.0
    is_int = 1.0 if cell_type == "interneuron" else 0.0

    return np.array([
        *intersections,
        max_sholl, peak_r, n_sub_nodes, n_bifs, max_radial, mean_radius,
        total_len, max_root_to_tip,
        abs(pa_x), abs(pa_y), abs(pa_z),
        z_max, z_min,
        is_pyr,  # one feature is enough since cell_type is binary
    ], dtype=np.float64)


def _subtree_majority_label(topo: _Topo, sub_root: int) -> int:
    sub = _subtree_indices(topo, sub_root)
    counts: dict[int, int] = {}
    for i in sub:
        t = int(topo.nodes[i].type)
        if t == 1:
            continue
        counts[t] = counts.get(t, 0) + 1
    if not counts:
        return 3
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _extract_subtree_dataset(files_by_ct: dict[str, list[Path]], label: str):
    X: list[np.ndarray] = []
    y: list[int] = []
    print(f"  [{label}] extracting per-subtree Sholl features over {sum(len(v) for v in files_by_ct.values())} files...")
    for ct, files in files_by_ct.items():
        for f in files:
            try:
                nodes = parse_swc(f)
            except Exception:
                continue
            if not nodes:
                continue
            topo = _build_topo(nodes)
            proxy = _select_proxy_root(topo)
            for sub_root in topo.children[proxy]:
                X.append(_sholl_subtree_features(topo, proxy, sub_root, ct))
                y.append(_subtree_majority_label(topo, sub_root))
    n_features = len(SHOLL_RADII_UM) + 14
    X_arr = np.vstack(X) if X else np.zeros((0, n_features))
    return X_arr, np.array(y, dtype=np.int64)


def _make_sholl_predictor(clf, predict_fn):
    """Wrap a fitted classifier into a `(nodes, cell_type) -> labels` fn."""
    def fn(nodes, cell_type):
        if not nodes:
            return []
        topo = _build_topo(nodes)
        proxy = _select_proxy_root(topo)
        out = [int(n.type) for n in nodes]
        for i, nd in enumerate(nodes):
            if nd.type == 1:
                out[i] = 1
        out[proxy] = 1
        valid = set(CELL_TYPE_LABEL_SETS.get(cell_type, {1, 2, 3}))

        for sub_root in topo.children[proxy]:
            feats = _sholl_subtree_features(topo, proxy, sub_root, cell_type)
            pred = int(predict_fn(clf, feats))
            if pred not in valid:
                pred = 3 if 3 in valid else next(iter(valid - {1}), 3)
            for i in _subtree_indices(topo, sub_root):
                if topo.nodes[i].type == 1:
                    continue
                out[i] = pred
        return out
    return fn


def predict_sholl_rf(train_files: dict[str, list[Path]], seed: int = 42):
    from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415
    Xtr, ytr = _extract_subtree_dataset(train_files, "train")
    print(f"  fitting RandomForest on {Xtr.shape[0]} subtrees, {Xtr.shape[1]} features...")
    clf = RandomForestClassifier(
        n_estimators=400, max_depth=None, n_jobs=-1, random_state=seed,
        class_weight="balanced",
    )
    clf.fit(Xtr, ytr)

    def predict(model, feats):
        return model.predict(feats[np.newaxis, :])[0]
    return _make_sholl_predictor(clf, predict)


def predict_sholl_mlp(train_files: dict[str, list[Path]], seed: int = 42):
    """Sholl features → small MLP (mirrors the Emissah/Ascoli 2026 architecture
    style more directly than RF). Uses sklearn MLPClassifier so we don't pull
    in torch for this one — Emissah et al. report their MLP is "compact" and
    the feature dim is small enough that sklearn's MLP suffices."""
    from sklearn.neural_network import MLPClassifier  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
    from sklearn.pipeline import Pipeline  # noqa: PLC0415

    Xtr, ytr = _extract_subtree_dataset(train_files, "train")
    print(f"  fitting MLP on {Xtr.shape[0]} subtrees, {Xtr.shape[1]} features...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu",
            max_iter=300, early_stopping=True, random_state=seed,
        )),
    ])
    pipe.fit(Xtr, ytr)

    def predict(model, feats):
        return model.predict(feats[np.newaxis, :])[0]
    return _make_sholl_predictor(pipe, predict)


# =============================================================================
# Driver
# =============================================================================

def _collect_test_files_v9(data_dir: Path, test_size: float = 0.2, seed: int = 42):
    """Same hash-bucket split convention used by `hybrid.evaluate` and
    `paper.baselines`. Returns (train_files, test_files) as dicts ct→list."""
    files_by_type: dict[str, list[Path]] = {}
    for ct in ("pyramidal", "interneuron"):
        ct_dir = data_dir / ct
        swc_dir = ct_dir / "swc" if (ct_dir / "swc").is_dir() else ct_dir
        if not swc_dir.is_dir():
            continue
        files_by_type[ct] = sorted(swc_dir.glob("*.swc"))
    train_files, test_files = _file_level_split(files_by_type, test_size, seed)
    return train_files, test_files


def _run_one(method: str, data_dir: Path, seed: int) -> tuple[dict, list[dict]]:
    print(f"\n=== Baseline: {method} ===")
    train_files, test_files = _collect_test_files_v9(data_dir, seed=seed)
    print(f"  train files: {sum(len(v) for v in train_files.values())}")
    print(f"  test files: {sum(len(v) for v in test_files.values())}")

    if method == "neurom_rf":
        predict_fn = predict_neurom_rf(train_files, seed=seed)
    elif method == "sholl_rf":
        predict_fn = predict_sholl_rf(train_files, seed=seed)
    elif method == "sholl_mlp":
        predict_fn = predict_sholl_mlp(train_files, seed=seed)
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"  predicting on {sum(len(v) for v in test_files.values())} test files...")
    file_rows, all_gt, all_pred, gt_by_ct, pred_by_ct = _aggregate_per_file(test_files, predict_fn)
    result = _build_result(method, file_rows, all_gt, all_pred, gt_by_ct, pred_by_ct)
    print(f"  -> overall acc={result['accuracy']:.4f}  "
          f"neurite_macro_f1={result['neurite_macro_f1']:.4f}  "
          f"per_file mean={result['per_file']['mean']:.4f}")
    return result, file_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "method", nargs="?", default="all",
        choices=("all", "neurom_rf", "sholl_rf", "sholl_mlp"),
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "external_baselines_results.json",
    )
    parser.add_argument(
        "--snapshot-json",
        type=Path,
        default=SNAPSHOT_DIR / "external_baselines.json",
    )
    parser.add_argument(
        "--snapshot-csv",
        type=Path,
        default=SNAPSHOT_DIR / "external_baselines_per_file.csv",
    )
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"ERROR: data dir not found: {args.data_dir}", file=sys.stderr)
        return 2

    methods = ["neurom_rf", "sholl_rf", "sholl_mlp"] if args.method == "all" else [args.method]
    all_results: dict[str, dict] = {}
    all_file_rows: list[dict] = []
    for method in methods:
        result, file_rows = _run_one(method, args.data_dir, args.seed)
        all_results[method] = result
        for row in file_rows:
            row = dict(row)
            row["method"] = method
            all_file_rows.append(row)

    # Write outputs.
    args.out_json.write_text(json.dumps(all_results, indent=2))
    args.snapshot_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {args.out_json}")
    print(f"Wrote {args.snapshot_json}")

    if all_file_rows:
        cols = ["method", "path", "cell_type", "n_nodes", "neurite_macro_f1", "macro_f1"]
        with args.snapshot_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in all_file_rows:
                w.writerow({c: r.get(c, "") for c in cols})
        print(f"Wrote {args.snapshot_csv}")

    # Compact summary line for the paper.
    print("\nSummary:")
    print(f"{'method':<15s}  {'acc':>6s}  {'macro_f1':>8s}  {'neurite_f1':>10s}  "
          f"{'soma':>6s}  {'axon':>6s}  {'basal':>6s}  {'apical':>6s}  "
          f"{'pf_mean':>8s}  {'pf_p10':>7s}")
    for method, r in all_results.items():
        per = r.get("per_class_f1", {})
        pf = r.get("per_file", {})
        print(f"{method:<15s}  {r['accuracy']:.4f}  {r['macro_f1']:.4f}  "
              f"{r['neurite_macro_f1']:.4f}    "
              f"{(per.get('soma') or 0):.4f}  {(per.get('axon') or 0):.4f}  "
              f"{(per.get('basal/dendrite') or 0):.4f}  {(per.get('apical') or 0):.4f}  "
              f"{pf.get('mean', 0):.4f}  {pf.get('p10', 0):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
