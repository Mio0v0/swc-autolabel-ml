"""Train an interneuron-only 2-class branch rescue GNN ("Branch2").

Analogous to the pyramidal Branch3 rescue head, but for interneurons, which
have no apical dendrite. It predicts, per branch:

    0 = axon            (SWC 2)
    1 = basal/dendrite  (SWC 3)

Motivation. On the heterogeneous corpus the interneuron path mislabels a
non-trivial fraction of true dendrite as axon (~6% of dendrite nodes), the
mirror image of the apical-as-axon error Branch3 rescues on pyramidals. The
interneuron path has no trained correction head — it leans on Stage 3's
heuristic thin-axon bias, which over-fires. Branch2 is a trained head that
runs AFTER Stage 3 (a "final correction head") and gently flips branches
whose morphology + owner evidence contradict the current axon/dendrite call.

Unlike Branch3 (which runs before Stage 3 so pyramidal hard subtree-voting
can consolidate it), Branch2 runs after Stage 3: interneurons use only soft
Stage-3 rules (axon may emerge from within a dendritic subtree), so there is
no hard consolidation to run afterward, and running last prevents Stage 3's
axon-bias from re-introducing the very error we are correcting. The head
forms an INDEPENDENT morphology + subtree-owner opinion — it is not shown the
pipeline's current label, which would make it echo the label rather than
correct it.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid.branch_features import BRANCH_FEATURE_NAMES, MorphologyBranches, extract_branches
from hybrid.features import parse_swc
from hybrid.pipeline import (
    _branch_feature_with_owner,
    _predict_subtree_owner_map,
)
from hybrid.train_stage2 import OWNER_AUG_FEATURE_NAMES
from paper.gnn_apical_basal import ApicalBasalSAGE, FeatureScaler
from paper.gnn_dataset import _build_edge_index


DATA_DIR = ROOT / "data" / "v12_uncurated"
DEFAULT_MODEL_DIR = ROOT / "paper" / "models" / "v12_gentle_seed123"

CELL_TYPE = "interneuron"
SWC_LABELS = (2, 3)
LABEL_NAMES = {2: "axon", 3: "basal"}
LABEL_TO_CLASS = {2: 0, 3: 1}
CLASS_TO_LABEL = {0: 2, 1: 3}
CLASS_IGNORE = -100

# Extra (per-branch, non-owner-aug) features specific to the rescue head.
# NO current-label ("anchor") features: including the pipeline's own
# post-Stage-3 label made the head echo it — it reproduced the current label
# on ~99% of branches and corrected almost none of the errors. Dropping them
# forces the head to form an INDEPENDENT axon/dendrite opinion from
# morphology + subtree-owner evidence, which it can then use to overturn
# Stage 3's over-promotions. Only owner-derived margins remain.
BRANCH2_EXTRA_FEATURE_NAMES: tuple[str, ...] = (
    "owner_axon_margin",
    "owner_basal_margin",
    "has_owner_info",
)

BRANCH2_FEATURE_NAMES: tuple[str, ...] = (
    tuple(BRANCH_FEATURE_NAMES)
    + tuple(OWNER_AUG_FEATURE_NAMES)
    + BRANCH2_EXTRA_FEATURE_NAMES
)


@dataclass
class TrainConfig:
    hidden: int = 64
    n_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 160
    patience: int = 25
    batch_size: int = 16
    seed: int = 2024
    gnn_type: str = "sage"


@dataclass
class FoldResult:
    fold: int
    best_epoch: int
    train_loss: float
    val_branch_macro_f1: float
    val_branch_axon_f1: float
    val_branch_basal_f1: float
    val_cell_mean_macro_f1: float
    n_train: int
    n_val: int
    epoch_history: list[dict] = field(default_factory=list)


def _load_stage2_bundle(model_dir: Path) -> dict:
    with (model_dir / "branch_classifier.pkl").open("rb") as fh:
        return pickle.load(fh)


def _load_split(model_dir: Path, data_dir: Path) -> tuple[list[Path], list[Path]]:
    payload = json.loads((model_dir / "train_test_split.json").read_text(encoding="utf-8"))
    train_names = payload["train"].get(CELL_TYPE, [])
    test_names = payload["test"].get(CELL_TYPE, [])
    swc_dir = data_dir / CELL_TYPE / "swc"
    train = [swc_dir / fn for fn in train_names if (swc_dir / fn).is_file()]
    test = [swc_dir / fn for fn in test_names if (swc_dir / fn).is_file()]
    return train, test


def _subtree_owner_model(bundle: dict) -> object | None:
    by_ct = bundle.get("subtree_owner_models_by_cell_type")
    if by_ct:
        return by_ct.get(CELL_TYPE)
    return None


def _owner_extra(
    primary_root_idx: int | None,
    subtree_owner_map: dict[int, dict[str, float | int]],
) -> np.ndarray:
    """[owner_axon_margin, owner_basal_margin, has_owner_info]."""
    if primary_root_idx is None or primary_root_idx not in subtree_owner_map:
        return np.zeros(len(BRANCH2_EXTRA_FEATURE_NAMES) - 3, dtype=np.float32)
    info = subtree_owner_map[primary_root_idx]
    p2 = float(info.get("prob_2", 0.0))
    p3 = float(info.get("prob_3", 0.0))
    p4 = float(info.get("prob_4", 0.0))
    return np.array(
        [
            p2 - max(p3, p4),
            p3 - max(p2, p4),
            1.0,
        ],
        dtype=np.float32,
    )


def _branch_feature_vector(
    br,
    subtree_owner_map: dict[int, dict[str, float | int]],
) -> np.ndarray:
    base = _branch_feature_with_owner(br, subtree_owner_map).astype(np.float32)
    owner = _owner_extra(br.primary_root_idx, subtree_owner_map)
    out = np.concatenate([base, owner]).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def morphology_to_branch2_data(
    morph: MorphologyBranches,
    subtree_owner_map: dict[int, dict[str, float | int]],
) -> Data:
    if not morph.branches:
        x = torch.zeros((0, len(BRANCH2_FEATURE_NAMES)), dtype=torch.float32)
    else:
        x = torch.tensor(
            np.stack(
                [_branch_feature_vector(br, subtree_owner_map) for br in morph.branches]
            ),
            dtype=torch.float32,
        )

    y = torch.full((len(morph.branches),), CLASS_IGNORE, dtype=torch.long)
    sample_weight = torch.zeros((len(morph.branches),), dtype=torch.float32)
    mask = torch.zeros((len(morph.branches),), dtype=torch.bool)
    for i, br in enumerate(morph.branches):
        counts = {
            lbl: int(cnt)
            for lbl, cnt in br.gt_label_counts.items()
            if lbl in LABEL_TO_CLASS and cnt > 0
        }
        if not counts:
            continue
        target = max(counts, key=lambda lbl: counts[lbl])
        y[i] = LABEL_TO_CLASS[target]
        sample_weight[i] = float(sum(counts.values()))
        mask[i] = True

    data = Data(x=x, edge_index=_build_edge_index(morph.branches), y=y)
    data.sample_weight = sample_weight
    data.branch2_mask = mask
    data.file_path = morph.file_path
    data.feature_names = BRANCH2_FEATURE_NAMES
    data.n_branches = int(len(morph.branches))
    data.n_labeled = int(mask.sum().item())
    return data


def build_graphs(
    files: Sequence[Path],
    bundle: dict,
    progress: bool = True,
) -> list[Data]:
    owner_model = _subtree_owner_model(bundle)
    graphs: list[Data] = []
    t0 = time.perf_counter()
    for i, path in enumerate(files):
        try:
            nodes = parse_swc(path)
            morph = extract_branches(nodes, CELL_TYPE, str(path))
            owner_map = _predict_subtree_owner_map(nodes, CELL_TYPE, owner_model)
            graph = morphology_to_branch2_data(morph, owner_map)
        except Exception as exc:
            print(f"  WARN build_graph {path.name}: {exc}")
            continue
        if graph.n_branches > 0 and graph.n_labeled > 0:
            graphs.append(graph)
        if progress and (i + 1) % 200 == 0:
            elapsed = (time.perf_counter() - t0) / 60.0
            print(
                f"  processed {i+1}/{len(files)} files, kept {len(graphs)}, "
                f"branches={sum(g.n_branches for g in graphs)}, "
                f"labeled={sum(g.n_labeled for g in graphs)} ({elapsed:.1f} min)",
                flush=True,
            )
    return graphs


def _make_loader(graphs: Sequence[Data], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(list(graphs), batch_size=batch_size, shuffle=shuffle)


def _compute_class_weights(graphs: Sequence[Data], device: torch.device) -> torch.Tensor | None:
    mode = (os.environ.get("SWCAL_BRANCH2_CLASS_WEIGHT", "inverse_sqrt") or "off").lower()
    if mode == "off":
        return None
    totals = np.zeros(2, dtype=np.float64)
    for g in graphs:
        y = g.y.cpu().numpy()
        w = g.sample_weight.cpu().numpy()
        for cls in range(2):
            totals[cls] += float(w[y == cls].sum())
    total = float(totals.sum())
    if total <= 0:
        return None
    raw = total / (2.0 * np.maximum(totals, 1.0))
    if mode == "inverse_sqrt":
        raw = np.sqrt(raw)
    elif mode != "inverse":
        return None
    return torch.tensor(raw, dtype=torch.float32, device=device)


def _branch2_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
    class_weight: torch.Tensor | None,
) -> torch.Tensor:
    valid = target != CLASS_IGNORE
    if not valid.any():
        return logits.sum() * 0.0
    target_v = target[valid]
    log_probs = F.log_softmax(logits[valid], dim=-1)
    log_p_t = log_probs.gather(1, target_v.unsqueeze(1)).squeeze(1)
    loss = -log_p_t

    gamma = float(os.environ.get("SWCAL_BRANCH2_FOCAL_GAMMA", os.environ.get("SWCAL_GNN_FOCAL_GAMMA", "2.0")))
    if gamma > 0.0:
        p_t = log_p_t.exp()
        loss = loss * (1.0 - p_t).pow(gamma)
    if class_weight is not None:
        loss = loss * class_weight[target_v]

    sw = sample_weight[valid].float().clamp_min(1.0)
    loss = loss * sw
    return loss.sum() / sw.sum().clamp_min(1.0)


def _eval_predictions(
    model: ApicalBasalSAGE,
    graphs: Sequence[Data],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    model.eval()
    flat_true: list[np.ndarray] = []
    flat_pred: list[np.ndarray] = []
    per_graph: list[tuple[np.ndarray, np.ndarray]] = []
    with torch.no_grad():
        for batch in _make_loader(graphs, batch_size=batch_size, shuffle=False):
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            pred = logits.argmax(dim=1).cpu().numpy()
            y = batch.y.cpu().numpy()
            graph_idx = batch.batch.cpu().numpy()
            n_graphs = int(graph_idx.max()) + 1 if graph_idx.size else 0
            for gi in range(n_graphs):
                mask = (graph_idx == gi) & (y != CLASS_IGNORE)
                yt = y[mask]
                yp = pred[mask]
                if yt.size == 0:
                    continue
                flat_true.append(yt)
                flat_pred.append(yp)
                per_graph.append((yt, yp))
    if not flat_true:
        return np.array([], dtype=int), np.array([], dtype=int), per_graph
    return np.concatenate(flat_true), np.concatenate(flat_pred), per_graph


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    if y_true.size == 0:
        return 0.0, 0.0, 0.0
    f1s = f1_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    present = [float(f1s[c]) for c in range(2) if np.any(y_true == c)]
    macro = float(np.mean(present)) if present else 0.0
    return macro, float(f1s[0]), float(f1s[1])


def _per_cell_macro_f1(per_graph: list[tuple[np.ndarray, np.ndarray]]) -> float:
    vals = [_macro_f1(yt, yp)[0] for yt, yp in per_graph]
    return float(np.mean(vals)) if vals else 0.0


def train_one_fold(
    train_graphs: Sequence[Data],
    val_graphs: Sequence[Data],
    cfg: TrainConfig,
    device: torch.device,
    fold: int,
    in_dim: int,
) -> tuple[ApicalBasalSAGE, FoldResult]:
    torch.manual_seed(cfg.seed + fold)
    np.random.seed(cfg.seed + fold)

    scaler = FeatureScaler.fit(train_graphs)
    train_z = scaler.transform(train_graphs)
    val_z = scaler.transform(val_graphs)
    model = ApicalBasalSAGE(
        in_dim=in_dim,
        hidden=cfg.hidden,
        dropout=cfg.dropout,
        n_classes=2,
        n_layers=cfg.n_layers,
        gnn_type=cfg.gnn_type,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    class_weight = _compute_class_weights(train_z, device)
    if class_weight is not None:
        print(f"    Branch2 class weight: {class_weight.tolist()}")

    best_macro = -1.0
    best_epoch = -1
    best_state: dict | None = None
    best_scores = (0.0, 0.0, 0.0)
    cell_mean = 0.0
    epochs_since = 0
    history: list[dict] = []
    final_loss = 0.0
    loader = _make_loader(train_z, cfg.batch_size, shuffle=True)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index)
            loss = _branch2_loss(logits, batch.y, batch.sample_weight, class_weight)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
        final_loss = total_loss / max(1, n_batches)

        yt, yp, per_graph = _eval_predictions(model, val_z, device, cfg.batch_size)
        macro, axon, basal = _macro_f1(yt, yp)
        cell_mean = _per_cell_macro_f1(per_graph)
        history.append({
            "epoch": epoch,
            "train_loss": final_loss,
            "val_macro_f1": macro,
            "val_axon_f1": axon,
            "val_basal_f1": basal,
            "val_cell_mean_macro_f1": cell_mean,
        })

        if macro > best_macro + 1e-6:
            best_macro = macro
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_scores = (macro, axon, basal)
            epochs_since = 0
        else:
            epochs_since += 1

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"  fold{fold} ep{epoch:03d} loss={final_loss:.4f} "
                f"val macro={macro:.4f} ax={axon:.4f} ba={basal:.4f} "
                f"cell={cell_mean:.4f} best={best_macro:.4f}"
            )
        if epochs_since >= cfg.patience:
            print(f"  fold{fold} early stop at ep{epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    macro, axon, basal = best_scores
    return model, FoldResult(
        fold=fold,
        best_epoch=best_epoch,
        train_loss=final_loss,
        val_branch_macro_f1=macro,
        val_branch_axon_f1=axon,
        val_branch_basal_f1=basal,
        val_cell_mean_macro_f1=cell_mean,
        n_train=len(train_graphs),
        n_val=len(val_graphs),
        epoch_history=history,
    )


def cross_validate(
    graphs: Sequence[Data],
    cfg: TrainConfig,
    device: torch.device,
    n_folds: int,
    in_dim: int,
) -> list[FoldResult]:
    rng = np.random.default_rng(cfg.seed)
    indices = np.arange(len(graphs))
    rng.shuffle(indices)
    kf = KFold(n_splits=n_folds, shuffle=False)
    results: list[FoldResult] = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(indices)):
        tr = [graphs[i] for i in indices[tr_idx]]
        va = [graphs[i] for i in indices[va_idx]]
        print(f"\n=== Branch2 fold {fold + 1}/{n_folds}: train={len(tr)} val={len(va)} ===")
        _, res = train_one_fold(tr, va, cfg, device, fold, in_dim)
        print(
            f"  fold{fold} best ep={res.best_epoch} macro={res.val_branch_macro_f1:.4f} "
            f"axon={res.val_branch_axon_f1:.4f} basal={res.val_branch_basal_f1:.4f}"
        )
        results.append(res)
    return results


def fit_final(
    train_graphs: Sequence[Data],
    test_graphs: Sequence[Data],
    cfg: TrainConfig,
    device: torch.device,
    in_dim: int,
    n_epochs: int,
) -> tuple[ApicalBasalSAGE, FeatureScaler, dict]:
    print(f"\n=== Branch2 final retrain on {len(train_graphs)} files for {n_epochs} epochs ===")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    scaler = FeatureScaler.fit(train_graphs)
    train_z = scaler.transform(train_graphs)
    test_z = scaler.transform(test_graphs)
    model = ApicalBasalSAGE(
        in_dim=in_dim,
        hidden=cfg.hidden,
        dropout=cfg.dropout,
        n_classes=2,
        n_layers=cfg.n_layers,
        gnn_type=cfg.gnn_type,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    class_weight = _compute_class_weights(train_z, device)
    if class_weight is not None:
        print(f"    Branch2 class weight: {class_weight.tolist()}")
    loader = _make_loader(train_z, cfg.batch_size, shuffle=True)
    for epoch in range(n_epochs):
        model.train()
        total = 0.0
        n = 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index)
            loss = _branch2_loss(logits, batch.y, batch.sample_weight, class_weight)
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"  ep{epoch:03d} train_loss={total / max(1, n):.4f}")

    yt, yp, per_graph = _eval_predictions(model, test_z, device, cfg.batch_size)
    macro, axon, basal = _macro_f1(yt, yp)
    metrics = {
        "n_test_files": len(test_graphs),
        "n_test_branches": int(yt.size),
        "branch_macro_f1": macro,
        "branch_axon_f1": axon,
        "branch_basal_f1": basal,
        "cell_mean_macro_f1": _per_cell_macro_f1(per_graph),
    }
    print(
        f"\nBranch2 held-out branch metrics: macro={macro:.4f} "
        f"axon={axon:.4f} basal={basal:.4f} cell={metrics['cell_mean_macro_f1']:.4f}"
    )
    return model, scaler, metrics


def save_checkpoint(
    path: Path,
    model: ApicalBasalSAGE,
    scaler: FeatureScaler,
    cfg: TrainConfig,
    cv_results: list[FoldResult],
    test_metrics: dict | None,
    final_epochs: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "model_config": {
            "in_dim": model.in_dim,
            "hidden": model.hidden,
            "n_classes": model.n_classes,
            "dropout": model.dropout,
            "n_layers": model.n_layers,
            "gnn_type": getattr(model, "gnn_type", "sage"),
        },
        "scaler": scaler.to_state(),
        "feature_names": list(BRANCH2_FEATURE_NAMES),
        "label_to_class": LABEL_TO_CLASS,
        "class_to_label": CLASS_TO_LABEL,
        "train_config": asdict(cfg),
        "cv_summary": [
            {
                "fold": r.fold,
                "best_epoch": r.best_epoch,
                "val_branch_macro_f1": r.val_branch_macro_f1,
                "val_branch_axon_f1": r.val_branch_axon_f1,
                "val_branch_basal_f1": r.val_branch_basal_f1,
                "val_cell_mean_macro_f1": r.val_cell_mean_macro_f1,
                "n_train": r.n_train,
                "n_val": r.n_val,
            }
            for r in cv_results
        ],
        "test_metrics": test_metrics,
        "final_epochs": final_epochs,
    }
    torch.save(payload, path)
    print(f"\nSaved Branch2 checkpoint to {path} ({path.stat().st_size / 1024:.1f} KB)")


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="Optional graph cache dir. Defaults to <model-dir>/branch2_cache.")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--gnn-type", choices=["sage", "gat", "gin"], default="sage")
    ap.add_argument("--quick", action="store_true",
                    help="Small smoke run: 1 fold, 12 epochs, no final checkpoint overwrite.")
    ap.add_argument("--skip-final", action="store_true")
    return ap


def _cache_path(cache_dir: Path, split: str, quick: bool) -> Path:
    suffix = "quick" if quick else "full"
    return cache_dir / f"{split}_{suffix}.pt"


def _load_or_build_graphs(
    cache_path: Path,
    files: Sequence[Path],
    bundle: dict,
    use_cache: bool,
) -> list[Data]:
    if use_cache and cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if tuple(payload.get("feature_names", [])) == tuple(BRANCH2_FEATURE_NAMES):
            graphs = payload.get("graphs", [])
            print(f"  loaded cache {cache_path} ({len(graphs)} graphs)")
            return graphs
        print(f"  cache schema mismatch, rebuilding: {cache_path}")
    graphs = build_graphs(files, bundle, progress=True)
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"feature_names": list(BRANCH2_FEATURE_NAMES), "graphs": graphs}, cache_path)
        print(f"  saved cache {cache_path} ({len(graphs)} graphs)")
    return graphs


def main() -> int:
    args = _build_argparser().parse_args()
    model_dir = args.model_dir.resolve()
    ckpt = args.ckpt or (model_dir / "gnn_branch2_rescue.pt")
    cache_dir = (args.cache_dir or (model_dir / "branch2_cache")).resolve()
    cfg = TrainConfig(
        hidden=args.hidden,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
        gnn_type=args.gnn_type,
    )
    if args.quick:
        cfg.epochs = min(cfg.epochs, 12)
        cfg.patience = min(cfg.patience, 6)
        args.n_folds = 1
        args.skip_final = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model dir: {model_dir.relative_to(ROOT)}")
    print(f"Branch2 focal gamma: {os.environ.get('SWCAL_BRANCH2_FOCAL_GAMMA', os.environ.get('SWCAL_GNN_FOCAL_GAMMA', '2.0'))}")
    print(f"Branch2 class weight: {os.environ.get('SWCAL_BRANCH2_CLASS_WEIGHT', 'inverse_sqrt')}")

    train_files, test_files = _load_split(model_dir, args.data_dir)
    if args.quick:
        train_files = train_files[:120]
        test_files = test_files[:60]
    print(f"Split: train={len(train_files)} test={len(test_files)}")
    bundle = _load_stage2_bundle(model_dir)

    print("\nBuilding Branch2 train graphs (morphology + owner features)...")
    train_graphs = _load_or_build_graphs(
        _cache_path(cache_dir, "train", args.quick),
        train_files,
        bundle,
        use_cache=not args.quick,
    )
    print("\nBuilding Branch2 held-out graphs...")
    test_graphs = _load_or_build_graphs(
        _cache_path(cache_dir, "test", args.quick),
        test_files,
        bundle,
        use_cache=not args.quick,
    )
    if not train_graphs:
        raise SystemExit("No train graphs built.")
    in_dim = int(train_graphs[0].x.shape[1])
    print(
        f"\nGraphs: train={len(train_graphs)} test={len(test_graphs)} "
        f"in_dim={in_dim} train_labeled={sum(g.n_labeled for g in train_graphs)} "
        f"test_labeled={sum(g.n_labeled for g in test_graphs)}"
    )

    if args.n_folds <= 1:
        fold_results = []
        n_val = max(1, len(train_graphs) // 5)
        _, res = train_one_fold(train_graphs[n_val:], train_graphs[:n_val], cfg, device, 0, in_dim)
        fold_results.append(res)
    else:
        fold_results = cross_validate(train_graphs, cfg, device, args.n_folds, in_dim)

    final_epochs = int(np.median([max(1, r.best_epoch + 1) for r in fold_results]))
    test_metrics = None
    if not args.skip_final:
        model, scaler, test_metrics = fit_final(
            train_graphs, test_graphs, cfg, device, in_dim, n_epochs=final_epochs
        )
        save_checkpoint(ckpt, model, scaler, cfg, fold_results, test_metrics, final_epochs)
    else:
        print("\nSkipping final retrain/checkpoint (--skip-final or --quick).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
