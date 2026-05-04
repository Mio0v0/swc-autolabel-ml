"""GraphSAGE apical-vs-basal head for pyramidal dendrite branches.

Day 2-3 deliverable from CONTINUATION.md §6 + §9. A small graph neural
network that re-decides apical vs basal for branches that Stage 2 has
already classified as dendrite. Stage 2 stays unchanged for axon-vs-
dendrite (it's already at F1 0.997 there); we only attack the remaining
hand-crafted-feature ceiling on apical vs basal (current F1 0.9348 / 0.9722
per CONTINUATION §5).

Architecture (CONTINUATION §6):
    Input:  51 dendrite-relevant per-branch features (paper.gnn_dataset
            DENDRITE_FEATURE_NAMES)
    Layer 1: SAGEConv(51 -> hidden) + ReLU + Dropout
    Layer 2: SAGEConv(hidden -> hidden) + ReLU + Dropout
    Head:    Linear(hidden -> 2)
    Edges:   parent <-> child branches in the cell tree (undirected)
    Output:  2-class softmax (basal=0, apical=1) per branch
    Loss:    CrossEntropy with ignore_index=-100 (non-pyramidal-dendrite
             branches are masked via gnn_dataset's y=-100 convention)

Training:
    - Test split is the SAME hash-bucketed 169 pyramidal files from
      hybrid/models/eval_split.json. Never seen during training.
    - Within the 763 train pyramidals, run 5-fold CV. Per fold:
        * Standardize features using train-fold mean/std
        * Adam(lr=1e-3, weight_decay=5e-4)
        * Up to 200 epochs, early stopping on val macro-F1 (patience 25)
        * Track per-branch and per-cell macro-F1 on the val fold
    - Final model: retrain on ALL 763 train cells using the best epoch
      count seen in CV, save to paper/models/gnn_apical_basal.pt with
      embedded scaler params and feature names.

Usage:
    # Sanity-test on 1 fold, fewer epochs
    python -m paper.gnn_apical_basal --quick

    # Full 5-fold CV + final retrain
    python -m paper.gnn_apical_basal

CLI flags (see _build_argparser):
    --hidden, --dropout, --lr, --weight-decay, --epochs, --patience,
    --n-folds, --batch-size, --quick, --skip-final, --no-cuda
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv

from paper.gnn_dataset import (
    CLASS_APICAL,
    CLASS_BASAL,
    CLASS_IGNORE,
    DENDRITE_FEATURE_NAMES,
    build_dataset,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data" / "benchmark_pyramidal_interneuron_v1_qc_diag_pruned"
EVAL_SPLIT_PATH = ROOT / "hybrid" / "models" / "eval_split.json"
DEFAULT_CKPT_PATH = ROOT / "paper" / "models" / "gnn_apical_basal.pt"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ApicalBasalSAGE(nn.Module):
    """N-layer GraphSAGE for binary node classification (apical vs basal).

    n_layers >= 2. With aggr='mean', each layer is roughly
        sigma(W_self @ x_v + W_neigh @ mean(x_u for u in N(v))).
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        dropout: float = 0.2,
        n_classes: int = 2,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        if n_layers < 2:
            raise ValueError(f"n_layers must be >= 2 (got {n_layers})")
        dims = [in_dim] + [hidden] * n_layers
        self.convs = nn.ModuleList(
            [SAGEConv(dims[i], dims[i + 1], aggr="mean") for i in range(n_layers)]
        )
        self.head = nn.Linear(hidden, n_classes)
        self.dropout = dropout
        self.in_dim = in_dim
        self.hidden = hidden
        self.n_classes = n_classes
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Feature scaling (per-fold; persisted with checkpoint)
# ---------------------------------------------------------------------------


@dataclass
class FeatureScaler:
    """Standard z-score scaler. Stored with the checkpoint so inference
    on a new SWC file reproduces the exact training-time normalization."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, graphs: Sequence[Data]) -> "FeatureScaler":
        # Concatenate node features across all graphs in the train fold.
        all_x = torch.cat([g.x for g in graphs], dim=0).numpy()
        mean = all_x.mean(axis=0)
        std = all_x.std(axis=0)
        std[std < 1e-8] = 1.0  # avoid div-by-zero on constant features
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, graphs: Sequence[Data]) -> list[Data]:
        m = torch.from_numpy(self.mean)
        s = torch.from_numpy(self.std)
        out: list[Data] = []
        for g in graphs:
            g2 = g.clone()
            g2.x = (g.x - m) / s
            out.append(g2)
        return out

    def to_state(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state(cls, state: dict) -> "FeatureScaler":
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            std=np.asarray(state["std"], dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# Data loading (re-uses gnn_dataset; respects existing eval_split.json)
# ---------------------------------------------------------------------------


def load_pyramidal_split(
    data_dir: Path,
    eval_split_path: Path,
    feature_names: Sequence[str] = DENDRITE_FEATURE_NAMES,
    progress: bool = True,
) -> tuple[list[Data], list[Data]]:
    """Load pyramidal graphs and split them into (train_graphs, test_graphs)
    using the SAME held-out test files defined in eval_split.json. This
    keeps the GNN's test set identical to Stage 1+2's, so its contribution
    is directly comparable to the pipeline headline numbers.
    """
    with eval_split_path.open() as f:
        eval_split = json.load(f)
    test_pyr_files = set(eval_split["test_files"]["pyramidal"])
    # eval_split.json stores names like
    # "data\benchmark_pyramidal_interneuron_v1_qc_diag_pruned\pyramidal\swc\<name>.swc"
    # — keep only the basename for cross-platform matching.
    test_basenames = {Path(p).name for p in test_pyr_files}

    print(f"Loading pyramidal graphs from {data_dir} ...")
    graphs, stats = build_dataset(
        data_dir,
        feature_names=feature_names,
        include_interneurons=False,
        progress=progress,
    )
    train, test = [], []
    for g in graphs:
        name = Path(g.file_path).name
        (test if name in test_basenames else train).append(g)
    print(
        f"  total={len(graphs)}  train={len(train)}  test={len(test)}  "
        f"(expected: 763 / 169 from eval_split.json)"
    )
    print(
        f"  train apical/basal labeled branches: "
        f"{sum(g.n_apical_basal for g in train)}"
    )
    print(
        f"  test  apical/basal labeled branches: "
        f"{sum(g.n_apical_basal for g in test)}"
    )
    if len(test) != len(test_basenames):
        missing = test_basenames - {Path(g.file_path).name for g in test}
        if missing:
            print(f"  WARN: {len(missing)} test files not found: e.g. {next(iter(missing))}")
    return train, test


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    hidden: int = 64
    n_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 200
    patience: int = 25
    batch_size: int = 16
    seed: int = 42


@dataclass
class FoldResult:
    fold: int
    best_epoch: int
    train_loss: float
    val_branch_macro_f1: float
    val_branch_apical_f1: float
    val_branch_basal_f1: float
    val_cell_mean_macro_f1: float
    n_train: int
    n_val: int
    epoch_history: list[dict] = field(default_factory=list)


def _make_loader(graphs: Sequence[Data], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(list(graphs), batch_size=batch_size, shuffle=shuffle)


def _eval_predictions(
    model: ApicalBasalSAGE,
    graphs: Sequence[Data],
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Run inference on `graphs`. Returns:
        flat_y_true, flat_y_pred (apical/basal labeled nodes only),
        per_graph_pairs (list of (y_true, y_pred) per graph for per-cell metrics).
    """
    model.eval()
    flat_true, flat_pred = [], []
    per_graph: list[tuple[np.ndarray, np.ndarray]] = []
    with torch.no_grad():
        for batch in _make_loader(graphs, batch_size=batch_size, shuffle=False):
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            preds = logits.argmax(dim=1).cpu().numpy()
            y = batch.y.cpu().numpy()
            graph_idx = batch.batch.cpu().numpy()
            for gi in range(int(graph_idx.max()) + 1 if len(graph_idx) else 0):
                mask = (graph_idx == gi) & (y != CLASS_IGNORE)
                yt = y[mask]
                yp = preds[mask]
                if yt.size == 0:
                    continue
                per_graph.append((yt, yp))
                flat_true.append(yt)
                flat_pred.append(yp)
    if flat_true:
        return (np.concatenate(flat_true), np.concatenate(flat_pred), per_graph)
    return np.array([], dtype=int), np.array([], dtype=int), per_graph


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Return (macro_f1, apical_f1, basal_f1)."""
    if y_true.size == 0:
        return 0.0, 0.0, 0.0
    f1s = f1_score(y_true, y_pred, labels=[CLASS_BASAL, CLASS_APICAL], average=None, zero_division=0)
    basal_f1, apical_f1 = float(f1s[0]), float(f1s[1])
    return (basal_f1 + apical_f1) / 2.0, apical_f1, basal_f1


def _per_cell_macro_f1(per_graph: list[tuple[np.ndarray, np.ndarray]]) -> float:
    if not per_graph:
        return 0.0
    f1s = []
    for yt, yp in per_graph:
        macro, _, _ = _macro_f1(yt, yp)
        f1s.append(macro)
    return float(np.mean(f1s))


def train_one_fold(
    train_graphs: Sequence[Data],
    val_graphs: Sequence[Data],
    cfg: TrainConfig,
    device: torch.device,
    fold: int,
    in_dim: int,
    log_prefix: str = "",
) -> tuple[ApicalBasalSAGE, FoldResult]:
    torch.manual_seed(cfg.seed + fold)
    np.random.seed(cfg.seed + fold)

    # Standardize using TRAIN-fold stats only.
    scaler = FeatureScaler.fit(train_graphs)
    train_z = scaler.transform(train_graphs)
    val_z = scaler.transform(val_graphs)

    model = ApicalBasalSAGE(
        in_dim=in_dim, hidden=cfg.hidden, dropout=cfg.dropout, n_layers=cfg.n_layers,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_macro_f1 = -1.0
    best_epoch = -1
    best_state: dict | None = None
    best_apical = 0.0
    best_basal = 0.0
    best_cell_mean = 0.0
    epochs_since_improve = 0
    epoch_history: list[dict] = []

    train_loader = _make_loader(train_z, batch_size=cfg.batch_size, shuffle=True)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(logits, batch.y, ignore_index=CLASS_IGNORE)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)

        # Validation
        yt, yp, per_graph = _eval_predictions(model, val_z, device)
        macro, apical, basal = _macro_f1(yt, yp)
        cell_mean = _per_cell_macro_f1(per_graph)
        epoch_history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_macro_f1": macro,
            "val_apical_f1": apical,
            "val_basal_f1": basal,
            "val_cell_mean_macro_f1": cell_mean,
        })

        improved = macro > best_macro_f1 + 1e-6
        if improved:
            best_macro_f1 = macro
            best_epoch = epoch
            best_apical = apical
            best_basal = basal
            best_cell_mean = cell_mean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  {log_prefix}fold{fold} ep{epoch:03d}  loss={avg_loss:.4f}  "
                f"val: macroF1={macro:.4f} apF1={apical:.4f} baF1={basal:.4f} "
                f"cellF1={cell_mean:.4f}  "
                f"(best ep{best_epoch} F1={best_macro_f1:.4f})"
            )
        if epochs_since_improve >= cfg.patience:
            print(f"  {log_prefix}fold{fold} early stop at ep{epoch} (no val improvement for {cfg.patience} ep)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    res = FoldResult(
        fold=fold,
        best_epoch=best_epoch,
        train_loss=avg_loss,
        val_branch_macro_f1=best_macro_f1,
        val_branch_apical_f1=best_apical,
        val_branch_basal_f1=best_basal,
        val_cell_mean_macro_f1=best_cell_mean,
        n_train=len(train_graphs),
        n_val=len(val_graphs),
        epoch_history=epoch_history,
    )
    return model, res


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

    kf = KFold(n_splits=n_folds, shuffle=False)  # already shuffled
    fold_results: list[FoldResult] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        train_graphs = [graphs[i] for i in indices[train_idx]]
        val_graphs = [graphs[i] for i in indices[val_idx]]
        print(
            f"\n=== Fold {fold + 1}/{n_folds}  "
            f"train={len(train_graphs)}  val={len(val_graphs)} ==="
        )
        t0 = time.time()
        _, res = train_one_fold(
            train_graphs, val_graphs, cfg, device, fold=fold, in_dim=in_dim,
        )
        dt = time.time() - t0
        print(
            f"  fold{fold} best ep={res.best_epoch}  val macroF1={res.val_branch_macro_f1:.4f}  "
            f"({dt:.0f}s)"
        )
        fold_results.append(res)

    return fold_results


def fit_final(
    train_graphs: Sequence[Data],
    test_graphs: Sequence[Data],
    cfg: TrainConfig,
    device: torch.device,
    in_dim: int,
    n_epochs: int,
) -> tuple[ApicalBasalSAGE, FeatureScaler, dict]:
    """Retrain on the entire train pool for a fixed number of epochs, then
    score on the held-out test set."""
    print(f"\n=== Final retrain on full train (n={len(train_graphs)}) for {n_epochs} epochs ===")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    scaler = FeatureScaler.fit(train_graphs)
    train_z = scaler.transform(train_graphs)
    test_z = scaler.transform(test_graphs)
    model = ApicalBasalSAGE(
        in_dim=in_dim, hidden=cfg.hidden, dropout=cfg.dropout, n_layers=cfg.n_layers,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = _make_loader(train_z, batch_size=cfg.batch_size, shuffle=True)
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(logits, batch.y, ignore_index=CLASS_IGNORE)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  ep{epoch:03d}  train_loss={total_loss / max(n,1):.4f}")
    # Score on held-out test set
    yt, yp, per_graph = _eval_predictions(model, test_z, device)
    macro, apical, basal = _macro_f1(yt, yp)
    cell_mean = _per_cell_macro_f1(per_graph)
    test_metrics = {
        "n_test_files": len(test_graphs),
        "n_test_branches_apical_basal": int(yt.size),
        "branch_macro_f1": macro,
        "branch_apical_f1": apical,
        "branch_basal_f1": basal,
        "cell_mean_macro_f1": cell_mean,
    }
    print(
        f"\nHELD-OUT TEST  branchMacroF1={macro:.4f}  apF1={apical:.4f}  baF1={basal:.4f}  "
        f"cellMeanF1={cell_mean:.4f}"
    )
    return model, scaler, test_metrics


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: ApicalBasalSAGE,
    scaler: FeatureScaler,
    cfg: TrainConfig,
    feature_names: Sequence[str],
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
        },
        "scaler": scaler.to_state(),
        "feature_names": list(feature_names),
        "train_config": asdict(cfg),
        "cv_summary": [
            {
                "fold": r.fold,
                "best_epoch": r.best_epoch,
                "val_branch_macro_f1": r.val_branch_macro_f1,
                "val_branch_apical_f1": r.val_branch_apical_f1,
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
    print(f"\nSaved checkpoint to {path}  ({path.stat().st_size / 1024:.1f} KB)")


def load_checkpoint(path: Path, device: torch.device) -> tuple[ApicalBasalSAGE, FeatureScaler, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload["model_config"]
    model = ApicalBasalSAGE(
        in_dim=cfg["in_dim"],
        hidden=cfg["hidden"],
        n_classes=cfg["n_classes"],
        dropout=cfg["dropout"],
        n_layers=cfg.get("n_layers", 2),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    scaler = FeatureScaler.from_state(payload["scaler"])
    return model, scaler, payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--eval-split", type=Path, default=EVAL_SPLIT_PATH)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT_PATH)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true",
                    help="Sanity-test: 1 fold, 30 epochs, patience 10")
    ap.add_argument("--skip-final", action="store_true",
                    help="Skip the final retrain on all train data")
    ap.add_argument("--no-cuda", action="store_true")
    return ap


def main() -> int:
    args = _build_argparser().parse_args()

    cfg = TrainConfig(
        hidden=args.hidden, n_layers=args.n_layers, dropout=args.dropout,
        lr=args.lr, weight_decay=args.weight_decay,
        epochs=30 if args.quick else args.epochs,
        patience=10 if args.quick else args.patience,
        batch_size=args.batch_size, seed=args.seed,
    )
    n_folds = 1 if args.quick else args.n_folds

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    train_graphs, test_graphs = load_pyramidal_split(
        args.data_dir, args.eval_split,
        feature_names=DENDRITE_FEATURE_NAMES,
        progress=True,
    )
    if not train_graphs:
        print("No train graphs — aborting.")
        return 1
    in_dim = train_graphs[0].x.shape[1]
    n_train_branches = sum(g.n_apical_basal for g in train_graphs)
    print(f"\nTrain graphs: {len(train_graphs)}  apical/basal-labeled branches: {n_train_branches}")
    print(f"Input feature dim: {in_dim}")

    # Build a model just to print param count.
    probe = ApicalBasalSAGE(
        in_dim=in_dim, hidden=cfg.hidden, dropout=cfg.dropout, n_layers=cfg.n_layers,
    )
    print(
        f"Model: {cfg.n_layers}-layer GraphSAGE, hidden={cfg.hidden}, "
        f"dropout={cfg.dropout}, params={probe.n_params:,}"
    )
    del probe

    # === Cross-validation ===
    quick_model: ApicalBasalSAGE | None = None
    quick_scaler: FeatureScaler | None = None
    if args.quick:
        # Just one quick train-on-90/val-on-10 sanity run
        rng = np.random.default_rng(cfg.seed)
        indices = np.arange(len(train_graphs))
        rng.shuffle(indices)
        cut = int(0.9 * len(indices))
        tr = [train_graphs[i] for i in indices[:cut]]
        va = [train_graphs[i] for i in indices[cut:]]
        print(f"\n=== Quick sanity run  train={len(tr)}  val={len(va)} ===")
        quick_scaler = FeatureScaler.fit(tr)
        quick_model, res = train_one_fold(tr, va, cfg, device, fold=0, in_dim=in_dim)
        cv_results = [res]
    else:
        cv_results = cross_validate(train_graphs, cfg, device, n_folds=n_folds, in_dim=in_dim)

    # CV summary
    macro = np.array([r.val_branch_macro_f1 for r in cv_results])
    apical = np.array([r.val_branch_apical_f1 for r in cv_results])
    basal = np.array([r.val_branch_basal_f1 for r in cv_results])
    cell = np.array([r.val_cell_mean_macro_f1 for r in cv_results])
    print("\n=== CV summary ===")
    print(f"  branch macro-F1: {macro.mean():.4f} +/- {macro.std():.4f}")
    print(f"  apical-F1      : {apical.mean():.4f} +/- {apical.std():.4f}")
    print(f"  basal-F1       : {basal.mean():.4f} +/- {basal.std():.4f}")
    print(f"  per-cell mean  : {cell.mean():.4f} +/- {cell.std():.4f}")

    # === Final retrain + held-out test ===
    if args.skip_final or args.quick:
        print("\nSkipping final retrain (--skip-final or --quick).")
        if quick_model is not None and quick_scaler is not None:
            # In --quick mode, save the quick-run model so the checkpoint is usable.
            save_checkpoint(
                args.ckpt, quick_model, quick_scaler, cfg,
                DENDRITE_FEATURE_NAMES, cv_results, None, final_epochs=0,
            )
        else:
            print(f"  No model to save (skip-final without --quick); "
                  f"only CV summary reported.")
        return 0

    median_best_epoch = int(np.median([r.best_epoch for r in cv_results]))
    final_epochs = max(median_best_epoch + 1, 30)
    model, scaler, test_metrics = fit_final(
        train_graphs, test_graphs, cfg, device, in_dim=in_dim, n_epochs=final_epochs,
    )
    save_checkpoint(
        args.ckpt, model, scaler, cfg, DENDRITE_FEATURE_NAMES,
        cv_results, test_metrics, final_epochs=final_epochs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
