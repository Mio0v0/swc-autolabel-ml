"""Hyperparameter sweep for the apical-vs-basal GraphSAGE head (CONTINUATION §9, Day 4-5).

Grid (default):
    hidden  in {32, 64, 128}
    layers  in {2, 3}
    dropout in {0.0, 0.2, 0.5}
  -> 18 configs x 5-fold CV = 90 fold-trainings.

Each fold uses the same hash-bucketed train/test split as the rest of the
pipeline (held-out test set = the 169 pyramidal files in eval_split.json,
NEVER touched here). The sweep only cares about validation F1 from the
5-fold CV on the 763 train files.

Output:
    paper/results/gnn_sweep.json    (configs + per-fold val metrics + ranked summary)
    paper/results/gnn_sweep.csv     (one row per (config, fold))
    Stdout: live leaderboard updated after each config

After the sweep, separately run:
    python -m paper.gnn_apical_basal --hidden=<best> --dropout=<best> --n-layers=<best>
to do the final retrain on all 763 train files and evaluate on the 169 held-out test files.

Usage:
    # Full sweep (~60-120 min on RTX 4080)
    python -m paper.sweep_gnn

    # Faster: shorter epoch budget per fold
    python -m paper.sweep_gnn --epochs 80 --patience 12

    # Subset for sanity-test (~5 min)
    python -m paper.sweep_gnn --hidden 32 64 --layers 2 --dropout 0.2
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from paper.gnn_apical_basal import (
    DEFAULT_DATA_DIR,
    EVAL_SPLIT_PATH,
    DENDRITE_FEATURE_NAMES,
    ApicalBasalSAGE,
    TrainConfig,
    cross_validate,
    load_pyramidal_split,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "paper" / "results"
SWEEP_JSON = RESULTS_DIR / "gnn_sweep.json"
SWEEP_CSV = RESULTS_DIR / "gnn_sweep.csv"


# ---------------------------------------------------------------------------
# Sweep config
# ---------------------------------------------------------------------------


@dataclass
class SweepCell:
    """One point in the hyperparameter grid."""
    hidden: int
    n_layers: int
    dropout: float


def _build_grid(hidden: Sequence[int], layers: Sequence[int], dropout: Sequence[float]) -> list[SweepCell]:
    return [SweepCell(h, L, d) for h, L, d in itertools.product(hidden, layers, dropout)]


# ---------------------------------------------------------------------------
# Per-config runner
# ---------------------------------------------------------------------------


def run_one_config(
    cell: SweepCell,
    train_graphs,
    in_dim: int,
    base_cfg: TrainConfig,
    n_folds: int,
    device: torch.device,
) -> dict:
    cfg = TrainConfig(
        hidden=cell.hidden,
        n_layers=cell.n_layers,
        dropout=cell.dropout,
        lr=base_cfg.lr,
        weight_decay=base_cfg.weight_decay,
        epochs=base_cfg.epochs,
        patience=base_cfg.patience,
        batch_size=base_cfg.batch_size,
        seed=base_cfg.seed,
    )
    probe = ApicalBasalSAGE(
        in_dim=in_dim, hidden=cfg.hidden, dropout=cfg.dropout, n_layers=cfg.n_layers,
    )
    n_params = probe.n_params
    del probe

    print(
        f"\n#### Config: hidden={cell.hidden}  layers={cell.n_layers}  "
        f"dropout={cell.dropout}  params={n_params:,}"
    )
    t0 = time.time()
    folds = cross_validate(
        train_graphs, cfg, device, n_folds=n_folds, in_dim=in_dim,
    )
    dt = time.time() - t0

    macro = np.array([r.val_branch_macro_f1 for r in folds])
    apical = np.array([r.val_branch_apical_f1 for r in folds])
    basal = np.array([r.val_branch_basal_f1 for r in folds])
    cell_mean = np.array([r.val_cell_mean_macro_f1 for r in folds])
    best_epochs = [r.best_epoch for r in folds]

    summary = {
        "config": asdict(cell),
        "n_params": n_params,
        "elapsed_sec": round(dt, 1),
        "cv_branch_macro_f1_mean": float(macro.mean()),
        "cv_branch_macro_f1_std": float(macro.std()),
        "cv_branch_apical_f1_mean": float(apical.mean()),
        "cv_branch_basal_f1_mean": float(basal.mean()),
        "cv_cell_mean_macro_f1_mean": float(cell_mean.mean()),
        "cv_best_epochs": best_epochs,
        "fold_details": [
            {
                "fold": r.fold,
                "best_epoch": r.best_epoch,
                "branch_macro_f1": r.val_branch_macro_f1,
                "branch_apical_f1": r.val_branch_apical_f1,
                "branch_basal_f1": r.val_branch_basal_f1,
                "cell_mean_macro_f1": r.val_cell_mean_macro_f1,
            }
            for r in folds
        ],
    }
    print(
        f"   -> macroF1={macro.mean():.4f}+/-{macro.std():.4f}  "
        f"apF1={apical.mean():.4f}  baF1={basal.mean():.4f}  "
        f"cellF1={cell_mean.mean():.4f}  ({dt:.0f}s)"
    )
    return summary


def _print_leaderboard(summaries: list[dict]) -> None:
    if not summaries:
        return
    ranked = sorted(summaries, key=lambda s: s["cv_branch_macro_f1_mean"], reverse=True)
    print("\n=== Leaderboard (so far, ranked by CV mean branch macro-F1) ===")
    print(
        f"  {'rank':>4}  {'hidden':>6} {'layers':>6} {'drop':>6}  "
        f"{'macroF1':>10}  {'apF1':>8}  {'baF1':>8}  {'cellF1':>8}  {'params':>9}  {'sec':>6}"
    )
    for i, s in enumerate(ranked):
        c = s["config"]
        print(
            f"  {i+1:>4}  {c['hidden']:>6} {c['n_layers']:>6} {c['dropout']:>6.2f}  "
            f"{s['cv_branch_macro_f1_mean']:>7.4f}+/-{s['cv_branch_macro_f1_std']:.3f}  "
            f"{s['cv_branch_apical_f1_mean']:>8.4f}  {s['cv_branch_basal_f1_mean']:>8.4f}  "
            f"{s['cv_cell_mean_macro_f1_mean']:>8.4f}  {s['n_params']:>9,}  {s['elapsed_sec']:>6.0f}"
        )


# ---------------------------------------------------------------------------
# Persistence (incremental — write after each config so a crash doesn't lose work)
# ---------------------------------------------------------------------------


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _save_csv(path: Path, summaries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in summaries:
        c = s["config"]
        for f in s["fold_details"]:
            rows.append({
                "hidden": c["hidden"],
                "n_layers": c["n_layers"],
                "dropout": c["dropout"],
                "n_params": s["n_params"],
                "fold": f["fold"],
                "best_epoch": f["best_epoch"],
                "branch_macro_f1": f["branch_macro_f1"],
                "branch_apical_f1": f["branch_apical_f1"],
                "branch_basal_f1": f["branch_basal_f1"],
                "cell_mean_macro_f1": f["cell_mean_macro_f1"],
            })
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--eval-split", type=Path, default=EVAL_SPLIT_PATH)
    ap.add_argument(
        "--hidden", type=int, nargs="+", default=[32, 64, 128],
        help="hidden dim sweep values",
    )
    ap.add_argument(
        "--layers", type=int, nargs="+", default=[2, 3],
        help="n_layers sweep values",
    )
    ap.add_argument(
        "--dropout", type=float, nargs="+", default=[0.0, 0.2, 0.5],
        help="dropout sweep values",
    )
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=120,
                    help="max epochs per fold (early stopping kicks in earlier)")
    ap.add_argument("--patience", type=int, default=15,
                    help="early stopping patience per fold")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--out-json", type=Path, default=SWEEP_JSON)
    ap.add_argument("--out-csv", type=Path, default=SWEEP_CSV)
    return ap


def main() -> int:
    args = _build_argparser().parse_args()
    grid = _build_grid(args.hidden, args.layers, args.dropout)
    print(
        f"Sweep grid: hidden={list(args.hidden)} x layers={list(args.layers)} "
        f"x dropout={list(args.dropout)}  -> {len(grid)} configs"
    )
    print(f"Per fold: epochs<={args.epochs}, patience={args.patience}, n_folds={args.n_folds}")
    print(f"Total fold-trainings: {len(grid) * args.n_folds}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    base_cfg = TrainConfig(
        hidden=64, n_layers=2, dropout=0.2,
        lr=args.lr, weight_decay=args.weight_decay,
        epochs=args.epochs, patience=args.patience,
        batch_size=args.batch_size, seed=args.seed,
    )

    train_graphs, _test_graphs = load_pyramidal_split(
        args.data_dir, args.eval_split,
        feature_names=DENDRITE_FEATURE_NAMES, progress=True,
    )
    if not train_graphs:
        print("No train graphs — aborting.")
        return 1
    in_dim = train_graphs[0].x.shape[1]

    summaries: list[dict] = []
    sweep_t0 = time.time()
    for i, cell in enumerate(grid):
        print(f"\n========== Config {i+1}/{len(grid)} ==========")
        s = run_one_config(
            cell, train_graphs, in_dim, base_cfg,
            n_folds=args.n_folds, device=device,
        )
        summaries.append(s)
        # Incremental save (so a crash doesn't lose finished configs)
        ranked = sorted(summaries, key=lambda x: x["cv_branch_macro_f1_mean"], reverse=True)
        _save_json(args.out_json, {
            "grid": {"hidden": list(args.hidden), "layers": list(args.layers),
                     "dropout": list(args.dropout)},
            "epochs": args.epochs, "patience": args.patience,
            "n_folds": args.n_folds, "seed": args.seed,
            "n_train_files": len(train_graphs),
            "in_dim": in_dim,
            "completed_configs": len(summaries),
            "total_configs": len(grid),
            "elapsed_total_sec": round(time.time() - sweep_t0, 1),
            "ranked": ranked,
        })
        _save_csv(args.out_csv, summaries)
        _print_leaderboard(summaries)

    print(f"\n=== Sweep complete. Total time: {(time.time() - sweep_t0) / 60:.1f} min ===")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_csv}")
    if summaries:
        best = max(summaries, key=lambda s: s["cv_branch_macro_f1_mean"])
        c = best["config"]
        print(
            f"\nBest config: hidden={c['hidden']}  layers={c['n_layers']}  "
            f"dropout={c['dropout']}  CV macroF1={best['cv_branch_macro_f1_mean']:.4f}"
        )
        print("\nTo retrain on all 763 train files and evaluate on the 169 test files:")
        print(
            f"  python -m paper.gnn_apical_basal "
            f"--hidden {c['hidden']} --dropout {c['dropout']}"
            + (f" --n-layers {c['n_layers']}" if c['n_layers'] != 2 else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
