"""Train a learned accept/abstain gate for Branch3 corrections.

The Branch3 rescue GNN proposes branch-level changes. This script trains a
small deployment-safe classifier that decides whether to accept each proposed
change or keep the current pipeline label.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from paper.branch3_gate import GATE_FEATURE_NAMES, build_gate_features
from paper.gnn_branch3_inference import load_branch3
from paper.gnn_branch3_rescue import BRANCH3_FEATURE_NAMES, CLASS_IGNORE


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT / "paper" / "models" / "v12_gentle_seed123"


def _load_graph_cache(path: Path) -> list[Data]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if tuple(payload.get("feature_names", [])) != tuple(BRANCH3_FEATURE_NAMES):
        raise SystemExit(f"Branch3 cache schema mismatch: {path}")
    return payload["graphs"]


def _current_class_from_raw_x(raw_x: np.ndarray) -> np.ndarray:
    names = {name: i for i, name in enumerate(BRANCH3_FEATURE_NAMES)}
    cols = [names["current_is_axon"], names["current_is_basal"], names["current_is_apical"]]
    return np.argmax(raw_x[:, cols], axis=1).astype(np.int64)


@torch.no_grad()
def _score_graphs(graphs: list[Data], branch3_state, batch_size: int = 64) -> dict[str, np.ndarray]:
    y_all: list[np.ndarray] = []
    w_all: list[np.ndarray] = []
    cur_all: list[np.ndarray] = []
    pred_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    raw_all: list[np.ndarray] = []
    graph_all: list[np.ndarray] = []

    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        raw_x = batch.x.numpy().astype(np.float32)
        x = (raw_x - branch3_state.scaler.mean) / branch3_state.scaler.std
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        logits = branch3_state.model(
            torch.from_numpy(x).to(branch3_state.device),
            batch.edge_index.to(branch3_state.device),
        )
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred = probs.argmax(axis=1).astype(np.int64)
        y = batch.y.numpy().astype(np.int64)
        valid = y != CLASS_IGNORE
        y_all.append(y[valid])
        w_all.append(batch.sample_weight.numpy().astype(np.float32)[valid])
        cur_all.append(_current_class_from_raw_x(raw_x)[valid])
        pred_all.append(pred[valid])
        probs_all.append(probs[valid])
        raw_all.append(raw_x[valid])
        graph_all.append(batch.batch.numpy().astype(np.int64)[valid])

    return {
        "y": np.concatenate(y_all),
        "w": np.concatenate(w_all),
        "current": np.concatenate(cur_all),
        "pred": np.concatenate(pred_all),
        "probs": np.concatenate(probs_all),
        "raw_x": np.concatenate(raw_all),
        "graph": np.concatenate(graph_all),
    }


def _candidate_matrix(scored: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    changed = scored["pred"] != scored["current"]
    X = build_gate_features(
        scored["raw_x"][changed],
        scored["probs"][changed],
        scored["current"][changed],
        scored["pred"][changed],
    )
    # Accept only when Branch3 fixes a currently wrong branch. Otherwise
    # reject, including cases where both current and Branch3 are wrong.
    y_accept = (
        (scored["pred"][changed] == scored["y"][changed])
        & (scored["current"][changed] != scored["y"][changed])
    ).astype(np.int64)
    weights = scored["w"][changed].astype(np.float32)
    return X, y_accept, weights


def _balanced_weights(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    totals = {cls: float(w[y == cls].sum()) for cls in (0, 1)}
    total = sum(totals.values())
    if total <= 0:
        return w
    factors = {cls: total / (2.0 * max(1.0, mass)) for cls, mass in totals.items()}
    return np.array([float(wi) * factors[int(yi)] for yi, wi in zip(y, w)], dtype=np.float32)


def _apply_scores(scored: dict[str, np.ndarray], cand_scores: np.ndarray, threshold: float) -> np.ndarray:
    out = scored["current"].copy()
    changed_idx = np.flatnonzero(scored["pred"] != scored["current"])
    accept_idx = changed_idx[cand_scores >= threshold]
    out[accept_idx] = scored["pred"][accept_idx]
    return out


def _weighted_metrics(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> dict:
    per = f1_score(y, pred, labels=[0, 1, 2], average=None, sample_weight=w, zero_division=0)
    pr, rc, _, _ = precision_recall_fscore_support(
        y, pred, labels=[0, 1, 2], sample_weight=w, zero_division=0
    )
    return {
        "macro_f1": float(np.mean(per)),
        "axon_f1": float(per[0]),
        "basal_f1": float(per[1]),
        "apical_f1": float(per[2]),
        "apical_precision": float(pr[2]),
        "apical_recall": float(rc[2]),
    }


def _threshold_sweep(scored: dict[str, np.ndarray], cand_scores: np.ndarray) -> tuple[float, list[dict]]:
    rows: list[dict] = []
    for thr in np.linspace(0.05, 0.95, 91):
        pred = _apply_scores(scored, cand_scores, float(thr))
        m = _weighted_metrics(scored["y"], pred, scored["w"])
        rows.append({
            "threshold": float(thr),
            **m,
            "changed_weight_frac": float(
                np.average(pred != scored["current"], weights=scored["w"])
            ),
        })
    # Optimize branch-level macro F1, but break ties toward the safer higher
    # threshold.
    best = max(rows, key=lambda r: (r["macro_f1"], r["threshold"]))
    return float(best["threshold"]), rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    model_dir = args.model_dir.resolve()
    out_path = (args.out or (model_dir / "branch3_gate.joblib")).resolve()
    train_cache = model_dir / "branch3_cache" / "train_full_with_old_gnn.pt"
    test_cache = model_dir / "branch3_cache" / "test_full_with_old_gnn.pt"
    for p in (train_cache, test_cache, model_dir / "gnn_branch3_rescue.pt"):
        if not p.is_file():
            raise SystemExit(f"MISSING: {p}")

    print(f"Loading Branch3: {model_dir / 'gnn_branch3_rescue.pt'}")
    branch3_state = load_branch3(model_dir / "gnn_branch3_rescue.pt")
    train_graphs = _load_graph_cache(train_cache)
    test_graphs = _load_graph_cache(test_cache)
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(train_graphs))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(idx) * 0.20)))
    val_idx = set(idx[:n_val].tolist())
    gate_train = [g for i, g in enumerate(train_graphs) if i not in val_idx]
    gate_val = [g for i, g in enumerate(train_graphs) if i in val_idx]
    print(f"Gate split: train_graphs={len(gate_train)} val_graphs={len(gate_val)} test_graphs={len(test_graphs)}")

    print("Scoring gate-train graphs...")
    train_scored = _score_graphs(gate_train, branch3_state, args.batch_size)
    print("Scoring gate-val graphs...")
    val_scored = _score_graphs(gate_val, branch3_state, args.batch_size)
    print("Scoring held-out test graphs...")
    test_scored = _score_graphs(test_graphs, branch3_state, args.batch_size)

    X_train, y_train, w_train = _candidate_matrix(train_scored)
    X_val, y_val, w_val = _candidate_matrix(val_scored)
    X_test, y_test, w_test = _candidate_matrix(test_scored)
    print(
        f"Candidates: train={len(y_train)} accept_rate={np.average(y_train, weights=w_train):.4f}; "
        f"val={len(y_val)} accept_rate={np.average(y_val, weights=w_val):.4f}; "
        f"test={len(y_test)} accept_rate={np.average(y_test, weights=w_test):.4f}"
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=args.seed,
    )
    model.fit(X_train, y_train, sample_weight=_balanced_weights(y_train, w_train))

    val_scores = model.predict_proba(X_val)[:, 1]
    threshold, sweep = _threshold_sweep(val_scored, val_scores)
    test_scores = model.predict_proba(X_test)[:, 1]
    test_pred = _apply_scores(test_scored, test_scores, threshold)

    current_metrics = _weighted_metrics(test_scored["y"], test_scored["current"], test_scored["w"])
    raw_metrics = _weighted_metrics(test_scored["y"], test_scored["pred"], test_scored["w"])
    gated_metrics = _weighted_metrics(test_scored["y"], test_pred, test_scored["w"])
    print(f"Chosen threshold: {threshold:.3f}")
    print(f"Test current: {current_metrics}")
    print(f"Test raw Branch3: {raw_metrics}")
    print(f"Test gated: {gated_metrics}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": list(GATE_FEATURE_NAMES),
            "threshold": threshold,
            "validation_sweep": sweep,
            "candidate_counts": {
                "train": int(len(y_train)),
                "val": int(len(y_val)),
                "test": int(len(y_test)),
            },
            "test_metrics": {
                "current": current_metrics,
                "raw_branch3": raw_metrics,
                "gated": gated_metrics,
            },
        },
        out_path,
    )
    print(f"Saved gate -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
