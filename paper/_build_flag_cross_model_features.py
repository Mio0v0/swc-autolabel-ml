#!/usr/bin/env python3
"""Build deployment-safe cross-model disagreement features for the flag model."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.features import parse_swc  # noqa: E402
from paper.external_baselines import predict_with_cache  # noqa: E402
from paper._eval_baselines_on_v12 import _load_split_from_qc  # noqa: E402

QC_CSV = ROOT / "paper" / "results" / "corpus_qc_v12_uncurated.csv"
BASELINE_CACHE = ROOT / "paper" / "models" / "baselines"
METHODS = ("neurom_rf", "lmeasure_rf", "sholl_rf", "sholl_mlp")
MODES = ("s1", "pyr")


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_qc_paths() -> dict[str, Path]:
    out: dict[str, Path] = {}
    with QC_CSV.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("qc_pass", "").strip().lower() not in {"true", "1"}:
                continue
            p = Path(row["path"])
            if p.is_file():
                out[p.name] = p
    return out


def _frac_counts(labels: list[int], n_nodes: int) -> dict[str, float]:
    denom = max(1.0, float(n_nodes - 1))
    c = Counter(int(x) for x in labels)
    return {
        "axon": c.get(2, 0) / denom,
        "basal": c.get(3, 0) / denom,
        "apical": c.get(4, 0) / denom,
    }


def _mode_cell_type(stage1_pred: str, mode: str) -> str:
    if mode == "pyr":
        return "pyramidal"
    return stage1_pred if stage1_pred in {"pyramidal", "interneuron"} else "pyramidal"


def _baseline_features(
    row: dict,
    nodes: list,
    predictors: dict[str, object],
) -> dict:
    n_nodes = int(row["n_nodes"])
    denom = max(1.0, float(n_nodes - 1))
    v12 = {
        "axon": float(row.get("pred_axon", 0) or 0) / denom,
        "basal": float(row.get("pred_basal", 0) or 0) / denom,
        "apical": float(row.get("pred_apical", 0) or 0) / denom,
    }
    out: dict[str, float] = {}
    agg: dict[str, dict[str, list[float]]] = {
        mode: {
            "l1": [],
            "apical_delta": [],
            "apical_present": [],
            "axon_present": [],
            "class_count": [],
        }
        for mode in MODES
    }

    stage1_pred = str(row.get("stage1_pred", ""))
    for method, predict_fn in predictors.items():
        for mode in MODES:
            cell_type = _mode_cell_type(stage1_pred, mode)
            pred = list(predict_fn(nodes, cell_type))
            if len(pred) != n_nodes:
                raise ValueError(f"{method}:{mode} returned {len(pred)} labels for {n_nodes} nodes")
            frac = _frac_counts(pred, n_nodes)
            prefix = f"xmodel_{method}_{mode}"
            class_count = float(sum(1 for cls in ("axon", "basal", "apical") if frac[cls] > 0.0))
            l1 = float(sum(abs(frac[cls] - v12[cls]) for cls in ("axon", "basal", "apical")))
            ap_delta = float(abs(frac["apical"] - v12["apical"]))
            out[f"{prefix}_axon_frac"] = frac["axon"]
            out[f"{prefix}_basal_frac"] = frac["basal"]
            out[f"{prefix}_apical_frac"] = frac["apical"]
            out[f"{prefix}_class_count"] = class_count
            out[f"{prefix}_v12_l1_frac_delta"] = l1
            out[f"{prefix}_v12_apical_frac_delta"] = ap_delta
            out[f"{prefix}_v12_apical_zero_mismatch"] = float((frac["apical"] == 0.0) != (v12["apical"] == 0.0))
            out[f"{prefix}_v12_axon_zero_mismatch"] = float((frac["axon"] == 0.0) != (v12["axon"] == 0.0))
            agg[mode]["l1"].append(l1)
            agg[mode]["apical_delta"].append(ap_delta)
            agg[mode]["apical_present"].append(float(frac["apical"] > 0.0))
            agg[mode]["axon_present"].append(float(frac["axon"] > 0.0))
            agg[mode]["class_count"].append(class_count)

    for mode, parts in agg.items():
        for key, values in parts.items():
            arr = np.asarray(values, dtype=float)
            out[f"xmodel_{mode}_{key}_mean"] = float(arr.mean()) if arr.size else 0.0
            out[f"xmodel_{mode}_{key}_max"] = float(arr.max()) if arr.size else 0.0
            out[f"xmodel_{mode}_{key}_std"] = float(arr.std()) if arr.size else 0.0
        ap = np.asarray(parts["apical_present"], dtype=float)
        ax = np.asarray(parts["axon_present"], dtype=float)
        out[f"xmodel_{mode}_apical_present_vote_frac"] = float(ap.mean()) if ap.size else 0.0
        out[f"xmodel_{mode}_axon_present_vote_frac"] = float(ax.mean()) if ax.size else 0.0
        out[f"xmodel_{mode}_apical_presence_disagreement"] = float(ap.max() - ap.min()) if ap.size else 0.0
        out[f"xmodel_{mode}_axon_presence_disagreement"] = float(ax.max() - ax.min()) if ax.size else 0.0

    out["xmodel_stage1_pred_is_pyramidal"] = float(stage1_pred == "pyramidal")
    out["xmodel_v12_apical_frac"] = v12["apical"]
    out["xmodel_v12_axon_frac"] = v12["axon"]
    out["xmodel_v12_basal_frac"] = v12["basal"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=ROOT / "paper" / "results" / "heldout_per_cell_f1_seed123_branch3.csv")
    ap.add_argument("--base-features", type=Path, default=ROOT / "paper" / "results" / "flag_confidence_features_seed123_branch3.csv")
    ap.add_argument("--out-features", type=Path, default=ROOT / "paper" / "results" / "flag_features_seed123_branch3_xmodel.csv")
    ap.add_argument("--out-xmodel", type=Path, default=ROOT / "paper" / "results" / "flag_cross_model_features_seed123_branch3.csv")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--max-cells", type=int, default=None)
    args = ap.parse_args()

    labels = pd.read_csv(args.labels)
    if args.max_cells is not None:
        labels = labels.head(args.max_cells).copy()
    qc_paths = _load_qc_paths()
    train, _test = _load_split_from_qc(args.seed)
    predictors: dict[str, object] = {}
    for method in METHODS:
        predictors[method] = predict_with_cache(
            method,
            train,
            seed=args.seed,
            cache_path=BASELINE_CACHE / f"{method}.pkl",
            force_retrain=False,
        )

    rows: list[dict] = []
    t0 = time.perf_counter()
    print(f"Building cross-model features for {len(labels)} rows")
    for i, row in enumerate(labels.to_dict("records"), 1):
        p = qc_paths.get(str(row["file"]))
        if p is None:
            raise SystemExit(f"Missing path for {row['file']}")
        nodes = parse_swc(p)
        feat = {"file": row["file"]}
        feat.update(_baseline_features(row, nodes, predictors))
        rows.append(feat)
        if i % args.progress_every == 0:
            elapsed = (time.perf_counter() - t0) / 60.0
            eta = elapsed * (len(labels) - i) / max(1, i)
            print(f"  ... {i}/{len(labels)} ({elapsed:.1f} min, ETA {eta:.1f} min)", flush=True)

    xdf = pd.DataFrame(rows)
    args.out_xmodel.parent.mkdir(parents=True, exist_ok=True)
    xdf.to_csv(args.out_xmodel, index=False)
    base = pd.read_csv(args.base_features)
    merged = base.merge(xdf, on="file", how="left")
    merged.to_csv(args.out_features, index=False)
    summary = {
        "labels": _display(args.labels),
        "base_features": _display(args.base_features),
        "out_xmodel": _display(args.out_xmodel),
        "out_features": _display(args.out_features),
        "n_rows": int(len(merged)),
        "n_xmodel_columns": int(len([c for c in merged.columns if c.startswith("xmodel_")])),
        "elapsed_min": (time.perf_counter() - t0) / 60.0,
    }
    (args.out_features.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
