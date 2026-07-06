#!/usr/bin/env python3
"""Build out-of-fold baseline and multi-v12 disagreement features for flagging."""
from __future__ import annotations

import argparse
import csv
import gc
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
from hybrid.pipeline import run_pipeline_on_nodes  # noqa: E402
from paper._eval_baselines_on_v12 import _load_split_from_qc  # noqa: E402
from paper.external_baselines import predict_with_cache  # noqa: E402
from paper.gnn_branch3_inference import load_branch3  # noqa: E402
from paper.gnn_inference import load_gnn  # noqa: E402


QC_CSV = ROOT / "paper" / "results" / "corpus_qc_v12_uncurated.csv"
SEEDS = (123, 42, 789)
METHODS = ("neurom_rf", "lmeasure_rf", "sholl_rf", "sholl_mlp")
MODES = ("s1", "pyr")


def _display(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


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


def _load_split_sets(seed: int) -> tuple[set[str], set[str]]:
    path = ROOT / "paper" / "models" / f"v12_gentle_seed{seed}" / "train_test_split.json"
    split = json.loads(path.read_text(encoding="utf-8"))
    train = set(split["train"].get("pyramidal", [])) | set(split["train"].get("interneuron", []))
    test = set(split["test"].get("pyramidal", [])) | set(split["test"].get("interneuron", []))
    return train, test


def _baseline_cache_dir(seed: int) -> Path:
    if seed == 123:
        return ROOT / "paper" / "models" / "baselines"
    return ROOT / "paper" / "models" / f"baselines_seed{seed}"


def _frac_counts(labels: list[int], n_nodes: int) -> dict[str, float]:
    denom = max(1.0, float(n_nodes - 1))
    c = Counter(int(x) for x in labels)
    return {
        "axon": c.get(2, 0) / denom,
        "basal": c.get(3, 0) / denom,
        "apical": c.get(4, 0) / denom,
    }


def _row_v12_fracs(row: dict) -> dict[str, float]:
    n_nodes = int(row["n_nodes"])
    denom = max(1.0, float(n_nodes - 1))
    return {
        "axon": float(row.get("pred_axon", 0) or 0) / denom,
        "basal": float(row.get("pred_basal", 0) or 0) / denom,
        "apical": float(row.get("pred_apical", 0) or 0) / denom,
    }


def _mode_cell_type(stage1_pred: str, mode: str) -> str:
    if mode == "pyr":
        return "pyramidal"
    return stage1_pred if stage1_pred in {"pyramidal", "interneuron"} else "pyramidal"


def _safe_std(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(arr.std()) if arr.size else 0.0


def _empty_baseline_agg() -> dict[str, dict[str, list[float]]]:
    return {
        mode: {
            "l1": [],
            "apical_delta": [],
            "axon_delta": [],
            "apical_present": [],
            "axon_present": [],
            "class_count": [],
        }
        for mode in MODES
    }


def _add_baseline_method_features(
    out_rows: dict[str, dict],
    agg_rows: dict[str, dict[str, dict[str, list[float]]]],
    row_records: list[dict],
    qc_paths: dict[str, Path],
    method: str,
    predict_fn,
) -> None:
    for row in row_records:
        row_id = str(row["row_id"])
        path = qc_paths.get(str(row["file"]))
        if path is None:
            raise SystemExit(f"Missing path for {row['file']}")
        nodes = parse_swc(path)
        n_nodes = int(row["n_nodes"])
        if len(nodes) != n_nodes:
            n_nodes = len(nodes)
        v12 = _row_v12_fracs(row)
        stage1_pred = str(row.get("stage1_pred", ""))
        for mode in MODES:
            cell_type = _mode_cell_type(stage1_pred, mode)
            pred = list(predict_fn(nodes, cell_type))
            if len(pred) != len(nodes):
                raise ValueError(f"{method}:{mode} returned {len(pred)} labels for {len(nodes)} nodes")
            frac = _frac_counts(pred, n_nodes)
            prefix = f"baseline_oof_{method}_{mode}"
            class_count = float(sum(1 for cls in ("axon", "basal", "apical") if frac[cls] > 0.0))
            l1 = float(sum(abs(frac[cls] - v12[cls]) for cls in ("axon", "basal", "apical")))
            ap_delta = float(abs(frac["apical"] - v12["apical"]))
            ax_delta = float(abs(frac["axon"] - v12["axon"]))
            out = out_rows[row_id]
            out[f"{prefix}_axon_frac"] = frac["axon"]
            out[f"{prefix}_basal_frac"] = frac["basal"]
            out[f"{prefix}_apical_frac"] = frac["apical"]
            out[f"{prefix}_class_count"] = class_count
            out[f"{prefix}_v12_l1_frac_delta"] = l1
            out[f"{prefix}_v12_apical_frac_delta"] = ap_delta
            out[f"{prefix}_v12_axon_frac_delta"] = ax_delta
            out[f"{prefix}_v12_apical_zero_mismatch"] = float((frac["apical"] == 0.0) != (v12["apical"] == 0.0))
            out[f"{prefix}_v12_axon_zero_mismatch"] = float((frac["axon"] == 0.0) != (v12["axon"] == 0.0))
            agg = agg_rows[row_id][mode]
            agg["l1"].append(l1)
            agg["apical_delta"].append(ap_delta)
            agg["axon_delta"].append(ax_delta)
            agg["apical_present"].append(float(frac["apical"] > 0.0))
            agg["axon_present"].append(float(frac["axon"] > 0.0))
            agg["class_count"].append(class_count)


def _finalize_baseline_agg(out_rows: dict[str, dict], agg_rows: dict[str, dict[str, dict[str, list[float]]]]) -> None:
    for row_id, mode_parts in agg_rows.items():
        out = out_rows[row_id]
        for mode, parts in mode_parts.items():
            for key, values in parts.items():
                arr = np.asarray(values, dtype=float)
                out[f"baseline_oof_{mode}_{key}_mean"] = float(arr.mean()) if arr.size else 0.0
                out[f"baseline_oof_{mode}_{key}_max"] = float(arr.max()) if arr.size else 0.0
                out[f"baseline_oof_{mode}_{key}_std"] = float(arr.std()) if arr.size else 0.0
            ap = np.asarray(parts["apical_present"], dtype=float)
            ax = np.asarray(parts["axon_present"], dtype=float)
            out[f"baseline_oof_{mode}_apical_present_vote_frac"] = float(ap.mean()) if ap.size else 0.0
            out[f"baseline_oof_{mode}_axon_present_vote_frac"] = float(ax.mean()) if ax.size else 0.0
            out[f"baseline_oof_{mode}_apical_presence_disagreement"] = float(ap.max() - ap.min()) if ap.size else 0.0
            out[f"baseline_oof_{mode}_axon_presence_disagreement"] = float(ax.max() - ax.min()) if ax.size else 0.0


def _load_v12_model(seed: int) -> dict:
    model_dir = ROOT / "paper" / "models" / f"v12_gentle_seed{seed}"
    s1 = model_dir / "cell_type_classifier.pkl"
    s2 = model_dir / "branch_classifier.pkl"
    gnn = model_dir / "gnn_apical_basal.pt"
    b3 = model_dir / "gnn_branch3_rescue.pt"
    return {
        "seed": seed,
        "stage1_model": s1,
        "stage2_model": s2,
        "gnn_state": load_gnn(gnn),
        "branch3_state": load_branch3(b3) if b3.is_file() else None,
    }


def _add_v12_oof_features(
    out_rows: dict[str, dict],
    row_records: list[dict],
    qc_paths: dict[str, Path],
    seed_test_sets: dict[int, set[str]],
    progress_every: int,
) -> None:
    models = {seed: _load_v12_model(seed) for seed in SEEDS}
    t0 = time.perf_counter()
    n_aux_predictions = 0
    for i, row in enumerate(row_records, 1):
        row_id = str(row["row_id"])
        own_seed = int(row["model_seed"])
        file_name = str(row["file"])
        path = qc_paths.get(file_name)
        if path is None:
            raise SystemExit(f"Missing path for {file_name}")
        nodes = None
        v12 = _row_v12_fracs(row)
        aux_l1: list[float] = []
        aux_ap_delta: list[float] = []
        aux_ax_delta: list[float] = []
        aux_stage1_disagree: list[float] = []
        aux_stage1_conf: list[float] = []
        aux_ap_present: list[float] = []
        aux_ax_present: list[float] = []
        aux_class_count: list[float] = []
        out = out_rows[row_id]
        for aux_seed, model in models.items():
            if aux_seed == own_seed:
                continue
            if file_name not in seed_test_sets[aux_seed]:
                continue
            if nodes is None:
                nodes = parse_swc(path)
            pr = run_pipeline_on_nodes(
                nodes,
                file_path="",
                stage1_model=model["stage1_model"],
                stage2_model=model["stage2_model"],
                gnn_state=model["gnn_state"],
                branch3_state=model["branch3_state"],
                use_subtree_stage2=True,
            )
            labels = list(pr.node_labels)
            frac = _frac_counts(labels, len(nodes))
            prefix = f"xmodel_v12_oof_seed{aux_seed}"
            class_count = float(sum(1 for cls in ("axon", "basal", "apical") if frac[cls] > 0.0))
            l1 = float(sum(abs(frac[cls] - v12[cls]) for cls in ("axon", "basal", "apical")))
            ap_delta = float(abs(frac["apical"] - v12["apical"]))
            ax_delta = float(abs(frac["axon"] - v12["axon"]))
            out[f"{prefix}_available"] = 1.0
            out[f"{prefix}_axon_frac"] = frac["axon"]
            out[f"{prefix}_basal_frac"] = frac["basal"]
            out[f"{prefix}_apical_frac"] = frac["apical"]
            out[f"{prefix}_class_count"] = class_count
            out[f"{prefix}_v12_l1_frac_delta"] = l1
            out[f"{prefix}_v12_apical_frac_delta"] = ap_delta
            out[f"{prefix}_v12_axon_frac_delta"] = ax_delta
            out[f"{prefix}_v12_apical_zero_mismatch"] = float((frac["apical"] == 0.0) != (v12["apical"] == 0.0))
            out[f"{prefix}_v12_axon_zero_mismatch"] = float((frac["axon"] == 0.0) != (v12["axon"] == 0.0))
            out[f"{prefix}_stage1_disagree"] = float(str(pr.stage1.cell_type) != str(row.get("stage1_pred", "")))
            out[f"{prefix}_stage1_conf"] = float(pr.stage1.confidence)
            aux_l1.append(l1)
            aux_ap_delta.append(ap_delta)
            aux_ax_delta.append(ax_delta)
            aux_stage1_disagree.append(float(str(pr.stage1.cell_type) != str(row.get("stage1_pred", ""))))
            aux_stage1_conf.append(float(pr.stage1.confidence))
            aux_ap_present.append(float(frac["apical"] > 0.0))
            aux_ax_present.append(float(frac["axon"] > 0.0))
            aux_class_count.append(class_count)
            n_aux_predictions += 1
        for aux_seed in SEEDS:
            if aux_seed != own_seed:
                out.setdefault(f"xmodel_v12_oof_seed{aux_seed}_available", 0.0)
        out["xmodel_v12_oof_aux_count"] = float(len(aux_l1))
        out["xmodel_v12_oof_l1_mean"] = float(np.mean(aux_l1)) if aux_l1 else 0.0
        out["xmodel_v12_oof_l1_max"] = float(np.max(aux_l1)) if aux_l1 else 0.0
        out["xmodel_v12_oof_l1_std"] = _safe_std(aux_l1)
        out["xmodel_v12_oof_apical_delta_mean"] = float(np.mean(aux_ap_delta)) if aux_ap_delta else 0.0
        out["xmodel_v12_oof_apical_delta_max"] = float(np.max(aux_ap_delta)) if aux_ap_delta else 0.0
        out["xmodel_v12_oof_axon_delta_mean"] = float(np.mean(aux_ax_delta)) if aux_ax_delta else 0.0
        out["xmodel_v12_oof_axon_delta_max"] = float(np.max(aux_ax_delta)) if aux_ax_delta else 0.0
        out["xmodel_v12_oof_stage1_disagree_mean"] = float(np.mean(aux_stage1_disagree)) if aux_stage1_disagree else 0.0
        out["xmodel_v12_oof_stage1_conf_mean"] = float(np.mean(aux_stage1_conf)) if aux_stage1_conf else 0.0
        out["xmodel_v12_oof_stage1_conf_min"] = float(np.min(aux_stage1_conf)) if aux_stage1_conf else 0.0
        out["xmodel_v12_oof_apical_present_vote_frac"] = float(np.mean(aux_ap_present)) if aux_ap_present else 0.0
        out["xmodel_v12_oof_axon_present_vote_frac"] = float(np.mean(aux_ax_present)) if aux_ax_present else 0.0
        out["xmodel_v12_oof_class_count_mean"] = float(np.mean(aux_class_count)) if aux_class_count else 0.0
        out["xmodel_v12_oof_class_count_std"] = _safe_std(aux_class_count)
        if i % progress_every == 0:
            elapsed = (time.perf_counter() - t0) / 60.0
            eta = elapsed * (len(row_records) - i) / max(1, i)
            print(
                f"  v12-oof ... {i}/{len(row_records)} rows, aux predictions={n_aux_predictions} "
                f"({elapsed:.1f} min, ETA {eta:.1f} min)",
                flush=True,
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=ROOT / "paper" / "results" / "final_flag_multiseed_labels.csv")
    ap.add_argument("--base-features", type=Path, default=ROOT / "paper" / "results" / "final_flag_multiseed_features.csv")
    ap.add_argument(
        "--out-features",
        type=Path,
        default=ROOT / "paper" / "results" / "final_flag_multiseed_features_oof_heavy.csv",
    )
    ap.add_argument(
        "--out-heavy",
        type=Path,
        default=ROOT / "paper" / "results" / "final_flag_multiseed_oof_heavy_features_only.csv",
    )
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-v12", action="store_true")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--progress-every", type=int, default=100)
    args = ap.parse_args()

    labels = pd.read_csv(args.labels)
    base = pd.read_csv(args.base_features)
    if "row_id" not in labels.columns or "row_id" not in base.columns:
        raise SystemExit("labels and base features must include row_id")
    labels = labels.copy()
    if args.max_rows is not None:
        labels = labels.head(args.max_rows).copy()
        base = base[base["row_id"].isin(set(labels["row_id"]))].copy()
    merged = labels.merge(base, on=["row_id", "model_seed", "file"], how="left", suffixes=("", "_feature"))
    if len(merged) != len(labels):
        raise SystemExit("Feature merge changed row count")
    if merged.filter(regex=r"_feature$").shape[1]:
        pass
    row_records = merged.to_dict("records")
    qc_paths = _load_qc_paths()
    out_rows: dict[str, dict] = {str(r["row_id"]): {"row_id": str(r["row_id"])} for r in row_records}
    t0 = time.perf_counter()

    if not args.skip_baselines:
        print(f"Building out-of-fold baseline features for {len(row_records)} rows", flush=True)
        for seed in SEEDS:
            seed_rows = [r for r in row_records if int(r["model_seed"]) == seed]
            if not seed_rows:
                continue
            print(f"Seed {seed}: {len(seed_rows)} rows", flush=True)
            train, _test = _load_split_from_qc(seed)
            cache_dir = _baseline_cache_dir(seed)
            agg_rows = {str(r["row_id"]): _empty_baseline_agg() for r in seed_rows}
            for method in METHODS:
                print(f"  loading {method} from {_display(cache_dir / (method + '.pkl'))}", flush=True)
                predict_fn = predict_with_cache(
                    method,
                    train,
                    seed=seed,
                    cache_path=cache_dir / f"{method}.pkl",
                    force_retrain=False,
                )
                mt0 = time.perf_counter()
                _add_baseline_method_features(out_rows, agg_rows, seed_rows, qc_paths, method, predict_fn)
                print(f"  {method} done in {(time.perf_counter() - mt0) / 60.0:.1f} min", flush=True)
                del predict_fn
                gc.collect()
            _finalize_baseline_agg(out_rows, agg_rows)
            del agg_rows
            gc.collect()

    if not args.skip_v12:
        print(f"Building sparse out-of-fold multi-v12 features for {len(row_records)} rows", flush=True)
        seed_test_sets = {seed: _load_split_sets(seed)[1] for seed in SEEDS}
        _add_v12_oof_features(out_rows, row_records, qc_paths, seed_test_sets, args.progress_every)

    heavy = pd.DataFrame([out_rows[str(r["row_id"])] for r in row_records])
    args.out_heavy.parent.mkdir(parents=True, exist_ok=True)
    heavy.to_csv(args.out_heavy, index=False)
    final = base.merge(heavy, on="row_id", how="left")
    final.to_csv(args.out_features, index=False)
    summary = {
        "labels": _display(args.labels),
        "base_features": _display(args.base_features),
        "out_heavy": _display(args.out_heavy),
        "out_features": _display(args.out_features),
        "n_rows": int(len(final)),
        "n_heavy_columns": int(len([c for c in final.columns if c.startswith(("baseline_oof_", "xmodel_v12_oof_"))])),
        "skip_baselines": bool(args.skip_baselines),
        "skip_v12": bool(args.skip_v12),
        "elapsed_min": (time.perf_counter() - t0) / 60.0,
    }
    args.out_features.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
