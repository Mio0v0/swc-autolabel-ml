#!/usr/bin/env python3
"""Canonical v12_gentle seed trainer — handles any seed identically.

Each call produces ONE complete model at `paper/models/v12_gentle_seed<N>/`:
    cell_type_classifier.pkl   Stage 1 (XGBoost RF, GPU)
    branch_classifier.pkl      Stage 2 (XGBoost RF, GPU, class-weighted)
    gnn_apical_basal.pt        Stage 3 (GraphSAGE, class-weighted CE)
    qc_gate.pkl                Structural QC + OOD detector
    train_test_split.json      The seed's hash-bucket split (for eval)
    eval_split_for_gnn.json    GNN-format eval split
    training_metrics.json      Timing + counts

Identical procedure across all seeds; only the split seed differs. This is
THE script for building the seed ensemble. Run it once per seed:

    python -m paper._retrain_v12_gentle_seed --seed 42
    python -m paper._retrain_v12_gentle_seed --seed 123
    python -m paper._retrain_v12_gentle_seed --seed 456
    python -m paper._retrain_v12_gentle_seed --seed 789

Gentle config (Stage 2 power=1.25, GNN inverse_sqrt class weighting) is
pinned via env vars BEFORE imports.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Set env vars BEFORE imports — pinned config for the ensemble.
os.environ["SWCAL_CLASS_BALANCE_POWER"] = "1.25"
os.environ["SWCAL_GNN_CLASS_WEIGHT"]    = "inverse_sqrt"
# Phase 1 winner: focal-loss GNN gamma=2 (+1.26 pp on P10). Adopted as new
# default. Override via env var SWCAL_GNN_FOCAL_GAMMA to experiment.
os.environ.setdefault("SWCAL_GNN_FOCAL_GAMMA", "2.0")
# SWCAL_SAMPLE_WEIGHT left at its default ("node") — cell-equal was tried in
# iter-1 and rolled back (hurt P10 by 2-5 pp).

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.evaluate import (                                           # noqa: E402
    _train_stage1, _train_stage2, _file_in_test_bucket,
)
from hybrid.features import parse_swc, extract_feature_vector, FEATURE_NAMES  # noqa: E402
from hybrid.qc_input import OODDetector, QCGate                         # noqa: E402

# SWCAL_QC_CSV overrides the QC CSV (use for cleaned-dataset experiments).
QC_CSV    = Path(os.environ.get("SWCAL_QC_CSV", str(
    ROOT / "paper" / "results" / "corpus_qc_v12_uncurated.csv"
)))
# SWCAL_MODEL_DIR_SUFFIX appends to the model output dir name
# (e.g. "_cleaned" produces "v12_gentle_seed42_cleaned/").
MODEL_DIR_SUFFIX = os.environ.get("SWCAL_MODEL_DIR_SUFFIX", "")
PYTHON    = Path("D:/Desktop/SWC-Studio/.venv/Scripts/python.exe")

TEST_SIZE    = 0.20
OOD_QUANTILE = 0.99


def _load_qc_passed() -> dict[str, list[Path]]:
    if not QC_CSV.is_file():
        raise FileNotFoundError(
            f"{QC_CSV} not found — run `python -m paper._scan_corpus_qc` first."
        )
    out: dict[str, list[Path]] = {"pyramidal": [], "interneuron": []}
    n_total = n_pass = 0
    with QC_CSV.open("r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            n_total += 1
            if r.get("qc_pass", "").strip().lower() not in ("true", "1"):
                continue
            n_pass += 1
            ct = r["cell_type"]
            p = Path(r["path"])
            if p.is_file():
                out.setdefault(ct, []).append(p)
    print(f"QC CSV: {n_total} files, {n_pass} passed QC")
    for ct, paths in out.items():
        print(f"  {ct}: {len(paths)}")
    return out


def _hash_split(qc_passed, seed, test_size):
    train, test = {}, {}
    for ct, paths in qc_passed.items():
        train[ct] = []
        test[ct]  = []
        for p in sorted(paths):
            (test if _file_in_test_bucket(p.name, seed, test_size) else train)[ct].append(p)
    return train, test


def _save_split(out_dir, train, test, seed, test_size):
    payload = {
        "seed": seed,
        "test_size": test_size,
        "train": {ct: sorted(p.name for p in paths) for ct, paths in train.items()},
        "test":  {ct: sorted(p.name for p in paths) for ct, paths in test.items()},
        "counts": {
            "train": {ct: len(paths) for ct, paths in train.items()},
            "test":  {ct: len(paths) for ct, paths in test.items()},
        },
    }
    (out_dir / "train_test_split.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )
    (out_dir / "eval_split_for_gnn.json").write_text(json.dumps({
        "seed": seed, "test_size": test_size,
        "train_files": payload["train"],
        "test_files": payload["test"],
    }, indent=2), encoding="utf-8")


def _fit_ood(train_files):
    print("\nFitting OOD detector on training Stage-1 features...")
    t0 = time.perf_counter()
    feats: list[np.ndarray] = []
    n_failed = 0
    n_total = sum(len(v) for v in train_files.values())
    done = 0
    for ct, paths in train_files.items():
        for p in paths:
            done += 1
            try:
                nodes = parse_swc(p)
                if not nodes:
                    n_failed += 1; continue
                feats.append(np.asarray(extract_feature_vector(nodes), dtype=np.float64))
            except Exception:
                n_failed += 1
            if done % 1000 == 0:
                print(f"  ... {done}/{n_total}")
    X = np.vstack(feats)
    ood = OODDetector.fit(X, FEATURE_NAMES, quantile=OOD_QUANTILE)
    print(f"  fit {ood.n_train} cells in {(time.perf_counter()-t0)/60:.1f} min "
          f"(skipped {n_failed}); threshold={ood.threshold:.3f}")
    return ood


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True,
                    help="Split seed (and GNN training seed).")
    ap.add_argument("--force-retrain", action="store_true",
                    help="Re-train every stage even if its .pkl/.pt is already on disk. "
                         "Use after changing model hyperparameters/algorithm or QC corpus.")
    ap.add_argument("--test-size", type=float, default=None,
                    help="Override the default 0.20 test fraction (e.g. 0.5 for 50/50 split).")
    ap.add_argument("--suffix", type=str, default="",
                    help="Extra suffix on the output dir name, e.g. '_t05' for 50/50 split.")
    args = ap.parse_args()
    seed = args.seed
    force = args.force_retrain
    test_size = args.test_size if args.test_size is not None else TEST_SIZE

    out_dir = ROOT / "paper" / "models" / f"v12_gentle_seed{seed}{MODEL_DIR_SUFFIX}{args.suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    overall_t0 = time.perf_counter()

    print(f"=== v12_gentle seed={seed} ===")
    print(f"  SWCAL_CLASS_BALANCE_POWER = {os.environ['SWCAL_CLASS_BALANCE_POWER']}")
    print(f"  SWCAL_GNN_CLASS_WEIGHT    = {os.environ['SWCAL_GNN_CLASS_WEIGHT']}")
    print(f"  OUT_DIR                   = {out_dir}")
    print(f"  --force-retrain           = {force}")
    print()

    qc = _load_qc_passed()
    print(f"  test_size                 = {test_size}")
    train, test = _hash_split(qc, seed, test_size)
    for ct in ("pyramidal", "interneuron"):
        print(f"  {ct:12s}  train={len(train.get(ct,[]))}  test={len(test.get(ct,[]))}")
    _save_split(out_dir, train, test, seed, test_size)
    print(f"  split saved -> {out_dir / 'train_test_split.json'}")

    # [1] Stage 1
    s1_out = out_dir / "cell_type_classifier.pkl"
    if s1_out.is_file() and not force:
        print(f"\n[1] Stage 1 CACHED — skip ({s1_out.name})")
        s1_min = 0.0
    else:
        print(f"\n[1] Stage 1 train (cell-type RF)")
        t0 = time.perf_counter()
        _train_stage1(train, s1_out)
        s1_min = (time.perf_counter() - t0) / 60.0
        print(f"    Stage 1 elapsed: {s1_min:.1f} min")

    # [2] Stage 2 (gentle config from env vars)
    s2_out = out_dir / "branch_classifier.pkl"
    if s2_out.is_file() and not force:
        print(f"\n[2] Stage 2 CACHED — skip ({s2_out.name})")
        s2_min = 0.0
    else:
        print(f"\n[2] Stage 2 train (per-branch RF, SWCAL_CLASS_BALANCE_POWER=1.25)")
        t0 = time.perf_counter()
        _train_stage2(train, s2_out)
        s2_min = (time.perf_counter() - t0) / 60.0
        print(f"    Stage 2 elapsed: {s2_min:.1f} min")

    # [3] GNN (gentle config from env vars)
    gnn_out = out_dir / "gnn_apical_basal.pt"
    if gnn_out.is_file() and not force:
        print(f"\n[3] GNN CACHED — skip ({gnn_out.name})")
        gnn_min = 0.0
        rc = 0
    else:
        gnn_type = os.environ.get("SWCAL_GNN_ARCH", "sage").lower()
        print(f"\n[3] GNN train (SWCAL_GNN_CLASS_WEIGHT=inverse_sqrt, "
              f"--seed {seed}, --gnn-type {gnn_type})")
        t0 = time.perf_counter()
        env = os.environ.copy()
        cmd = [
            str(PYTHON), "-u", "-m", "paper.gnn_apical_basal",
            "--data-dir", str(ROOT / "data" / "v12_uncurated"),
            "--eval-split", str(out_dir / "eval_split_for_gnn.json"),
            "--ckpt", str(gnn_out),
            "--seed", str(seed),
            "--gnn-type", gnn_type,
        ]
        print(f"    Command: {' '.join(cmd)}")
        rc = subprocess.call(cmd, env=env, cwd=str(ROOT))
        gnn_min = (time.perf_counter() - t0) / 60.0
        print(f"    GNN elapsed: {gnn_min:.1f} min  (rc={rc})")
        if rc != 0:
            return rc

    # [4] OOD + QC gate
    qc_gate_out = out_dir / "qc_gate.pkl"
    if qc_gate_out.is_file() and not force:
        print(f"\n[4] OOD/QC gate CACHED — skip ({qc_gate_out.name})")
        ood_info = {"n_train": 0, "threshold": 0.0}
    else:
        print(f"\n[4] Fit OOD detector + save QC gate")
        ood = _fit_ood(train)
        gate = QCGate(ood_detector=ood)
        gate.save(qc_gate_out)
        ood_info = {"n_train": ood.n_train, "threshold": ood.threshold}
        print(f"    QC gate saved -> {qc_gate_out}")

    total_min = (time.perf_counter() - overall_t0) / 60.0
    (out_dir / "training_metrics.json").write_text(json.dumps({
        "seed":            seed,
        "test_size":       test_size,
        "swcal_class_balance_power": os.environ["SWCAL_CLASS_BALANCE_POWER"],
        "swcal_gnn_class_weight":    os.environ["SWCAL_GNN_CLASS_WEIGHT"],
        "n_train":         {ct: len(paths) for ct, paths in train.items()},
        "n_test":          {ct: len(paths) for ct, paths in test.items()},
        "stage1_train_min": s1_min,
        "stage2_train_min": s2_min,
        "gnn_train_min":    gnn_min,
        "ood_n_train":      ood_info["n_train"],
        "ood_threshold":    ood_info["threshold"],
        "ood_quantile":     OOD_QUANTILE,
        "total_min":        total_min,
    }, indent=2), encoding="utf-8")
    print(f"\nTOTAL ELAPSED: {total_min:.1f} min")
    print(f"Model -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
