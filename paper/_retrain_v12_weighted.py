#!/usr/bin/env python3
"""Retrain v12 Stage 2 + GNN with aggressive class weighting.

Steps:
  1. Set env vars:
        SWCAL_CLASS_BALANCE_POWER=1.5    (Stage 2: over-correct toward
                                          minority classes — basal/apical
                                          vs. axon)
        SWCAL_GNN_CLASS_WEIGHT=inverse   (GNN: inverse-frequency weighted
                                          cross-entropy — apical vs basal)
  2. Re-train Stage 2 on the same QC-passed v12 training split.
  3. Re-train the GNN apical/basal head on the same split.
  4. Save everything to paper/models/v12_weighted/ (the v12 baseline at
     paper/models/v12/ is left untouched for direct comparison).
  5. Copy the v12 Stage 1 model + QC gate over (unchanged) so the
     v12_weighted folder is a complete drop-in replacement.

Wall time: ~70-90 min on a 4080 GPU (Stage 2 ~40 min, GNN ~30 min).
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Set env vars BEFORE any module that consumes them imports.
os.environ["SWCAL_CLASS_BALANCE_POWER"] = "1.5"
os.environ["SWCAL_GNN_CLASS_WEIGHT"]    = "inverse"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.evaluate import _train_stage2, _file_in_test_bucket             # noqa: E402

# Pinned paths
QC_CSV       = ROOT / "paper" / "results" / "corpus_qc_v12_uncurated.csv"
V12_DIR      = ROOT / "paper" / "models" / "v12"
OUT_DIR      = ROOT / "paper" / "models" / "v12_weighted"
SPLIT_JSON   = OUT_DIR / "train_test_split.json"
S2_OUT       = OUT_DIR / "branch_classifier.pkl"
GNN_OUT      = OUT_DIR / "gnn_apical_basal.pt"
METRICS      = OUT_DIR / "training_metrics.json"
PYTHON       = Path("D:/Desktop/SWC-Studio/.venv/Scripts/python.exe")

SEED      = 42
TEST_SIZE = 0.20


def _load_qc_passed() -> dict:
    out = {"pyramidal": [], "interneuron": []}
    with QC_CSV.open("r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("qc_pass", "").strip().lower() not in ("true", "1"):
                continue
            ct = r["cell_type"]
            p = Path(r["path"])
            if p.is_file():
                out.setdefault(ct, []).append(p)
    return out


def _hash_split(qc_passed, seed, test_size):
    train, test = {}, {}
    for ct, paths in qc_passed.items():
        train[ct] = []
        test[ct] = []
        for p in sorted(paths):
            (test if _file_in_test_bucket(p.name, seed, test_size) else train)[ct].append(p)
    return train, test


def _save_split(train, test):
    payload = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "train": {ct: sorted(p.name for p in paths) for ct, paths in train.items()},
        "test":  {ct: sorted(p.name for p in paths) for ct, paths in test.items()},
        "counts": {
            "train": {ct: len(paths) for ct, paths in train.items()},
            "test":  {ct: len(paths) for ct, paths in test.items()},
        },
    }
    SPLIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_unchanged_artifacts():
    """Stage 1 + QC gate are not affected by the class-weighting changes;
    copy them verbatim from v12/ so v12_weighted/ is a complete model dir."""
    for fn in ("cell_type_classifier.pkl", "qc_gate.pkl"):
        src = V12_DIR / fn
        dst = OUT_DIR / fn
        if src.is_file():
            shutil.copy2(src, dst)
            print(f"  copied {fn} from v12 baseline")
    # Also copy the eval_split_for_gnn JSON
    src_split = V12_DIR / "eval_split_for_gnn.json"
    if src_split.is_file():
        shutil.copy2(src_split, OUT_DIR / "eval_split_for_gnn.json")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overall_t0 = time.perf_counter()

    print("=== v12 RETRAIN with class weighting ===")
    print(f"  SWCAL_CLASS_BALANCE_POWER = {os.environ['SWCAL_CLASS_BALANCE_POWER']}  (Stage 2)")
    print(f"  SWCAL_GNN_CLASS_WEIGHT    = {os.environ['SWCAL_GNN_CLASS_WEIGHT']}     (GNN)")
    print(f"  OUT_DIR                   = {OUT_DIR}")
    print()

    qc = _load_qc_passed()
    train, test = _hash_split(qc, SEED, TEST_SIZE)
    for ct in ("pyramidal", "interneuron"):
        print(f"  {ct:12s}  train={len(train.get(ct,[]))}  test={len(test.get(ct,[]))}")
    _save_split(train, test)
    print(f"\nSaved split -> {SPLIT_JSON}")

    # --- Stage 2 with power=1.5 ---
    print(f"\n[1] Re-training Stage 2 with SWCAL_CLASS_BALANCE_POWER=1.5")
    print(f"    -> {S2_OUT}")
    t0 = time.perf_counter()
    _train_stage2(train, S2_OUT)
    s2_min = (time.perf_counter() - t0) / 60.0
    print(f"    Stage 2 elapsed: {s2_min:.1f} min")

    # --- GNN with class weighting ---
    print(f"\n[2] Re-training GNN with SWCAL_GNN_CLASS_WEIGHT=inverse")
    print(f"    -> {GNN_OUT}")
    t0 = time.perf_counter()
    # Run as subprocess so we pick up the same env vars
    env = os.environ.copy()
    cmd = [
        str(PYTHON), "-u", "-m", "paper.gnn_apical_basal",
        "--data-dir", str(ROOT / "data" / "v12_uncurated"),
        "--eval-split", str(V12_DIR / "eval_split_for_gnn.json"),
        "--ckpt", str(GNN_OUT),
    ]
    print(f"    Command: {' '.join(cmd)}")
    rc = subprocess.call(cmd, env=env, cwd=str(ROOT))
    gnn_min = (time.perf_counter() - t0) / 60.0
    print(f"    GNN elapsed: {gnn_min:.1f} min  (rc={rc})")
    if rc != 0:
        print("    GNN training failed; aborting.")
        return rc

    # --- Copy unchanged artifacts ---
    print(f"\n[3] Copying unchanged Stage 1 + QC gate")
    _copy_unchanged_artifacts()

    # --- Metrics ---
    total_min = (time.perf_counter() - overall_t0) / 60.0
    metrics = {
        "seed": SEED, "test_size": TEST_SIZE,
        "swcal_class_balance_power": os.environ["SWCAL_CLASS_BALANCE_POWER"],
        "swcal_gnn_class_weight":    os.environ["SWCAL_GNN_CLASS_WEIGHT"],
        "n_train": {ct: len(paths) for ct, paths in train.items()},
        "n_test":  {ct: len(paths) for ct, paths in test.items()},
        "stage2_train_min": s2_min,
        "gnn_train_min":    gnn_min,
        "total_min":        total_min,
    }
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nTOTAL ELAPSED: {total_min:.1f} min")
    print(f"Models -> {OUT_DIR}")
    print(f"Metrics -> {METRICS}")
    print()
    print("Next steps:")
    print("  1. Re-run inference + calibration on the v12 test set with the")
    print("     weighted models (use _calibrate_flags with explicit paths):")
    print(f"     python -m paper._calibrate_flags --gnn-model {GNN_OUT}")
    print("  2. Compare v12 vs v12_weighted per-class F1 + per-cell P10.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
