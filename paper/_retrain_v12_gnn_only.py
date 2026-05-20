#!/usr/bin/env python3
"""Retrain ONLY the GNN with inverse_sqrt class weighting; reuse v12 baseline Stage 1+2.

Hypothesis: most of the gentle-variant's gain might come from the GNN
class weighting alone — Stage 2 power=1.25 may add little. This variant
isolates that question by retraining only the GNN (~30 min) and reusing
the v12 baseline Stage 1+2 models unchanged.

Env vars:
    SWCAL_GNN_CLASS_WEIGHT = inverse_sqrt
    SWCAL_CLASS_BALANCE_POWER = 1.0 (default — no Stage 2 change)

Outputs to paper/models/v12_gnn_only/, with Stage 1+2+QC-gate copied
verbatim from paper/models/v12/.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Set env var BEFORE imports — same pattern as the other retrain scripts.
os.environ["SWCAL_GNN_CLASS_WEIGHT"]    = "inverse_sqrt"
# Explicitly DON'T set SWCAL_CLASS_BALANCE_POWER → defaults to 1.0 (no change)

ROOT = Path(__file__).resolve().parent.parent

V12_DIR  = ROOT / "paper" / "models" / "v12"
OUT_DIR  = ROOT / "paper" / "models" / "v12_gnn_only"
GNN_OUT  = OUT_DIR / "gnn_apical_basal.pt"
METRICS  = OUT_DIR / "training_metrics.json"
PYTHON   = Path("D:/Desktop/SWC-Studio/.venv/Scripts/python.exe")


def _copy_unchanged_artifacts():
    """Stage 1, Stage 2, QC gate, and split are all reused from v12 baseline."""
    for fn in (
        "cell_type_classifier.pkl",
        "branch_classifier.pkl",
        "qc_gate.pkl",
        "train_test_split.json",
        "eval_split_for_gnn.json",
    ):
        src = V12_DIR / fn
        dst = OUT_DIR / fn
        if src.is_file():
            shutil.copy2(src, dst)
            print(f"  copied {fn} from v12 baseline")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overall_t0 = time.perf_counter()

    print("=== v12 RETRAIN: GNN-only with inverse_sqrt class weighting ===")
    print(f"  SWCAL_GNN_CLASS_WEIGHT    = {os.environ['SWCAL_GNN_CLASS_WEIGHT']}")
    print(f"  Stage 2 is REUSED from {V12_DIR} (no class-balance-power change)")
    print(f"  OUT_DIR                   = {OUT_DIR}")
    print()

    print("[1] Copy unchanged Stage 1 + Stage 2 + QC gate + split from v12 baseline")
    _copy_unchanged_artifacts()

    print(f"\n[2] Re-training GNN with SWCAL_GNN_CLASS_WEIGHT=inverse_sqrt")
    t0 = time.perf_counter()
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
        return rc

    total_min = (time.perf_counter() - overall_t0) / 60.0
    metrics = {
        "swcal_gnn_class_weight":     os.environ["SWCAL_GNN_CLASS_WEIGHT"],
        "swcal_class_balance_power":  "1.0 (default, Stage 2 not retrained)",
        "stage2_source":              str(V12_DIR / "branch_classifier.pkl"),
        "gnn_train_min":              gnn_min,
        "total_min":                  total_min,
    }
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nTOTAL ELAPSED: {total_min:.1f} min")
    print(f"Models -> {OUT_DIR}")
    print()
    print("Next: compare to baseline + gentle:")
    print(f"  python -m paper._compare_v12_weighted --model-dir {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
