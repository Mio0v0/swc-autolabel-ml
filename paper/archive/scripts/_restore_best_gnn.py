#!/usr/bin/env python3
"""After the iteration completes, restore the best-performing GNN config.

Reads paper/results/p10_iteration_log.json, finds the experiment with
the highest mean_p10. Then:
  * If best is "baseline" or an inference-only experiment (no retraining):
    restore the iter_baseline GNN backup
  * If best is a retraining experiment (focal loss variants):
    delete the current (last-trained) GNN and re-run that specific
    experiment to land its model on disk

Usage:
    python -m paper._restore_best_gnn
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path("D:/Desktop/SWC-Studio/.venv/Scripts/python.exe")

LOG = ROOT / "paper" / "results" / "p10_iteration_log.json"
SEEDS = [42, 789]


def main() -> int:
    if not LOG.is_file():
        print(f"MISSING: {LOG}")
        return 1
    d = json.loads(LOG.read_text(encoding="utf-8"))
    best_exp = d.get("best_experiment")
    best_p10 = d.get("best_p10")
    print(f"Best experiment recorded: {best_exp}  (mean_p10={best_p10:.4f})")

    # Find the experiment's full record (for env vars / retrain spec)
    history = d.get("history", [])
    rec = next((h for h in history if h.get("experiment") == best_exp), None)
    if rec is None:
        print(f"  no history entry — cannot determine if retraining needed")
        return 1
    retrain = rec.get("retrain", [])
    env     = rec.get("env", {})

    if not retrain:
        # Inference-only experiment — just restore the backup GNN
        print(f"  best is inference-only → restoring iter_baseline GNN for each seed")
        for seed in SEEDS:
            backup = ROOT / "paper" / "models" / f"v12_gentle_seed{seed}" / "gnn_apical_basal.iter_baseline.pt"
            live   = ROOT / "paper" / "models" / f"v12_gentle_seed{seed}" / "gnn_apical_basal.pt"
            if not backup.is_file():
                print(f"    WARN: backup missing for seed={seed} — leaving current GNN in place")
                continue
            shutil.copy2(backup, live)
            print(f"    restored seed={seed}")
        print("done")
        return 0

    # Retraining experiment — re-train with that experiment's env
    print(f"  best requires retraining GNN with env: {env}")
    for seed in SEEDS:
        live = ROOT / "paper" / "models" / f"v12_gentle_seed{seed}" / "gnn_apical_basal.pt"
        if live.is_file():
            live.unlink()
            print(f"    deleted seed={seed}/gnn_apical_basal.pt to force retrain")
        # Build env
        env_full = os.environ.copy()
        for k, v in env.items():
            if v is None:
                env_full.pop(k, None)
            else:
                env_full[k] = str(v)
        cmd = [str(PYTHON), "-u", "-m", "paper._retrain_v12_gentle_seed", "--seed", str(seed)]
        print(f"    retraining seed={seed} ...")
        rc = subprocess.call(cmd, env=env_full, cwd=str(ROOT))
        if rc != 0:
            print(f"    seed={seed} retrain FAILED rc={rc}")
            return rc
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
