#!/usr/bin/env python3
"""Backup seed=42 and seed=789 GNN models before any retraining experiment.

Run BEFORE the iteration framework's experiment 5 (the first retraining
experiment). After the iteration completes, paper/_restore_best_gnn.py
selects whichever config the iteration log records as best-performing
and restores or retrains accordingly.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for seed in (42, 789):
    src = ROOT / "paper" / "models" / f"v12_gentle_seed{seed}" / "gnn_apical_basal.pt"
    dst = ROOT / "paper" / "models" / f"v12_gentle_seed{seed}" / "gnn_apical_basal.iter_baseline.pt"
    if not src.is_file():
        print(f"  WARN: {src} missing — cannot back up")
        continue
    shutil.copy2(src, dst)
    print(f"  backup: {src.name} -> {dst.name}  ({dst.stat().st_size/1024:.1f} KB)")

print("done")
