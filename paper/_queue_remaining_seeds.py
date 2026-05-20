#!/usr/bin/env python3
"""Sequential retrain queue for seeds 456 and 789.

Designed to be launched AFTER paper._retrain_v12_gentle_seed --seed 123
finishes. Runs both remaining seeds back-to-back via subprocess; total
wall time ~170 min.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path("D:/Desktop/SWC-Studio/.venv/Scripts/python.exe")

SEEDS = [456, 789]

t0 = time.perf_counter()
for seed in SEEDS:
    print(f"\n{'='*72}\nLaunching seed {seed}...\n{'='*72}")
    rc = subprocess.call(
        [str(PYTHON), "-u", "-m", "paper._retrain_v12_gentle_seed", "--seed", str(seed)],
        cwd=str(ROOT),
    )
    print(f"  seed {seed}: rc={rc}  (cumulative {(time.perf_counter()-t0)/60:.1f} min)")
    if rc != 0:
        print(f"  FAILED, aborting queue.")
        sys.exit(rc)
print(f"\nQueue complete in {(time.perf_counter()-t0)/60:.1f} min")
