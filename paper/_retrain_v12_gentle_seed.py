#!/usr/bin/env python3
"""Train one v12_weighted_gentle model at a custom split seed.

Same gentle config (Stage 2 power=1.25, GNN inverse_sqrt) as
_retrain_v12_weighted_gentle.py, but the split seed is exposed via
CLI. Used to build a seed ensemble:

    python -m paper._retrain_v12_gentle_seed --seed 123
    python -m paper._retrain_v12_gentle_seed --seed 456
    python -m paper._retrain_v12_gentle_seed --seed 789

Each run saves to paper/models/v12_gentle_seed<N>/.

Note: changing the seed changes the train/test partition — each model
sees a different subset of cells in training. For inference on a fresh
input, all N models predict, and we average their per-node softmax;
their disagreement is the natural confidence signal for the flag layer.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Set env vars BEFORE imports
os.environ["SWCAL_CLASS_BALANCE_POWER"] = "1.25"
os.environ["SWCAL_GNN_CLASS_WEIGHT"]    = "inverse_sqrt"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.evaluate import _train_stage2, _file_in_test_bucket   # noqa: E402

QC_CSV = ROOT / "paper" / "results" / "corpus_qc_v12_uncurated.csv"
V12_DIR = ROOT / "paper" / "models" / "v12"
PYTHON  = Path("D:/Desktop/SWC-Studio/.venv/Scripts/python.exe")

TEST_SIZE = 0.20


def _load_qc_passed() -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {"pyramidal": [], "interneuron": []}
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


def _save_split(out_dir: Path, train, test, seed: int):
    payload = {
        "seed": seed,
        "test_size": TEST_SIZE,
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
    # Also write the GNN-format eval_split.json
    gnn_split = {
        "seed":      seed,
        "test_size": TEST_SIZE,
        "test_files": payload["test"],
    }
    (out_dir / "eval_split_for_gnn.json").write_text(
        json.dumps(gnn_split, indent=2), encoding="utf-8",
    )


def _copy_unchanged_artifacts(out_dir: Path):
    """Stage 1 + QC gate from v12 baseline — these don't depend on seed
    in a meaningful way for the ensemble's purpose (the ensemble only
    averages Stage 2 + Stage 3 outputs; Stage 1 just routes)."""
    for fn in ("cell_type_classifier.pkl", "qc_gate.pkl"):
        src = V12_DIR / fn
        dst = out_dir / fn
        if src.is_file():
            shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True,
                    help="Split seed (and GNN training seed).")
    args = ap.parse_args()
    seed = args.seed

    out_dir = ROOT / "paper" / "models" / f"v12_gentle_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    overall_t0 = time.perf_counter()

    print(f"=== v12_gentle seed={seed} ===")
    print(f"  SWCAL_CLASS_BALANCE_POWER = {os.environ['SWCAL_CLASS_BALANCE_POWER']}")
    print(f"  SWCAL_GNN_CLASS_WEIGHT    = {os.environ['SWCAL_GNN_CLASS_WEIGHT']}")
    print(f"  OUT_DIR                   = {out_dir}")
    print()

    qc = _load_qc_passed()
    train, test = _hash_split(qc, seed, TEST_SIZE)
    for ct in ("pyramidal", "interneuron"):
        print(f"  {ct:12s}  train={len(train.get(ct,[]))}  test={len(test.get(ct,[]))}")
    _save_split(out_dir, train, test, seed)
    print(f"\n  split saved to {out_dir / 'train_test_split.json'}")

    print(f"\n[1] Stage 2 train (SWCAL_CLASS_BALANCE_POWER=1.25)")
    s2_out = out_dir / "branch_classifier.pkl"
    t0 = time.perf_counter()
    _train_stage2(train, s2_out)
    s2_min = (time.perf_counter() - t0) / 60.0
    print(f"    Stage 2 elapsed: {s2_min:.1f} min")

    print(f"\n[2] GNN train (SWCAL_GNN_CLASS_WEIGHT=inverse_sqrt, --seed {seed})")
    gnn_out = out_dir / "gnn_apical_basal.pt"
    t0 = time.perf_counter()
    env = os.environ.copy()
    cmd = [
        str(PYTHON), "-u", "-m", "paper.gnn_apical_basal",
        "--data-dir", str(ROOT / "data" / "v12_uncurated"),
        "--eval-split", str(out_dir / "eval_split_for_gnn.json"),
        "--ckpt", str(gnn_out),
        "--seed", str(seed),
    ]
    print(f"    Command: {' '.join(cmd)}")
    rc = subprocess.call(cmd, env=env, cwd=str(ROOT))
    gnn_min = (time.perf_counter() - t0) / 60.0
    print(f"    GNN elapsed: {gnn_min:.1f} min  (rc={rc})")
    if rc != 0:
        return rc

    print(f"\n[3] Copy Stage 1 + QC gate from v12 baseline")
    _copy_unchanged_artifacts(out_dir)

    total_min = (time.perf_counter() - overall_t0) / 60.0
    (out_dir / "training_metrics.json").write_text(json.dumps({
        "seed": seed,
        "test_size": TEST_SIZE,
        "swcal_class_balance_power": os.environ["SWCAL_CLASS_BALANCE_POWER"],
        "swcal_gnn_class_weight":    os.environ["SWCAL_GNN_CLASS_WEIGHT"],
        "n_train": {ct: len(paths) for ct, paths in train.items()},
        "n_test":  {ct: len(paths) for ct, paths in test.items()},
        "stage2_train_min": s2_min,
        "gnn_train_min":    gnn_min,
        "total_min":        total_min,
    }, indent=2), encoding="utf-8")
    print(f"\nTOTAL ELAPSED: {total_min:.1f} min")
    print(f"Models -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
