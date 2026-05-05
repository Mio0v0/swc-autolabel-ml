#!/usr/bin/env python3
"""Feature ablation + multi-seed stability runner.

Generates the ablation rows reviewers expect for the algorithm paper:

    no-pca              Drop PCA principal-axis features from Stage 1+2
    no-soft-handoff     Force a hard Stage-1 → Stage-2 cascade (threshold = 0)
    no-trunk            Drop the 4 trunk-detection features from Stage 2
    no-subtree-stage2   Run Stage 2 per-branch (v7) instead of per-subtree (v8)
    no-gnn              Pipeline without the Stage 2b GraphSAGE GNN

Plus a multi-seed stability check on the full v9 pipeline:

    seed=42   (production seed)
    seed=123
    seed=456

Each ablation requires retraining Stages 1+2 (~2-4 hr per run on CPU,
~30 min on GPU) and rerunning evaluate.py with the matching flags. The
GNN ablation can reuse cached models, so it's the cheapest one. Plan
on roughly:

    no-pca             ~3 hr  (full retrain, no PCA features)
    no-soft-handoff    ~5 min (cached models, just re-evaluate)
    no-trunk           ~3 hr  (full retrain, 4 features dropped)
    no-subtree-stage2  ~5 min (already have v7 snapshot)
    no-gnn             ~5 min (cached models, just re-evaluate)
    multi-seed (×3)    ~9 hr  (3 full retrains)

Total: ~20-25 hr of compute for the full ablation grid. Recommend
running each ablation separately and snapshotting between runs.

Inputs / outputs
----------------
The runner shells out to ``hybrid.evaluate`` with new flags
(``--no-pca``, ``--no-soft-handoff``, ``--no-trunk``) which must be
added to ``hybrid/evaluate.py`` and threaded through into
``hybrid/features.py``, ``hybrid/branch_features.py``, and
``hybrid/pipeline.py``. The first part of this file documents what
the engine-side patches need to look like; the second part runs the
matrix.

Usage
-----
    python -m paper.run_ablations all
    python -m paper.run_ablations no-pca
    python -m paper.run_ablations multi-seed
    python -m paper.run_ablations multi-seed --seeds 42,123,456

Outputs land in:
    paper/results/snapshots/ablation_<name>.json
    paper/results/snapshots/ablation_<name>_per_file.csv
    paper/results/snapshots/multi_seed_<seed>.json
    paper/results/snapshots/multi_seed_<seed>_per_file.csv
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = ROOT / "paper" / "results" / "snapshots"
DEFAULT_DATA_DIR = ROOT / "data" / "v9_merged_dataset"


# =============================================================================
# Engine-side patches required before this runner works.
#
# Add the following CLI flags to hybrid/evaluate.py main():
#
#     parser.add_argument(
#         "--no-pca", action="store_true",
#         help="Disable PCA principal-axis features (ablation row).",
#     )
#     parser.add_argument(
#         "--no-soft-handoff", action="store_true",
#         help="Force hard Stage 1 -> Stage 2 cascade (no soft handoff).",
#     )
#     parser.add_argument(
#         "--no-trunk", action="store_true",
#         help="Drop the 4 trunk-detection features from Stage 2.",
#     )
#
# Then thread them as feature-mask environment variables into the feature
# extractors. In hybrid/features.py and hybrid/branch_features.py, at the
# top of each module:
#
#     _DISABLE_PCA = os.environ.get("SWCAL_NO_PCA") == "1"
#     _DISABLE_TRUNK = os.environ.get("SWCAL_NO_TRUNK") == "1"
#
# and in the feature-extraction loop, skip the corresponding columns when
# the flag is set.
#
# In hybrid/pipeline.py:
#     soft_handoff_threshold = 0.0 if os.environ.get("SWCAL_NO_SOFT_HANDOFF") == "1" else DEFAULT_SOFT_HANDOFF_THRESHOLD
# =============================================================================


def _run(cmd: list[str], env_overrides: dict[str, str] | None = None) -> int:
    """Run a subprocess with optional environment overrides."""
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    print("+ " + " ".join(cmd))
    if env_overrides:
        print("  env: " + ", ".join(f"{k}={v}" for k, v in env_overrides.items()))
    return subprocess.call(cmd, env=env, cwd=ROOT)


def _eval_cmd(seed: int, data_dir: Path, extra_args: list[str]) -> list[str]:
    return [
        sys.executable, "-m", "hybrid.evaluate",
        "--data-dir", str(data_dir),
        "--seed", str(seed),
        "--use-gnn",
        "--use-subtree-stage2",
        *extra_args,
    ]


# =============================================================================
# Ablation cases
# =============================================================================

ABLATIONS: dict[str, dict] = {
    "no-pca": {
        "description": "Drop PCA principal-axis features from Stages 1+2 (full retrain).",
        "env": {"SWCAL_NO_PCA": "1"},
        "extra_args": [],
        "estimated_hours": 3,
    },
    "no-soft-handoff": {
        "description": "Force hard Stage 1 -> Stage 2 cascade (cached models, ~5 min).",
        "env": {"SWCAL_NO_SOFT_HANDOFF": "1"},
        "extra_args": [],
        "estimated_hours": 0.1,
    },
    "no-trunk": {
        "description": "Drop the 4 trunk-detection features from Stage 2 (full retrain).",
        "env": {"SWCAL_NO_TRUNK": "1"},
        "extra_args": [],
        "estimated_hours": 3,
    },
    "no-subtree-stage2": {
        "description": "Per-branch Stage 2 (v7) instead of per-subtree (v8/v9).",
        "env": {},
        "extra_args": [],  # Will drop --use-subtree-stage2 from the eval cmd.
        "estimated_hours": 0.1,
    },
    "no-gnn": {
        "description": "Pipeline without the Stage 2b GraphSAGE GNN.",
        "env": {},
        "extra_args": [],  # Will drop --use-gnn from the eval cmd.
        "estimated_hours": 0.1,
    },
}


def run_ablation(name: str, data_dir: Path, seed: int) -> int:
    if name not in ABLATIONS:
        print(f"Unknown ablation: {name}", file=sys.stderr)
        return 2
    spec = ABLATIONS[name]
    print(f"\n=== Ablation: {name} ===")
    print(f"  {spec['description']}")
    print(f"  estimated {spec['estimated_hours']} hr on CPU")

    base = _eval_cmd(seed, data_dir, spec["extra_args"])
    if name == "no-subtree-stage2":
        base = [a for a in base if a != "--use-subtree-stage2"]
    if name == "no-gnn":
        base = [a for a in base if a != "--use-gnn"]
    return _run(base, spec["env"])


def run_multi_seed(seeds: list[int], data_dir: Path) -> int:
    rc = 0
    for seed in seeds:
        print(f"\n=== Multi-seed: seed={seed} ===")
        cmd = _eval_cmd(seed, data_dir, [])
        seed_rc = _run(cmd)
        if seed_rc != 0:
            print(f"  seed={seed} returned {seed_rc}", file=sys.stderr)
            rc = seed_rc
    return rc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target", nargs="?", default="all",
        help=(
            "Which ablation/seed bundle to run. One of: "
            + ", ".join(["all", "multi-seed"] + list(ABLATIONS.keys()))
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds", default="42,123,456",
        help="Comma-separated seed list for `multi-seed` (default 42,123,456).",
    )
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"ERROR: data dir not found: {args.data_dir}", file=sys.stderr)
        return 2

    if args.target == "multi-seed":
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        return run_multi_seed(seeds, args.data_dir)

    if args.target == "all":
        rc = 0
        for name in ABLATIONS:
            r = run_ablation(name, args.data_dir, args.seed)
            if r != 0:
                rc = r
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        r = run_multi_seed(seeds, args.data_dir)
        return r or rc

    return run_ablation(args.target, args.data_dir, args.seed)


if __name__ == "__main__":
    sys.exit(main())
