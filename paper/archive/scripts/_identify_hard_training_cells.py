#!/usr/bin/env python3
"""Identify worst-decile training cells for hard-cell upweighting (Phase 3a, exp β).

Runs the current Stage 1 + Stage 2 + GNN pipeline on the TRAINING split
(no leakage worry — we're labeling-mining, not eval-reporting), computes
per-cell F1, then writes the bottom-decile cell names so a follow-up
training pass can upweight them.

Usage:
    python -m paper._identify_hard_training_cells --seed 42
    python -m paper._identify_hard_training_cells --seed 789

Output:
    paper/models/v12_gentle_seed<N>/hard_cells.json
        {"hard_cells": [<file>, ...], "upweight_multiplier": 5.0}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.features import parse_swc                                 # noqa: E402
from hybrid.pipeline import run_pipeline_on_nodes                     # noqa: E402
from hybrid.evaluate import per_cell_neurite_f1                       # noqa: E402
from paper.gnn_inference import load_gnn                              # noqa: E402

DATA_DIR = ROOT / "data" / "v12_uncurated"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=None,
                    help="Convenience: use v12_gentle_seed<N> dir. Ignored if --model-dir given.")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="Override: path to ANY model dir with train_test_split.json + 3 model files.")
    ap.add_argument("--bottom-decile", type=float, default=0.10,
                    help="Fraction of training cells to flag as 'hard' (default 0.10).")
    ap.add_argument("--upweight", type=float, default=5.0,
                    help="Sample-weight multiplier for hard cells (default 5x).")
    args = ap.parse_args()

    if args.model_dir is not None:
        model_dir = args.model_dir.resolve()
    elif args.seed is not None:
        model_dir = ROOT / "paper" / "models" / f"v12_gentle_seed{args.seed}"
    else:
        raise SystemExit("Must specify --model-dir or --seed")
    split_json = model_dir / "train_test_split.json"
    s1 = model_dir / "cell_type_classifier.pkl"
    s2 = model_dir / "branch_classifier.pkl"
    gnn = model_dir / "gnn_apical_basal.pt"
    for p in (split_json, s1, s2, gnn):
        if not p.is_file():
            raise SystemExit(f"MISSING: {p}")

    sp = json.loads(split_json.read_text(encoding="utf-8"))
    train_files: list[tuple[str, Path]] = []
    for ct in ("pyramidal", "interneuron"):
        for fn in sp["train"].get(ct, []):
            p = DATA_DIR / ct / "swc" / fn
            if p.is_file():
                train_files.append((ct, p))
    print(f"Seed={args.seed}: scanning {len(train_files)} TRAINING cells "
          f"to find bottom-{args.bottom_decile*100:.0f}% by per-cell F1")

    gnn_state = load_gnn(gnn)
    per_cell: list[tuple[str, float]] = []   # (filename, F1)
    t0 = time.perf_counter()
    for i, (ct_gt, p) in enumerate(train_files):
        try:
            nodes = parse_swc(p)
            if not nodes:
                continue
            gt = [n.type for n in nodes]
        except Exception as exc:
            print(f"  WARN {p.name}: {exc}")
            continue
        pr = run_pipeline_on_nodes(
            nodes, file_path="", stage1_model=s1, stage2_model=s2,
            gnn_state=gnn_state, use_subtree_stage2=True,
        )
        pred = list(pr.node_labels)
        f1 = per_cell_neurite_f1(gt, pred, ct_gt)
        per_cell.append((p.name, f1))
        if (i + 1) % 500 == 0:
            print(f"    ... {i+1}/{len(train_files)} ({(time.perf_counter()-t0)/60:.1f} min)")

    elapsed = (time.perf_counter() - t0) / 60.0
    print(f"Inference on training set: {elapsed:.1f} min")

    # Pick bottom decile
    per_cell.sort(key=lambda x: x[1])
    n_hard = max(1, int(len(per_cell) * args.bottom_decile))
    hard = per_cell[:n_hard]
    threshold = hard[-1][1]
    print(f"Bottom {args.bottom_decile*100:.0f}% threshold: F1 <= {threshold:.4f}")
    print(f"  n_hard cells: {n_hard}")
    print(f"  F1 distribution of hard cells: min={hard[0][1]:.4f}, max={threshold:.4f}, "
          f"mean={np.mean([f for _,f in hard]):.4f}")

    out_path = model_dir / "hard_cells.json"
    out_path.write_text(json.dumps({
        "seed":                args.seed,
        "n_train_cells":       len(per_cell),
        "n_hard_cells":        n_hard,
        "bottom_decile":       args.bottom_decile,
        "f1_threshold":        threshold,
        "upweight_multiplier": args.upweight,
        "hard_cells":          [name for name, _ in hard],
        "scoring_inference_min": elapsed,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
