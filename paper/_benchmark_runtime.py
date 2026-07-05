#!/usr/bin/env python3
"""Runtime benchmark — per-cell inference latency for v12+Branch3 and all
four constructed baselines. Reports median, mean, and per-node throughput.

Excludes one-time model load and SWC parse so numbers reflect the
marginal cost of labeling one additional cell once the model is
warm. Each method is timed on the SAME sample of cells so the
comparison is apples-to-apples per cell.

Usage:
    python -m paper._benchmark_runtime [--n-cells 100] [--seed 123]
        [--methods all,v12,neurom_rf,sholl_rf,sholl_mlp,lmeasure_rf]

Output:
    paper/results/runtime_benchmark.json
    paper/results/runtime_benchmark.csv     per-cell wall times
    paper/results/runtime_benchmark.txt     pretty summary table
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.features import parse_swc                              # noqa: E402
from hybrid.pipeline import run_pipeline_on_nodes                   # noqa: E402
from paper.gnn_inference import load_gnn                            # noqa: E402
from paper.gnn_branch3_inference import load_branch3                # noqa: E402
from paper.external_baselines import predict_with_cache             # noqa: E402

DATA_DIR = ROOT / "data" / "v12_uncurated"
MODEL_DIR = ROOT / "paper" / "models" / "v12_gentle_seed123"
BASELINE_CACHE_DIR = ROOT / "paper" / "models" / "baselines"
SPLIT_JSON = MODEL_DIR / "train_test_split.json"

METHODS_ALL = ("v12", "neurom_rf", "sholl_rf", "sholl_mlp", "lmeasure_rf")


def _sample_test_files(seed: int, n: int) -> list[tuple[str, Path]]:
    sp = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    rows: list[tuple[str, Path]] = []
    for ct in ("pyramidal", "interneuron"):
        for fn in sp["test"].get(ct, []):
            p = DATA_DIR / ct / "swc" / fn
            if p.is_file():
                rows.append((ct, p))
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def _time_v12(cells_parsed, s1_path, s2_path, gnn_state, branch3_state):
    """Return list of (file, n_nodes, total_ms, parts: dict) for v12 pipeline."""
    results = []
    for ct_gt, fname, nodes in cells_parsed:
        n_nodes = len(nodes)
        t0 = time.perf_counter()
        pr = run_pipeline_on_nodes(
            nodes, file_path="", stage1_model=s1_path, stage2_model=s2_path,
            gnn_state=gnn_state, branch3_state=branch3_state,
            use_subtree_stage2=True,
            override_cell_type=ct_gt,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _ = pr.node_labels
        results.append({
            "file": fname, "n_nodes": n_nodes,
            "total_ms": elapsed_ms,
        })
    return results


def _time_baseline(cells_parsed, predict_fn):
    """Return list of (file, n_nodes, total_ms) for one baseline predictor."""
    results = []
    for ct_gt, fname, nodes in cells_parsed:
        n_nodes = len(nodes)
        t0 = time.perf_counter()
        _ = predict_fn(nodes, ct_gt)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        results.append({
            "file": fname, "n_nodes": n_nodes,
            "total_ms": elapsed_ms,
        })
    return results


def _summarize(rows: list[dict], method: str) -> dict:
    times = [r["total_ms"] for r in rows]
    nodes = [r["n_nodes"] for r in rows]
    return {
        "method":         method,
        "n_cells":        len(rows),
        "total_nodes":    sum(nodes),
        "total_wall_s":   round(sum(times) / 1000.0, 3),
        "per_cell_mean_ms":   round(statistics.mean(times), 2),
        "per_cell_median_ms": round(statistics.median(times), 2),
        "per_cell_p10_ms":    round(sorted(times)[len(times)//10], 2) if len(times) >= 10 else None,
        "per_cell_p90_ms":    round(sorted(times)[max(0, len(times)*9//10 - 1)], 2) if len(times) >= 10 else None,
        "per_cell_min_ms":    round(min(times), 2),
        "per_cell_max_ms":    round(max(times), 2),
        "throughput_nodes_per_s": round(sum(nodes) / (sum(times) / 1000.0), 0)
                                  if sum(times) > 0 else 0.0,
        "throughput_cells_per_s": round(len(rows) / (sum(times) / 1000.0), 2)
                                  if sum(times) > 0 else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cells", type=int, default=100)
    ap.add_argument("--seed",    type=int, default=123)
    ap.add_argument("--methods", default="all",
                    help="comma-separated subset of " + ",".join(METHODS_ALL))
    args = ap.parse_args()
    methods = METHODS_ALL if args.methods == "all" else tuple(args.methods.split(","))

    print(f"=== Runtime benchmark ===")
    print(f"  seed={args.seed}, n_cells={args.n_cells}, methods={methods}")
    print()

    cells = _sample_test_files(args.seed, args.n_cells)
    if len(cells) < args.n_cells:
        print(f"  Note: only {len(cells)} test files available, requested {args.n_cells}.")
    print(f"  Pre-parsing {len(cells)} SWC files (excluded from timing)...")
    cells_parsed: list[tuple[str, str, list]] = []
    t0 = time.perf_counter()
    for ct_gt, p in cells:
        try:
            nodes = parse_swc(p)
            if nodes:
                cells_parsed.append((ct_gt, p.name, nodes))
        except Exception as exc:
            print(f"    skip {p.name}: {exc}")
    parse_s = time.perf_counter() - t0
    print(f"  parsed {len(cells_parsed)} cells in {parse_s:.1f}s")
    median_nodes = statistics.median(len(c[2]) for c in cells_parsed)
    mean_nodes   = statistics.mean(len(c[2]) for c in cells_parsed)
    print(f"  per-cell node count: mean={mean_nodes:.0f}, median={median_nodes:.0f}")
    print()

    summaries: list[dict] = []
    per_cell_rows: list[dict] = []

    if "v12" in methods:
        print("=== v12 (Stage 1 OFF, GT cell type) ===")
        print("  loading models...")
        s1_path     = MODEL_DIR / "cell_type_classifier.pkl"
        s2_path     = MODEL_DIR / "branch_classifier.pkl"
        gnn_state   = load_gnn(MODEL_DIR / "gnn_apical_basal.pt")
        branch3_state = load_branch3(MODEL_DIR / "gnn_branch3_rescue.pt", gate_path=None)
        print("  warming up on 3 cells...")
        _ = _time_v12(cells_parsed[:3], s1_path, s2_path, gnn_state, branch3_state)
        print(f"  timing {len(cells_parsed)} cells...")
        rows = _time_v12(cells_parsed, s1_path, s2_path, gnn_state, branch3_state)
        for r in rows:
            per_cell_rows.append({"method": "v12", **r})
        summary = _summarize(rows, "v12")
        summaries.append(summary)
        print(f"  done. per-cell median {summary['per_cell_median_ms']} ms, "
              f"throughput {summary['throughput_cells_per_s']} cells/s")
        print()

    # Need cross-source train set ONLY as cache key for the baseline loaders;
    # the cached pickles are what actually do inference.
    if any(m in methods for m in ("neurom_rf", "sholl_rf", "sholl_mlp", "lmeasure_rf")):
        print("  loading split (for baseline cache key only)...")
        sp = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
        train: dict[str, list[Path]] = {"pyramidal": [], "interneuron": []}
        for ct in ("pyramidal", "interneuron"):
            for fn in sp["train"].get(ct, []):
                p = DATA_DIR / ct / "swc" / fn
                if p.is_file():
                    train[ct].append(p)

    for method in ("neurom_rf", "sholl_rf", "sholl_mlp", "lmeasure_rf"):
        if method not in methods:
            continue
        print(f"=== {method} ===")
        cache_path = BASELINE_CACHE_DIR / f"{method}.pkl"
        if not cache_path.is_file():
            print(f"  MISSING cache: {cache_path}; skipping")
            continue
        print(f"  loading cached model from {cache_path.name}...")
        predict_fn = predict_with_cache(method, train, seed=args.seed,
                                         cache_path=cache_path, force_retrain=False)
        print("  warming up on 3 cells...")
        _ = _time_baseline(cells_parsed[:3], predict_fn)
        print(f"  timing {len(cells_parsed)} cells...")
        rows = _time_baseline(cells_parsed, predict_fn)
        for r in rows:
            per_cell_rows.append({"method": method, **r})
        summary = _summarize(rows, method)
        summaries.append(summary)
        print(f"  done. per-cell median {summary['per_cell_median_ms']} ms, "
              f"throughput {summary['throughput_cells_per_s']} cells/s")
        print()

    out_json = ROOT / "paper" / "results" / "runtime_benchmark.json"
    out_csv  = ROOT / "paper" / "results" / "runtime_benchmark.csv"
    out_txt  = ROOT / "paper" / "results" / "runtime_benchmark.txt"

    payload = {
        "seed": args.seed,
        "n_cells": len(cells_parsed),
        "node_count_mean":   round(mean_nodes, 1),
        "node_count_median": round(median_nodes, 1),
        "model_dir": str(MODEL_DIR.relative_to(ROOT)),
        "baselines_cache_dir": str(BASELINE_CACHE_DIR.relative_to(ROOT)),
        "platform_note": "single Windows workstation; v12 Stage 2 on GPU (XGBoost), GNN/Branch3 on CPU, baselines on CPU.",
        "exclusions": "SWC parse and one-time model load excluded from per-cell timing; warmup on 3 cells discarded.",
        "summary_by_method": summaries,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if per_cell_rows:
        with out_csv.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_cell_rows[0].keys()))
            w.writeheader()
            for r in per_cell_rows:
                w.writerow(r)

    lines = ["Runtime benchmark",
             "=" * 64,
             "",
             f"n_cells={len(cells_parsed)}  (seed={args.seed}; sampled from v12_gentle_seed123 test split)",
             f"per-cell node count: mean={mean_nodes:.0f}  median={median_nodes:.0f}",
             "Excludes SWC parse + model load. Warmup discarded.",
             "",
             f"{'method':14s} {'cells/s':>9} {'nodes/s':>10} "
             f"{'mean ms':>10} {'median ms':>10} {'p10':>8} {'p90':>8} {'min':>7} {'max':>7}"]
    for s in summaries:
        lines.append(f"{s['method']:14s} {s['throughput_cells_per_s']:>9.2f} "
                     f"{s['throughput_nodes_per_s']:>10.0f} "
                     f"{s['per_cell_mean_ms']:>10.1f} {s['per_cell_median_ms']:>10.1f} "
                     f"{s['per_cell_p10_ms'] if s['per_cell_p10_ms'] is not None else '-':>8} "
                     f"{s['per_cell_p90_ms'] if s['per_cell_p90_ms'] is not None else '-':>8} "
                     f"{s['per_cell_min_ms']:>7.1f} {s['per_cell_max_ms']:>7.1f}")
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 64)
    print(f"  wrote {out_json.name}, {out_csv.name}, {out_txt.name}")
    print("=" * 64)
    print(out_txt.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
