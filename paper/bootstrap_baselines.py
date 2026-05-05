#!/usr/bin/env python3
"""Bootstrap 95% CIs on per-file F1 for every method in the paper table.

Reviewers expect a confidence interval next to every "method X reaches
F1 = 0.96" claim. This script reads the same per-file F1 CSVs the
significance-tests script consumes and produces 95% CIs on the per-file
mean / median / P10 for each method.

Files at the **cell** level are the unit of statistical independence
(two nodes from the same cell are highly correlated), so we resample
files with replacement, not nodes.

Inputs:
    paper/results/snapshots/v6_full_pipeline_per_file.csv
    paper/results/snapshots/v9_baseline_no_gnn.csv
    paper/results/snapshots/v9_final_subtree_gnn.csv
    paper/results/snapshots/external_baselines_per_file.csv

Outputs:
    paper/results/snapshots/baselines_bootstrap_ci.json
    paper/results/snapshots/baselines_bootstrap_ci.txt

Usage::

    python -m paper.bootstrap_baselines
    python -m paper.bootstrap_baselines --n-bootstrap 5000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = ROOT / "paper" / "results" / "snapshots"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Re-use the loaders from significance_tests so we share the exact same
# per-file F1 vectors the Wilcoxon tests use.
from paper.significance_tests import _build_method_table  # noqa: E402
from paper.bootstrap_ci import bootstrap_per_file_distribution  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-json", type=Path,
        default=SNAPSHOT_DIR / "baselines_bootstrap_ci.json",
    )
    parser.add_argument(
        "--out-text", type=Path,
        default=SNAPSHOT_DIR / "baselines_bootstrap_ci.txt",
    )
    args = parser.parse_args()

    print(f"Loading per-file F1 vectors from {SNAPSHOT_DIR}...")
    methods = _build_method_table(SNAPSHOT_DIR)
    if not methods:
        print("ERROR: no per-file F1 vectors found.", file=sys.stderr)
        return 2

    print(f"\nBootstrapping each method ({args.n_bootstrap} resamples, seed={args.seed})\n")
    out: dict[str, dict] = {}
    for name in sorted(methods):
        f1s = list(methods[name].values())
        if not f1s:
            continue
        ci = bootstrap_per_file_distribution(
            f1s, n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
        out[name] = ci
        print(
            f"  {name:<22s}  n={ci['n_files']:>4d}  "
            f"mean={ci['mean_observed']:.4f} "
            f"[{ci['mean_ci_lo_2_5']:.4f}, {ci['mean_ci_hi_97_5']:.4f}]  "
            f"P10={ci['p10_observed']:.4f} "
            f"[{ci['p10_ci_lo_2_5']:.4f}, {ci['p10_ci_hi_97_5']:.4f}]"
        )

    payload = {
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "by_method": out,
    }
    args.out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out_json}")

    # Plain-text table for the paper.
    lines = [
        f"Bootstrap 95% CIs on per-file neurite-macro-F1",
        f"  n_bootstrap={args.n_bootstrap}, seed={args.seed}, file-level resampling",
        "",
        f"  {'method':<22s}  {'n':>4s}  "
        f"{'mean':>7s}  {'mean 95% CI':>22s}  "
        f"{'median':>7s}  {'P10':>7s}  {'P10 95% CI':>22s}",
    ]
    for name in sorted(out, key=lambda k: -out[k]["mean_observed"]):
        c = out[name]
        lines.append(
            f"  {name:<22s}  {c['n_files']:>4d}  "
            f"{c['mean_observed']:>7.4f}  "
            f"[{c['mean_ci_lo_2_5']:.4f}, {c['mean_ci_hi_97_5']:.4f}]    "
            f"{c['median_observed']:>7.4f}  "
            f"{c['p10_observed']:>7.4f}  "
            f"[{c['p10_ci_lo_2_5']:.4f}, {c['p10_ci_hi_97_5']:.4f}]"
        )
    text = "\n".join(lines)
    args.out_text.write_text(text, encoding="utf-8")
    print(f"Wrote {args.out_text}")
    print()
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
