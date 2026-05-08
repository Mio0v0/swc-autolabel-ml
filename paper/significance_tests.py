#!/usr/bin/env python3
"""Paired-Wilcoxon significance tests for the auto-typing paper.

Reviewers consistently want a p-value next to every "method A beats
method B" claim. This script reads the per-file F1 CSVs we've already
snapshotted and runs paired-Wilcoxon signed-rank tests between every
pair of methods on the *same files*. Bonferroni-corrected over the
total number of comparisons.

Inputs (per-file F1 CSVs already in paper/results/snapshots/):
    v6_full_pipeline_per_file.csv       (v6 baseline)
    v7_gnn_branch.json (no per-file CSV — derived from snapshot JSON)
    v8_subtree_gnn.json (same)
    v9_baseline_no_gnn.csv              (v9 without Stage 2b GNN)
    v9_final_subtree_gnn.csv            (v9 final — current SOTA)
    external_baselines_per_file.csv     (NeuroM-RF, Sholl-RF, Sholl-MLP)

Output:
    paper/results/snapshots/significance_tests.json
    paper/results/snapshots/significance_tests.txt    paper-friendly table

Usage
-----
    python -m paper.significance_tests
    python -m paper.significance_tests --alpha 0.01
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = ROOT / "paper" / "results" / "snapshots"


# Per-file F1 CSVs to load. Maps method label → (csv_path, file_col, f1_col).
# The hybrid.evaluate output uses ``neurite_macro_f1_stage23`` (Stage 2+3
# refined). The eval_engine_on_test output (no-soft-handoff) uses
# ``neurite_macro_f1`` directly because it only runs the inference path.
PER_FILE_CSVS: dict[str, tuple[str, str, str]] = {
    "v6":              ("v6_full_pipeline_per_file.csv",      "path", "neurite_macro_f1_stage23"),
    "v9_no_gnn":       ("v9_baseline_no_gnn.csv",             "path", "neurite_macro_f1_stage23"),
    "v9_final":        ("v9_final_subtree_gnn.csv",           "path", "neurite_macro_f1_stage23"),
    # Ablation rows from the overnight queue (full retrain under env vars
    # / different seeds). Same schema as v9_final's per-file CSV.
    "no_pca":          ("eval_no_pca_per_file.csv",                      "path", "neurite_macro_f1_stage23"),
    "no_trunk":        ("eval_no_trunk_per_file.csv",                    "path", "neurite_macro_f1_stage23"),
    "no_trunk_plus_no_soft_handoff": (
        "eval_no_trunk_plus_no_soft_handoff_per_file.csv",                "path", "neurite_macro_f1_stage23"),
    "multi_seed_123":  ("eval_multi_seed_123_per_file.csv",              "path", "neurite_macro_f1_stage23"),
    "multi_seed_456":  ("eval_multi_seed_456_per_file.csv",              "path", "neurite_macro_f1_stage23"),
    # Inference-only ablation (no retrain) uses a slimmer schema.
    "no_soft_handoff": ("eval_no_soft_handoff_per_file.csv",  "path", "neurite_macro_f1"),
}

# External baselines were dumped as a long-format CSV with a `method` column;
# split it into per-method dicts at load time.
EXTERNAL_CSV = "external_baselines_per_file.csv"


def _load_per_file_csv(path: Path, file_col: str, f1_col: str) -> dict[str, float]:
    """Read a per-file CSV → {file_basename: f1}. Uses os.path.basename so
    minor path-prefix differences across snapshots don't break joins."""
    if not path.is_file():
        return {}
    out: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if file_col not in (reader.fieldnames or []) or f1_col not in (reader.fieldnames or []):
            print(
                f"  warning: {path.name} missing columns "
                f"(have: {reader.fieldnames}). Skipping.",
                file=sys.stderr,
            )
            return {}
        for row in reader:
            fname = Path(str(row[file_col])).name
            try:
                out[fname] = float(row[f1_col])
            except (TypeError, ValueError):
                continue
    return out


def _load_external_csv(path: Path) -> dict[str, dict[str, float]]:
    """Load the long-format external_baselines CSV → {method: {fname: f1}}."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            method = str(row.get("method", "")).strip()
            fname = Path(str(row.get("path", ""))).name
            try:
                f1 = float(row.get("neurite_macro_f1") or 0.0)
            except (TypeError, ValueError):
                continue
            if method and fname:
                out[method][fname] = f1
    return dict(out)


def _paired_wilcoxon(
    a: dict[str, float],
    b: dict[str, float],
) -> dict[str, float | int]:
    """Run a two-sided paired Wilcoxon signed-rank test on the intersection
    of files between two F1 dicts. Returns metrics + n_pairs."""
    common = sorted(set(a) & set(b))
    if not common:
        return {"n_pairs": 0, "mean_diff": float("nan"), "p_value": float("nan"),
                "median_diff": float("nan"), "wins": 0, "losses": 0, "ties": 0}
    a_arr = np.array([a[k] for k in common], dtype=np.float64)
    b_arr = np.array([b[k] for k in common], dtype=np.float64)
    diffs = a_arr - b_arr
    wins = int((diffs > 0).sum())
    losses = int((diffs < 0).sum())
    ties = int((diffs == 0).sum())
    if wins + losses == 0:
        # All differences zero — Wilcoxon is undefined.
        return {"n_pairs": len(common), "mean_diff": 0.0, "median_diff": 0.0,
                "p_value": 1.0, "wins": wins, "losses": losses, "ties": ties}
    try:
        stat = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        pv = float(stat.pvalue)
    except ValueError:
        pv = float("nan")
    return {
        "n_pairs": len(common),
        "mean_diff": float(diffs.mean()),
        "median_diff": float(np.median(diffs)),
        "p_value": pv,
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }


def _build_method_table(snapshot_dir: Path) -> dict[str, dict[str, float]]:
    """Aggregate every method's per-file F1 into a {method: {file: f1}} dict."""
    methods: dict[str, dict[str, float]] = {}
    for label, (csv_name, file_col, f1_col) in PER_FILE_CSVS.items():
        path = snapshot_dir / csv_name
        d = _load_per_file_csv(path, file_col, f1_col)
        if d:
            methods[label] = d
            print(f"  loaded {label}: {len(d)} files from {csv_name}")
    ext = _load_external_csv(snapshot_dir / EXTERNAL_CSV)
    for m, d in ext.items():
        methods[m] = d
        print(f"  loaded {m}: {len(d)} files from {EXTERNAL_CSV}")
    return methods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance threshold for the corrected p-value (default 0.05).",
    )
    parser.add_argument(
        "--snapshot-dir", type=Path, default=SNAPSHOT_DIR,
        help="Directory containing the per-file F1 CSVs.",
    )
    parser.add_argument(
        "--out-json", type=Path,
        default=SNAPSHOT_DIR / "significance_tests.json",
    )
    parser.add_argument(
        "--out-text", type=Path,
        default=SNAPSHOT_DIR / "significance_tests.txt",
    )
    args = parser.parse_args()

    print(f"Loading per-file F1 vectors from {args.snapshot_dir}...")
    methods = _build_method_table(args.snapshot_dir)
    if len(methods) < 2:
        print(
            "ERROR: need at least 2 methods to compare. Got: "
            + ", ".join(sorted(methods)),
            file=sys.stderr,
        )
        return 2

    # Run all pairwise paired-Wilcoxon tests.
    method_names = sorted(methods)
    pair_count = len(method_names) * (len(method_names) - 1) // 2
    bonferroni = max(1, pair_count)
    print(f"\nComparing {len(method_names)} methods, {pair_count} pairs, Bonferroni n={bonferroni}\n")

    results: list[dict] = []
    for a, b in combinations(method_names, 2):
        out = _paired_wilcoxon(methods[a], methods[b])
        out["method_a"] = a
        out["method_b"] = b
        out["p_value_bonferroni"] = (
            min(1.0, out["p_value"] * bonferroni)
            if isinstance(out["p_value"], float) and not np.isnan(out["p_value"])
            else float("nan")
        )
        out["significant_alpha_corrected"] = bool(
            isinstance(out["p_value_bonferroni"], float)
            and not np.isnan(out["p_value_bonferroni"])
            and out["p_value_bonferroni"] < args.alpha
        )
        results.append(out)

    # Summary by method, mean F1, n test files.
    method_summaries = {}
    for m, d in methods.items():
        if not d:
            continue
        arr = np.array(list(d.values()))
        method_summaries[m] = {
            "n_files": len(arr),
            "mean_f1": float(arr.mean()),
            "median_f1": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
        }

    payload = {
        "alpha": args.alpha,
        "bonferroni_correction_n": bonferroni,
        "method_summaries": method_summaries,
        "pairwise_tests": results,
    }
    args.out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out_json}")

    # Paper-style text output.
    lines = []
    lines.append(f"Paired Wilcoxon signed-rank tests (per-file neurite-macro-F1)")
    lines.append(f"  alpha={args.alpha}, Bonferroni n={bonferroni}")
    lines.append("")
    lines.append("Method summaries:")
    lines.append(f"  {'method':<22s}  {'n':>5s}  {'mean':>7s}  {'median':>7s}  {'p10':>7s}  {'p25':>7s}")
    for m in sorted(method_summaries, key=lambda k: -method_summaries[k]["mean_f1"]):
        s = method_summaries[m]
        lines.append(
            f"  {m:<22s}  {s['n_files']:>5d}  {s['mean_f1']:>7.4f}  "
            f"{s['median_f1']:>7.4f}  {s['p10']:>7.4f}  {s['p25']:>7.4f}"
        )
    lines.append("")
    lines.append("Pairwise comparisons (a vs b; positive mean_diff means a > b):")
    lines.append(
        f"  {'a':<22s} vs {'b':<22s}  {'n':>4s}  {'meandelta':>8s}  "
        f"{'mediandelta':>9s}  {'wins/loss/tie':>14s}  {'p':>9s}  {'p_bonf':>9s}  sig"
    )
    for r in sorted(results, key=lambda r: r["p_value_bonferroni"] if isinstance(r["p_value_bonferroni"], float) and not np.isnan(r["p_value_bonferroni"]) else 1.0):
        sig = "**" if r["significant_alpha_corrected"] else ""
        pv = r["p_value"]
        pvb = r["p_value_bonferroni"]
        lines.append(
            f"  {r['method_a']:<22s} vs {r['method_b']:<22s}  "
            f"{r['n_pairs']:>4d}  "
            f"{r['mean_diff']:>+8.4f}  {r['median_diff']:>+9.4f}  "
            f"{r['wins']:>4d}/{r['losses']:>3d}/{r['ties']:>3d}  "
            f"{pv:>9.2e}  {pvb:>9.2e}  {sig}"
        )

    text = "\n".join(lines)
    args.out_text.write_text(text, encoding="utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(f"Wrote {args.out_text}")
    print()
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
