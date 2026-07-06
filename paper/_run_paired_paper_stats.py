#!/usr/bin/env python3
"""Paired cell-level statistics for v12 vs external baselines.

The primary paired unit is the SWC file. If the same file appears in more than
one seed's held-out set, its paired differences are averaged first, then the
bootstrap/sign-flip tests run over unique files. This keeps multi-seed
evaluation from pretending repeated cells are independent.

Outputs:
  paper/results/final_paired_stats.csv
  paper/results/final_paired_stats.json
  paper/results/final_paired_stats.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "paper" / "results"
BASELINE_METHODS = ("lmeasure_rf", "neurom_rf", "sholl_rf", "sholl_mlp")
METRICS = ("accuracy", "neurite_macro_f1")


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"MISSING: {path}")
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _load_pairs(seeds: list[int], v12_pattern: str, baseline_pattern: str) -> list[dict]:
    pairs: list[dict] = []
    for seed in seeds:
        v12_path = RESULTS_DIR / v12_pattern.format(seed=seed)
        baseline_path = RESULTS_DIR / baseline_pattern.format(seed=seed)
        v12_rows = _read_csv(v12_path)
        baseline_rows = _read_csv(baseline_path)

        v12_by_file = {r["file"]: r for r in v12_rows}
        baseline_by_method_file: dict[tuple[str, str], dict] = {
            (r["method"], r["file"]): r for r in baseline_rows
        }
        for method in BASELINE_METHODS:
            for file_name, v12 in v12_by_file.items():
                base = baseline_by_method_file.get((method, file_name))
                if base is None:
                    continue
                for metric in METRICS:
                    pairs.append({
                        "seed": seed,
                        "file": file_name,
                        "cell_type": v12.get("cell_type_gt") or v12.get("cell_type") or base.get("cell_type", ""),
                        "baseline_method": method,
                        "metric": metric,
                        "v12_value": float(v12[metric]),
                        "baseline_value": float(base[metric]),
                        "diff": float(v12[metric]) - float(base[metric]),
                    })
    return pairs


def _load_pairs_from_ablation(
    seeds: list[int],
    ablation_rows_path: Path,
    ablation_mode: str,
    baseline_pattern: str,
) -> list[dict]:
    pairs: list[dict] = []
    ablation_rows = _read_csv(ablation_rows_path)
    v12_by_seed_file: dict[tuple[int, str], dict] = {}
    for row in ablation_rows:
        if row.get("ablation") != ablation_mode:
            continue
        try:
            seed = int(row["seed"])
        except Exception:
            continue
        v12_by_seed_file[(seed, row["file"])] = row

    for seed in seeds:
        baseline_path = RESULTS_DIR / baseline_pattern.format(seed=seed)
        baseline_rows = _read_csv(baseline_path)
        baseline_by_method_file: dict[tuple[str, str], dict] = {
            (r["method"], r["file"]): r for r in baseline_rows
        }
        seed_files = sorted(file_name for (s, file_name) in v12_by_seed_file if s == seed)
        for method in BASELINE_METHODS:
            for file_name in seed_files:
                v12 = v12_by_seed_file[(seed, file_name)]
                base = baseline_by_method_file.get((method, file_name))
                if base is None:
                    continue
                for metric in METRICS:
                    pairs.append({
                        "seed": seed,
                        "file": file_name,
                        "cell_type": v12.get("cell_type_gt") or base.get("cell_type", ""),
                        "baseline_method": method,
                        "metric": metric,
                        "v12_value": float(v12[metric]),
                        "baseline_value": float(base[metric]),
                        "diff": float(v12[metric]) - float(base[metric]),
                    })
    return pairs


def _cluster_by_file(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return file-cluster means: diff, v12, baseline."""
    by_file: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_file[row["file"]].append(row)
    diffs, v12_vals, base_vals = [], [], []
    for items in by_file.values():
        diffs.append(np.mean([r["diff"] for r in items]))
        v12_vals.append(np.mean([r["v12_value"] for r in items]))
        base_vals.append(np.mean([r["baseline_value"] for r in items]))
    return (
        np.asarray(diffs, dtype=float),
        np.asarray(v12_vals, dtype=float),
        np.asarray(base_vals, dtype=float),
    )


def _bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _bootstrap_p10_diff(
    v12_vals: np.ndarray,
    base_vals: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float, float]:
    if v12_vals.size == 0:
        return 0.0, 0.0, 0.0
    obs = float(np.percentile(v12_vals, 10) - np.percentile(base_vals, 10))
    idx = rng.integers(0, v12_vals.size, size=(n_boot, v12_vals.size))
    diffs = np.percentile(v12_vals[idx], 10, axis=1) - np.percentile(base_vals[idx], 10, axis=1)
    return obs, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _sign_flip_pvalue(values: np.ndarray, rng: np.random.Generator, n_perm: int) -> float:
    if values.size == 0:
        return 1.0
    obs = abs(float(values.mean()))
    if obs == 0.0:
        return 1.0
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, values.size), replace=True)
    perm = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(perm >= obs) + 1) / (n_perm + 1))


def _summarize_pairs(pairs: list[dict], n_boot: int, n_perm: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    paired_rows: list[dict] = []

    for method in BASELINE_METHODS:
        for metric in METRICS:
            subset = [r for r in pairs if r["baseline_method"] == method and r["metric"] == metric]
            if not subset:
                continue
            diffs, v12_vals, base_vals = _cluster_by_file(subset)
            ci_lo, ci_hi = _bootstrap_mean_ci(diffs, rng, n_boot)
            p10_diff, p10_lo, p10_hi = _bootstrap_p10_diff(v12_vals, base_vals, rng, n_boot)
            p_value = _sign_flip_pvalue(diffs, rng, n_perm)
            wins = int(np.count_nonzero(diffs > 0))
            losses = int(np.count_nonzero(diffs < 0))
            ties = int(np.count_nonzero(np.isclose(diffs, 0.0)))
            rows.append({
                "comparison": f"v12_branch3_gt_celltype_vs_{method}",
                "baseline_method": method,
                "metric": metric,
                "n_seed_file_pairs": len(subset),
                "n_unique_files": int(diffs.size),
                "v12_mean": float(v12_vals.mean()),
                "baseline_mean": float(base_vals.mean()),
                "mean_diff_v12_minus_baseline": float(diffs.mean()),
                "mean_diff_ci95_low": ci_lo,
                "mean_diff_ci95_high": ci_hi,
                "sign_flip_p_two_sided": p_value,
                "p10_diff_v12_minus_baseline": p10_diff,
                "p10_diff_ci95_low": p10_lo,
                "p10_diff_ci95_high": p10_hi,
                "v12_win_file_count": wins,
                "baseline_win_file_count": losses,
                "tie_file_count": ties,
                "v12_win_fraction": wins / max(1, int(diffs.size)),
            })
            paired_rows.extend(subset)
    return rows, paired_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = row.copy()
            for key, value in out.items():
                if isinstance(value, float):
                    if math.isfinite(value):
                        out[key] = f"{value:.8f}"
                    else:
                        out[key] = ""
            writer.writerow(out)


def _text(rows: list[dict]) -> str:
    lines = [
        "Paired v12 vs baseline statistics",
        "=================================",
        "",
        "Unit: unique SWC file. Repeated seed-file pairs are averaged before testing.",
        "Diff is v12 minus baseline. Bootstrap CI is 95%; p-value is paired sign-flip.",
        "",
    ]
    for row in rows:
        lines.append(f"{row['baseline_method']} / {row['metric']}")
        lines.append(
            f"  n_files={row['n_unique_files']}  n_seed_file_pairs={row['n_seed_file_pairs']}  "
            f"v12_mean={row['v12_mean']:.4f}  baseline_mean={row['baseline_mean']:.4f}"
        )
        lines.append(
            f"  mean_diff={row['mean_diff_v12_minus_baseline']:+.4f} "
            f"CI95=[{row['mean_diff_ci95_low']:+.4f}, {row['mean_diff_ci95_high']:+.4f}] "
            f"p={row['sign_flip_p_two_sided']:.6f}"
        )
        lines.append(
            f"  p10_diff={row['p10_diff_v12_minus_baseline']:+.4f} "
            f"CI95=[{row['p10_diff_ci95_low']:+.4f}, {row['p10_diff_ci95_high']:+.4f}] "
            f"wins={row['v12_win_file_count']} losses={row['baseline_win_file_count']} "
            f"ties={row['tie_file_count']}"
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="123,42,789")
    ap.add_argument("--v12-pattern", default="v12_gt_celltype_seed{seed}.csv")
    ap.add_argument("--v12-ablation-rows", type=Path, default=None,
                    help="Use a multi-mode ablation per-cell CSV instead of --v12-pattern.")
    ap.add_argument("--v12-ablation-mode", default="full_branch3_gt_celltype")
    ap.add_argument("--baseline-pattern", default="baselines_on_v12_seed{seed}.csv")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--n-perm", type=int, default=20000)
    ap.add_argument("--rng-seed", type=int, default=20260609)
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.v12_ablation_rows is not None:
        v12_ablation_rows = args.v12_ablation_rows
        if not v12_ablation_rows.is_absolute():
            v12_ablation_rows = (ROOT / v12_ablation_rows).resolve()
        pairs = _load_pairs_from_ablation(
            seeds,
            v12_ablation_rows,
            args.v12_ablation_mode,
            args.baseline_pattern,
        )
        v12_source = str(v12_ablation_rows.relative_to(ROOT))
    else:
        pairs = _load_pairs(seeds, args.v12_pattern, args.baseline_pattern)
        v12_source = args.v12_pattern
    summary_rows, paired_rows = _summarize_pairs(pairs, args.n_boot, args.n_perm, args.rng_seed)

    _write_csv(RESULTS_DIR / "final_paired_stats.csv", summary_rows)
    _write_csv(RESULTS_DIR / "final_paired_stats_pairs.csv", paired_rows)
    (RESULTS_DIR / "final_paired_stats.json").write_text(
        json.dumps(
            {
                "seeds": seeds,
                "v12_source": v12_source,
                "v12_ablation_mode": args.v12_ablation_mode if args.v12_ablation_rows is not None else None,
                "baseline_pattern": args.baseline_pattern,
                "n_boot": args.n_boot,
                "n_perm": args.n_perm,
                "rng_seed": args.rng_seed,
                "summary": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (RESULTS_DIR / "final_paired_stats.txt").write_text(_text(summary_rows), encoding="utf-8")

    print(_text(summary_rows))
    print(f"Wrote {RESULTS_DIR / 'final_paired_stats.txt'}")
    print(f"Wrote {RESULTS_DIR / 'final_paired_stats.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
