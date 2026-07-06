#!/usr/bin/env python3
"""Combine per-seed flag labels/features and audit leakage prerequisites."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


DEFAULT_SEEDS = (123, 42, 789)


def _display(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def _seed_from_model_name(raw: str) -> int:
    m = re.search(r"seed(\d+)", str(raw))
    if not m:
        raise ValueError(f"Cannot parse seed from {raw!r}")
    return int(m.group(1))


def _load_split(seed: int) -> dict:
    path = ROOT / "paper" / "models" / f"v12_gentle_seed{seed}" / "train_test_split.json"
    if not path.is_file():
        raise SystemExit(f"MISSING: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _split_sets(seed: int) -> tuple[set[str], set[str]]:
    split = _load_split(seed)
    train = set(split["train"].get("pyramidal", [])) | set(split["train"].get("interneuron", []))
    test = set(split["test"].get("pyramidal", [])) | set(split["test"].get("interneuron", []))
    return train, test


def _read_seed(seed: int, labels_pattern: str, features_pattern: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels_path = ROOT / labels_pattern.format(seed=seed)
    features_path = ROOT / features_pattern.format(seed=seed)
    if not labels_path.is_file():
        raise SystemExit(f"MISSING: {labels_path}")
    if not features_path.is_file():
        raise SystemExit(f"MISSING: {features_path}")
    labels = pd.read_csv(labels_path)
    features = pd.read_csv(features_path)
    if "file" not in labels.columns or "file" not in features.columns:
        raise SystemExit(f"{labels_path} or {features_path} is missing 'file'")
    labels = labels.copy()
    features = features.copy()
    labels["model_seed"] = seed
    features["model_seed"] = seed
    labels["row_id"] = [f"seed{seed}:{name}" for name in labels["file"].astype(str)]
    features["row_id"] = [f"seed{seed}:{name}" for name in features["file"].astype(str)]
    return labels, features


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="123,42,789")
    ap.add_argument(
        "--labels-pattern",
        default="paper/results/final_flag_seed{seed}_labels.csv",
        help="Path pattern relative to repo root. Must include {seed}.",
    )
    ap.add_argument(
        "--features-pattern",
        default="paper/results/final_flag_seed{seed}_features.csv",
        help="Path pattern relative to repo root. Must include {seed}.",
    )
    ap.add_argument("--out-labels", type=Path, default=ROOT / "paper" / "results" / "final_flag_multiseed_labels.csv")
    ap.add_argument("--out-features", type=Path, default=ROOT / "paper" / "results" / "final_flag_multiseed_features.csv")
    ap.add_argument(
        "--out-check",
        type=Path,
        default=ROOT / "paper" / "results" / "final_flag_multiseed_leakage_check.json",
    )
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    label_parts: list[pd.DataFrame] = []
    feature_parts: list[pd.DataFrame] = []
    seed_checks: dict[str, dict] = {}

    for seed in seeds:
        labels, features = _read_seed(seed, args.labels_pattern, args.features_pattern)
        train_set, test_set = _split_sets(seed)
        files = set(labels["file"].astype(str))
        in_train = sorted(files & train_set)
        not_in_test = sorted(files - test_set)
        duplicate_rows = int(labels["row_id"].duplicated().sum() + features["row_id"].duplicated().sum())
        missing_features = sorted(set(labels["row_id"]) - set(features["row_id"]))
        extra_features = sorted(set(features["row_id"]) - set(labels["row_id"]))
        seed_checks[str(seed)] = {
            "n_labels": int(len(labels)),
            "n_features": int(len(features)),
            "n_unique_files": int(labels["file"].nunique()),
            "files_in_model_train": len(in_train),
            "files_not_in_model_test": len(not_in_test),
            "duplicate_row_ids": duplicate_rows,
            "missing_feature_rows": len(missing_features),
            "extra_feature_rows": len(extra_features),
            "examples": {
                "files_in_model_train": in_train[:5],
                "files_not_in_model_test": not_in_test[:5],
                "missing_feature_rows": missing_features[:5],
                "extra_feature_rows": extra_features[:5],
            },
        }
        label_parts.append(labels)
        feature_parts.append(features)

    labels_all = pd.concat(label_parts, ignore_index=True)
    features_all = pd.concat(feature_parts, ignore_index=True)
    if labels_all["row_id"].duplicated().any():
        raise SystemExit("Duplicate row_id in combined labels")
    if features_all["row_id"].duplicated().any():
        raise SystemExit("Duplicate row_id in combined features")

    # Keep row_id/model_seed/file first for stable downstream merges.
    label_first = ["row_id", "model_seed", "file"]
    labels_all = labels_all[label_first + [c for c in labels_all.columns if c not in label_first]]
    feature_first = ["row_id", "model_seed", "file"]
    features_all = features_all[feature_first + [c for c in features_all.columns if c not in feature_first]]

    args.out_labels.parent.mkdir(parents=True, exist_ok=True)
    labels_all.to_csv(args.out_labels, index=False)
    features_all.to_csv(args.out_features, index=False)

    files_by_seed = {
        str(seed): set(labels_all.loc[labels_all["model_seed"] == seed, "file"].astype(str))
        for seed in seeds
    }
    pair_overlaps = {
        f"{a}_x_{b}": len(files_by_seed[str(a)] & files_by_seed[str(b)])
        for i, a in enumerate(seeds)
        for b in seeds[i + 1 :]
    }
    triple = len(set.intersection(*(files_by_seed[str(seed)] for seed in seeds))) if seeds else 0
    ok = all(
        check["files_in_model_train"] == 0
        and check["files_not_in_model_test"] == 0
        and check["duplicate_row_ids"] == 0
        and check["missing_feature_rows"] == 0
        and check["extra_feature_rows"] == 0
        for check in seed_checks.values()
    )
    summary = {
        "ok": ok,
        "seeds": seeds,
        "out_labels": _display(args.out_labels),
        "out_features": _display(args.out_features),
        "n_rows": int(len(labels_all)),
        "n_unique_files": int(labels_all["file"].nunique()),
        "duplicate_file_rows_across_seeds": int(len(labels_all) - labels_all["file"].nunique()),
        "pair_test_file_overlaps": pair_overlaps,
        "triple_test_file_overlap": triple,
        "by_seed": seed_checks,
    }
    args.out_check.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.out_check.with_suffix(".txt").write_text(
        "\n".join(
            [
                "Final multi-seed flag leakage precheck",
                f"ok: {summary['ok']}",
                f"rows: {summary['n_rows']}",
                f"unique files: {summary['n_unique_files']}",
                f"duplicate file rows across seeds: {summary['duplicate_file_rows_across_seeds']}",
                f"pair overlaps: {summary['pair_test_file_overlaps']}",
                f"triple overlap: {summary['triple_test_file_overlap']}",
                f"labels: {summary['out_labels']}",
                f"features: {summary['out_features']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not ok:
        raise SystemExit("Leakage precheck failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
