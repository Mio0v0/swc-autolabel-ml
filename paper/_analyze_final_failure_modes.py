#!/usr/bin/env python3
"""Final paper failure-mode analysis for the multi-seed v12+Branch3 run.

This is the current-paper replacement for the older ``gt_failure_modes.*``
files, which were built from a pre-final held-out CSV. The row unit here is
``(model_seed, file)`` from ``final_flag_multiseed_labels.csv``.

By default the script also regenerates no-leak leave-one-seed-out flag scores
for one selected flagger setting, then reports which failure modes are caught
by that flagger. The score model is fit only on the training seeds for each
fold and scored on the held-out seed rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper._train_eval_final_flag_multiseed import (  # noqa: E402
    TARGET_THRESHOLD,
    _feature_mode_frame,
    _fit_candidates,
    _load_joined,
    _loso_split,
    _select_candidate,
)
from paper._train_flag_model import F1_COL, _engineer_features  # noqa: E402


DEFAULT_LABELS = ROOT / "paper" / "results" / "final_flag_multiseed_labels.csv"
DEFAULT_FEATURES = ROOT / "paper" / "results" / "final_flag_multiseed_features_oof_heavy.csv"
DEFAULT_OUT_CSV = ROOT / "paper" / "results" / "final_failure_modes.csv"
DEFAULT_OUT_MD = ROOT / "paper" / "results" / "final_failure_modes.md"
DEFAULT_OUT_SCORES = ROOT / "paper" / "results" / "final_failure_mode_flag_scores.csv"
DEFAULT_OUT_JSON = ROOT / "paper" / "results" / "final_failure_modes.json"


Rule = tuple[str, str, Callable[[pd.Series], bool]]


def _num(row: pd.Series, col: str) -> float:
    try:
        val = row.get(col, np.nan)
        if val == "" or val is None:
            return float("nan")
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def _int(row: pd.Series, col: str) -> int:
    val = _num(row, col)
    if math.isnan(val):
        return 0
    return int(val)


def _bool_text(row: pd.Series, col: str) -> bool:
    return str(row.get(col, "")).strip().lower() == "true"


CATEGORIES: list[Rule] = [
    (
        "WHOLE_CELL_MISLABEL",
        "Stage 1 confidently disagrees with GT cell type.",
        lambda r: (not _bool_text(r, "stage1_correct")) and _num(r, "stage1_conf") > 0.85,
    ),
    (
        "APICAL_MISSED",
        "GT has apical nodes but the model predicted zero apical nodes.",
        lambda r: _num(r, "apical_frac") > 0.01 and _int(r, "pred_apical") == 0,
    ),
    (
        "APICAL_UNDERPREDICTED",
        "Model predicted less than half the GT apical node count and apical F1 is low.",
        lambda r: (
            _num(r, "apical_frac") > 0.01
            and _int(r, "pred_apical") < 0.5 * _num(r, "apical_frac") * max(1, _int(r, "n_nodes"))
            and _num(r, "apical_F1") < 0.5
        ),
    ),
    (
        "APICAL_OVERPREDICTED",
        "Model predicted far more apical nodes than GT and apical F1 is low.",
        lambda r: (
            _int(r, "pred_apical") > 1.5 * max(1.0, _num(r, "apical_frac") * max(1, _int(r, "n_nodes")))
            and _num(r, "apical_F1") < 0.5
            and _int(r, "pred_apical") > 100
        ),
    ),
    (
        "APICAL_INVERTED",
        "GT apical nodes are on average below the soma.",
        lambda r: (not math.isnan(_num(r, "apical_above_soma"))) and _num(r, "apical_above_soma") < 0,
    ),
    (
        "AXON_DENDRITE_CONFUSION",
        "Axon F1 is below 0.95.",
        lambda r: (not math.isnan(_num(r, "axon_F1"))) and _num(r, "axon_F1") < 0.95,
    ),
    (
        "BASAL_LOST",
        "Basal F1 is below 0.5 despite GT basal nodes.",
        lambda r: (not math.isnan(_num(r, "basal_F1"))) and _num(r, "basal_F1") < 0.5 and _num(r, "basal_frac") > 0.01,
    ),
    (
        "DISCONNECTED",
        "Reconstruction has more than one connected component.",
        lambda r: _int(r, "n_components") > 1,
    ),
    (
        "TRUNCATED_APICAL",
        "GT apical exists but has short z extent.",
        lambda r: 0.0 < _num(r, "apical_z_extent") < 50.0 and _num(r, "apical_frac") > 0.01,
    ),
    (
        "STAGE1_PROPAGATION",
        "Stage 1 is wrong and per-cell F1 is low.",
        lambda r: (not _bool_text(r, "stage1_correct")) and _num(r, "held_out_F1") < 0.7,
    ),
]


def _matching_categories(row: pd.Series) -> list[str]:
    out: list[str] = []
    for name, _description, rule in CATEGORIES:
        try:
            if bool(rule(row)):
                out.append(name)
        except Exception:  # noqa: BLE001
            continue
    return out


def _fit_loso_scores(
    labels_path: Path,
    features_path: Path,
    *,
    scope: str,
    feature_mode: str,
    reject_rate: float,
    seed: int,
) -> pd.DataFrame:
    joined = _load_joined(labels_path, features_path)
    if scope != "all":
        joined = joined[joined["cell_type_gt"].astype(str) == scope].copy()
    if joined.empty:
        raise SystemExit(f"No rows for scope={scope}")

    mode_df = _feature_mode_frame(joined, feature_mode)
    feat_df, numeric_features, categorical_features = _engineer_features(mode_df)
    f1 = pd.to_numeric(feat_df[F1_COL], errors="coerce")
    valid = f1.notna()
    feat_df = feat_df.loc[valid].reset_index(drop=True)
    f1 = f1.loc[valid].reset_index(drop=True)
    y = (f1 < TARGET_THRESHOLD).astype(int)

    score_rows: list[pd.DataFrame] = []
    for test_seed in sorted(feat_df["model_seed"].astype(int).unique()):
        split, leak = _loso_split(feat_df, y, int(test_seed), seed)
        if leak["train_test_file_overlap"] or leak["val_test_file_overlap"] or leak["train_val_file_overlap"]:
            raise SystemExit(f"Flag split leakage detected: {leak}")
        candidates = _fit_candidates(
            feat_df,
            numeric_features,
            categorical_features,
            y,
            split,
            seed + int(test_seed),
        )
        cand, _selected_val = _select_candidate(
            candidates,
            f1.loc[split.val].to_numpy(dtype=float),
            recall_target=0.8,
        )
        test_part = feat_df.loc[split.test, ["row_id", "model_seed", "file", "cell_type_gt", F1_COL]].copy()
        test_part["flag_score"] = cand["test_score"]
        k = max(1, int(round(reject_rate * len(test_part))))
        order = np.argsort(-test_part["flag_score"].to_numpy(dtype=float))
        rejected = np.zeros(len(test_part), dtype=bool)
        rejected[order[:k]] = True
        test_part["flagged"] = rejected
        test_part["flag_scope"] = scope
        test_part["flag_feature_mode"] = feature_mode
        test_part["flag_reject_rate"] = float(reject_rate)
        test_part["flag_selected_model"] = str(cand["model_name"])
        test_part["flag_selected_feature_set"] = str(cand["feature_set"])
        score_rows.append(test_part)
    return pd.concat(score_rows, ignore_index=True)


def _load_or_fit_loso_scores(
    labels_path: Path,
    features_path: Path,
    score_path: Path,
    *,
    scope: str,
    feature_mode: str,
    reject_rate: float,
    seed: int,
    force: bool,
) -> pd.DataFrame:
    if score_path.is_file() and not force:
        scores = pd.read_csv(score_path)
        expected = {
            "row_id",
            "flag_score",
            "flagged",
            "flag_scope",
            "flag_feature_mode",
            "flag_reject_rate",
        }
        if expected.issubset(scores.columns):
            scopes = set(scores["flag_scope"].astype(str))
            modes = set(scores["flag_feature_mode"].astype(str))
            rates = set(round(float(x), 8) for x in pd.to_numeric(scores["flag_reject_rate"], errors="coerce").dropna())
            if scopes == {scope} and modes == {feature_mode} and rates == {round(float(reject_rate), 8)}:
                return scores
    scores = _fit_loso_scores(
        labels_path,
        features_path,
        scope=scope,
        feature_mode=feature_mode,
        reject_rate=reject_rate,
        seed=seed,
    )
    score_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(score_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return scores


def _write_markdown(
    path: Path,
    *,
    labels_path: Path,
    bottom_pct: float,
    bottom_threshold: float,
    dataset: dict,
    category_summary: pd.DataFrame,
    unmatched: pd.DataFrame,
    examples_by_category: dict[str, pd.DataFrame],
    flag_scope: str,
    flag_feature_mode: str,
    flag_reject_rate: float,
    flag_enabled: bool,
) -> None:
    lines: list[str] = []
    add = lines.append
    add("# Final failure-mode analysis")
    add("")
    add(f"Input: `{labels_path.relative_to(ROOT)}`")
    add(f"Rows: {dataset['n_rows']} row-seed pairs, {dataset['n_unique_files']} unique files")
    add(f"Bad target: per-cell neurite F1 < {TARGET_THRESHOLD:.2f}")
    add(f"Bottom slice: lowest {bottom_pct:.1f}% = {dataset['n_bottom']} rows, F1 <= {bottom_threshold:.4f}")
    add("")
    if flag_enabled:
        add(
            f"Flag catch-rate columns use LOSO `{flag_feature_mode}` flagger, "
            f"scope `{flag_scope}`, fixed reject rate {flag_reject_rate:.2f}."
        )
    else:
        add("Flag scoring was disabled for this run.")
    add("")
    add("## Category summary")
    add("")
    add(
        "| Category | Bottom rows | Bottom bad rows | Unique bad files | Enrichment | "
        "Flagged bad | Flag recall in category | False flags in category |"
    )
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in category_summary.iterrows():
        add(
            f"| {r['category']} | {int(r['bottom_rows'])} | {int(r['bad_rows'])} | "
            f"{int(r['unique_bad_files'])} | {float(r['bottom_enrichment']):.2f}x | "
            f"{int(r['flagged_bad_rows'])} | {float(r['flag_recall_bad_rows']):.3f} | "
            f"{int(r['false_flag_rows'])} |"
        )
    add("")
    add("## Worst examples")
    add("")
    for category, ex in examples_by_category.items():
        if ex.empty:
            continue
        add(f"### {category}")
        add("")
        add("| file | seed | cell type | F1 | axon F1 | basal F1 | apical F1 | flagged | other categories |")
        add("|---|---:|---|---:|---:|---:|---:|---|---|")
        for _, r in ex.iterrows():
            add(
                f"| `{r['file']}` | {int(r['model_seed'])} | {r['cell_type_gt']} | "
                f"{float(r['held_out_F1']):.4f} | {_fmt_num(r.get('axon_F1'))} | "
                f"{_fmt_num(r.get('basal_F1'))} | {_fmt_num(r.get('apical_F1'))} | "
                f"{'yes' if bool(r.get('flagged', False)) else 'no'} | "
                f"{r.get('other_categories', '-') or '-'} |"
            )
        add("")
    add("## Unmatched bad rows")
    add("")
    add(f"{len(unmatched)} bad rows matched no category.")
    if not unmatched.empty:
        add("")
        add("| file | seed | cell type | F1 | axon F1 | basal F1 | apical F1 | flagged |")
        add("|---|---:|---|---:|---:|---:|---:|---|")
        for _, r in unmatched.head(20).iterrows():
            add(
                f"| `{r['file']}` | {int(r['model_seed'])} | {r['cell_type_gt']} | "
                f"{float(r['held_out_F1']):.4f} | {_fmt_num(r.get('axon_F1'))} | "
                f"{_fmt_num(r.get('basal_F1'))} | {_fmt_num(r.get('apical_F1'))} | "
                f"{'yes' if bool(r.get('flagged', False)) else 'no'} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_num(value) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(x):
        return "-"
    return f"{x:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    ap.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    ap.add_argument("--out-scores", type=Path, default=DEFAULT_OUT_SCORES)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--flag-score-input", type=Path, default=DEFAULT_OUT_SCORES)
    ap.add_argument("--bottom-pct", type=float, default=10.0)
    ap.add_argument("--flag-scope", choices=["all", "pyramidal", "interneuron"], default="all")
    ap.add_argument(
        "--flag-feature-mode",
        choices=["compact", "branch3", "baseline_oof", "v12_oof"],
        default="baseline_oof",
    )
    ap.add_argument("--flag-reject-rate", type=float, default=0.10)
    ap.add_argument("--top-examples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-flag-scores", action="store_true")
    ap.add_argument("--force-regenerate-flag-scores", action="store_true")
    args = ap.parse_args()

    labels = pd.read_csv(args.labels)
    labels["held_out_F1"] = pd.to_numeric(labels["held_out_F1"], errors="coerce")
    labels = labels[labels["held_out_F1"].notna()].copy()
    labels["bad_f060"] = labels["held_out_F1"] < TARGET_THRESHOLD
    labels["categories"] = ["|".join(_matching_categories(row)) for _, row in labels.iterrows()]
    labels["n_categories"] = labels["categories"].map(lambda x: 0 if not x else len(x.split("|")))

    scores = pd.DataFrame()
    if not args.no_flag_scores:
        print(
            f"Regenerating LOSO flag scores: scope={args.flag_scope} "
            f"feature_mode={args.flag_feature_mode} reject_rate={args.flag_reject_rate}",
            flush=True,
        )
        scores = _load_or_fit_loso_scores(
            args.labels,
            args.features,
            args.flag_score_input,
            scope=args.flag_scope,
            feature_mode=args.flag_feature_mode,
            reject_rate=args.flag_reject_rate,
            seed=args.seed,
            force=bool(args.force_regenerate_flag_scores),
        )
        args.out_scores.parent.mkdir(parents=True, exist_ok=True)
        if args.flag_score_input.resolve() != args.out_scores.resolve():
            scores.to_csv(args.out_scores, index=False, quoting=csv.QUOTE_MINIMAL)
        labels = labels.merge(
            scores[["row_id", "flag_score", "flagged", "flag_selected_model", "flag_selected_feature_set"]],
            on="row_id",
            how="left",
        )
        labels["flagged"] = labels["flagged"].fillna(False).astype(bool)
    else:
        labels["flagged"] = False

    threshold = float(np.nanpercentile(labels["held_out_F1"], args.bottom_pct))
    labels["in_bottom_slice"] = labels["held_out_F1"] <= threshold

    category_rows: list[dict] = []
    examples_by_category: dict[str, pd.DataFrame] = {}
    n_bottom = int(labels["in_bottom_slice"].sum())
    for name, description, _rule in CATEGORIES:
        in_cat = labels["categories"].fillna("").str.split("|").map(lambda xs: name in xs)
        bottom_cat = labels[in_cat & labels["in_bottom_slice"]]
        bad_cat = labels[in_cat & labels["bad_f060"]]
        flagged_bad = bad_cat[bad_cat["flagged"]]
        false_flags = labels[in_cat & labels["flagged"] & (~labels["bad_f060"])]
        rows_in_category = int(in_cat.sum())
        bottom_rate = len(bottom_cat) / max(1, n_bottom)
        corpus_rate = rows_in_category / max(1, len(labels))
        enrichment = float(bottom_rate / corpus_rate) if corpus_rate > 0 and rows_in_category > 0 else 0.0
        category_rows.append(
            {
                "category": name,
                "description": description,
                "rows": rows_in_category,
                "bottom_rows": int(len(bottom_cat)),
                "bad_rows": int(len(bad_cat)),
                "unique_bad_files": int(bad_cat["file"].nunique()),
                "bottom_enrichment": enrichment,
                "flagged_bad_rows": int(len(flagged_bad)),
                "flag_recall_bad_rows": float(len(flagged_bad) / max(1, len(bad_cat))),
                "false_flag_rows": int(len(false_flags)),
            }
        )
        ex = bad_cat.sort_values("held_out_F1").head(args.top_examples).copy()
        ex["other_categories"] = ex["categories"].map(
            lambda raw: ", ".join(c for c in str(raw).split("|") if c and c != name)
        )
        examples_by_category[name] = ex

    category_summary = pd.DataFrame(category_rows).sort_values(
        ["bottom_enrichment", "bad_rows"], ascending=[False, False]
    )
    unmatched_bad = labels[labels["bad_f060"] & (labels["n_categories"] == 0)].sort_values("held_out_F1")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.out_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    dataset = {
        "n_rows": int(len(labels)),
        "n_unique_files": int(labels["file"].nunique()),
        "n_bad_f060": int(labels["bad_f060"].sum()),
        "bad_rate_f060": float(labels["bad_f060"].mean()),
        "n_bottom": n_bottom,
        "bottom_threshold": threshold,
        "flag_scores_enabled": not args.no_flag_scores,
        "flag_scope": args.flag_scope,
        "flag_feature_mode": args.flag_feature_mode,
        "flag_reject_rate": args.flag_reject_rate,
    }
    args.out_json.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "category_summary": category_summary.to_dict("records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_markdown(
        args.out_md,
        labels_path=args.labels,
        bottom_pct=args.bottom_pct,
        bottom_threshold=threshold,
        dataset=dataset,
        category_summary=category_summary,
        unmatched=unmatched_bad,
        examples_by_category=examples_by_category,
        flag_scope=args.flag_scope,
        flag_feature_mode=args.flag_feature_mode,
        flag_reject_rate=args.flag_reject_rate,
        flag_enabled=not args.no_flag_scores,
    )
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_json}")
    if not args.no_flag_scores:
        print(f"Wrote {args.out_scores}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
