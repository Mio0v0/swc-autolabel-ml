#!/usr/bin/env python3
"""Baselines and ablations for the hybrid auto-labeling pipeline.

Produces the comparison table reviewers expect — every row in the format

    method | accuracy | macro_F1 | neurite_F1 | per_class F1 | per-file mean F1 | P10

against the same held-out test split used by `hybrid.evaluate`.

Subcommands
-----------
    floor          Run the three Tier 1 floor baselines:
                   - random           (sample by class frequency)
                   - majority         (predict the dominant non-soma class)
                   - heuristic        (rule-based: largest-z = apical, etc.)
    ablation       Run Tier 2 ablations on the existing pipeline:
                   - no-stage3        (Stage 2 only — extracted from eval JSON)
                   - no-soft-handoff  (force hard Stage-1 handoff)
                   - no-trunk         (skip the 4 trunk-detection features)
                   - no-pca           (skip PCA features in Stages 1+2)
                   - no-cell-type     (single Stage 2 model, no cell-type conditioning)
    all            Run everything; write a paper-ready table.

Usage
-----
    python -m paper.baselines floor
    python -m paper.baselines ablation --which no-soft-handoff
    python -m paper.baselines all

Outputs
-------
    paper/results/baselines_results.json  All results (one entry per method)
    paper/results/baselines_table.txt     Plain-text table for the paper
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid.features import SWCNode, parse_swc                                # noqa: E402
from hybrid.branch_features import extract_branches                            # noqa: E402
from hybrid.cell_type_detector import CELL_TYPE_LABEL_SETS                     # noqa: E402
from hybrid.evaluate import (                                                  # noqa: E402
    LABEL_NAMES,
    _compute_metrics,
    _evaluate_file,
    _file_level_split,
    _summarize_file_scores,
    _train_stage1,
    _train_stage2,
)


DEFAULT_DATA_DIR = ROOT / "data" / "benchmark_pyramidal_interneuron_v1_qc_diag_pruned"
DEFAULT_OUT = ROOT / "paper" / "results" / "baselines_results.json"
DEFAULT_TABLE = ROOT / "paper" / "results" / "baselines_table.txt"


# =============================================================================
# Helpers shared across baselines
# =============================================================================

def _collect_test_files(data_dir: Path, test_size: float = 0.2, seed: int = 42):
    """Use the same hash-bucket split as `hybrid.evaluate` so baseline numbers
    are directly comparable to the headline pipeline numbers."""
    files_by_type: dict[str, list[Path]] = {}
    for ct in ("pyramidal", "interneuron"):
        ct_dir = data_dir / ct
        swc_dir = ct_dir / "swc" if (ct_dir / "swc").is_dir() else ct_dir
        if not swc_dir.is_dir():
            continue
        files_by_type[ct] = sorted(swc_dir.glob("*.swc"))
    train_files, test_files = _file_level_split(files_by_type, test_size, seed)
    return train_files, test_files


def _gt_class_distribution(train_files: dict[str, list[Path]]) -> dict[int, float]:
    """Class-frequency prior on the train split (excluding soma)."""
    counts: Counter = Counter()
    for ct, files in train_files.items():
        for f in files:
            for nd in parse_swc(f):
                if nd.type != 1:
                    counts[nd.type] += 1
    total = sum(counts.values()) or 1
    return {lbl: c / total for lbl, c in counts.items()}


def _aggregate_per_file(
    test_files: dict[str, list[Path]],
    predict_fn,
) -> tuple[list[dict], list[int], list[int], dict[str, list[int]], dict[str, list[int]]]:
    """Apply predict_fn(nodes, cell_type) → list[int] to every test file.

    Returns (file_rows, all_gt, all_pred, gt_by_ct, pred_by_ct).
    """
    file_rows: list[dict] = []
    all_gt: list[int] = []
    all_pred: list[int] = []
    gt_by_ct: dict[str, list[int]] = defaultdict(list)
    pred_by_ct: dict[str, list[int]] = defaultdict(list)

    for ct, files in test_files.items():
        for f in files:
            nodes = parse_swc(f)
            if not nodes:
                continue
            gt = [nd.type for nd in nodes]
            pred = predict_fn(nodes, ct)
            assert len(pred) == len(gt), f"{f.name}: pred {len(pred)} != gt {len(gt)}"

            valid = set(CELL_TYPE_LABEL_SETS.get(ct, {1, 2, 3}))
            metrics = _compute_metrics(gt, pred, valid)
            try:
                rel_path = str(f.relative_to(ROOT))
            except ValueError:
                # Symlinks/abs paths from non-canonical data dirs (e.g. v9 merged
                # dataset) may not be under ROOT — use the full path as-is.
                rel_path = str(f)
            file_rows.append({
                "path": rel_path,
                "cell_type": ct,
                "n_nodes": len(nodes),
                "neurite_macro_f1": float(metrics.get("neurite_macro_f1", 0.0)),
                "macro_f1": float(metrics.get("macro_f1", 0.0)),
            })

            all_gt.extend(gt)
            all_pred.extend(pred)
            gt_by_ct[ct].extend(gt)
            pred_by_ct[ct].extend(pred)

    return file_rows, all_gt, all_pred, gt_by_ct, pred_by_ct


def _build_result(
    method_name: str,
    file_rows: list[dict],
    all_gt: list[int],
    all_pred: list[int],
    gt_by_ct: dict[str, list[int]],
    pred_by_ct: dict[str, list[int]],
) -> dict:
    overall_valid = {1, 2, 3, 4}
    overall = _compute_metrics(all_gt, all_pred, overall_valid)
    per_ct = {}
    for ct in gt_by_ct:
        valid = set(CELL_TYPE_LABEL_SETS.get(ct, {1, 2, 3}))
        per_ct[ct] = _compute_metrics(gt_by_ct[ct], pred_by_ct[ct], valid)

    file_summary = _summarize_file_scores(file_rows)

    # Pull per-class F1 from overall metrics (numbers reviewers cite)
    per_class = {
        name: overall["per_label"].get(name, {}).get("f1", None)
        for name in ("soma", "axon", "basal/dendrite", "apical")
    }

    return {
        "method": method_name,
        "accuracy": overall["accuracy"],
        "macro_f1": overall["macro_f1"],
        "neurite_macro_f1": overall["neurite_macro_f1"],
        "per_class_f1": per_class,
        "per_cell_type": {
            ct: {
                "accuracy": m["accuracy"],
                "neurite_macro_f1": m["neurite_macro_f1"],
            }
            for ct, m in per_ct.items()
        },
        "per_file": {
            "mean": file_summary.get("mean", 0.0),
            "median": file_summary.get("median", 0.0),
            "p10": file_summary.get("p10", 0.0),
            "n_files": file_summary.get("n_files", 0),
        },
        "confusion": overall["confusion"],
    }


# =============================================================================
# TIER 1 — Floor baselines
# =============================================================================

def predict_random(class_freq: dict[int, float], rng: np.random.Generator):
    """Predict each non-soma node by sampling from the train-set class frequency."""
    labels = list(class_freq.keys())
    probs = np.array([class_freq[l] for l in labels], dtype=np.float64)
    probs = probs / probs.sum()

    def fn(nodes, cell_type):
        valid = set(CELL_TYPE_LABEL_SETS.get(cell_type, {1, 2, 3}))
        # Restrict random sampling to the cell type's allowed labels
        ct_labels = [l for l in labels if l in valid and l != 1]
        ct_probs = np.array([class_freq[l] for l in ct_labels], dtype=np.float64)
        if ct_probs.sum() <= 0:
            ct_labels = sorted(valid - {1}) or [3]
            ct_probs = np.ones(len(ct_labels)) / len(ct_labels)
        else:
            ct_probs = ct_probs / ct_probs.sum()

        out: list[int] = []
        for nd in nodes:
            if nd.type == 1:
                out.append(1)  # soma is given by the proxy-root rule downstream
            else:
                out.append(int(rng.choice(ct_labels, p=ct_probs)))
        return out

    return fn


def predict_majority(class_freq: dict[int, float]):
    """Predict every non-soma node as the most-common neurite class on train."""
    most_common = max(
        ((l, p) for l, p in class_freq.items() if l != 1),
        key=lambda kv: kv[1],
        default=(2, 0),
    )[0]

    def fn(nodes, cell_type):
        valid = set(CELL_TYPE_LABEL_SETS.get(cell_type, {1, 2, 3}))
        # Pick the label allowed for this cell type with highest prior weight
        ct_labels = [(l, class_freq.get(l, 0.0)) for l in valid if l != 1]
        if ct_labels:
            most_common_ct = max(ct_labels, key=lambda kv: kv[1])[0]
        else:
            most_common_ct = most_common
        return [1 if nd.type == 1 else most_common_ct for nd in nodes]

    return fn


def predict_heuristic():
    """Pure rule-based labeler.

    For each cell:
      - Soma: nodes with original type == 1, or the proxy soma node.
      - Axon: the primary subtree with the smallest mean radius (thin chains).
      - Apical: the primary subtree with the largest +z extent above soma
                (pyramidal only).
      - Basal: every other primary subtree.
    """
    import math

    def fn(nodes, cell_type):
        if not nodes:
            return []
        # Build adjacency
        id_to_idx = {n.id: i for i, n in enumerate(nodes)}
        children: list[list[int]] = [[] for _ in nodes]
        roots: list[int] = []
        for i, nd in enumerate(nodes):
            pidx = id_to_idx.get(nd.parent)
            if pidx is None:
                roots.append(i)
            else:
                children[pidx].append(i)

        # Pick proxy soma: largest-radius root (label-free). Falls back to node 0.
        proxy = max(roots, key=lambda i: nodes[i].radius) if roots else 0
        soma_z = nodes[proxy].z

        # Find primary subtrees (children of proxy soma)
        primaries = list(children[proxy])

        # Walk each primary, gather: total path, mean radius, max-z-above-soma
        sub_stats = {}
        sub_nodes_of: dict[int, list[int]] = {}
        for pr in primaries:
            stack = [pr]
            sub_nodes: list[int] = []
            radii_sum = 0.0
            n_count = 0
            max_z = -math.inf
            while stack:
                idx = stack.pop()
                sub_nodes.append(idx)
                radii_sum += nodes[idx].radius
                n_count += 1
                z_above = nodes[idx].z - soma_z
                if z_above > max_z:
                    max_z = z_above
                for ci in children[idx]:
                    stack.append(ci)
            sub_stats[pr] = {
                "mean_radius": radii_sum / max(1, n_count),
                "max_z_above": max_z,
            }
            sub_nodes_of[pr] = sub_nodes

        valid = set(CELL_TYPE_LABEL_SETS.get(cell_type, {1, 2, 3}))
        labels = [3] * len(nodes)  # default basal
        # Soma nodes (preserve any explicitly-soma node in case Stage 1 missed)
        labels[proxy] = 1

        if not primaries:
            return labels

        # Axon = primary with smallest mean radius
        axon_pr = min(primaries, key=lambda pr: sub_stats[pr]["mean_radius"])
        for idx in sub_nodes_of[axon_pr]:
            labels[idx] = 2

        # Apical = primary with largest +z extent (pyramidal only, ≥ small threshold)
        if 4 in valid:
            apical_pr = max(primaries, key=lambda pr: sub_stats[pr]["max_z_above"])
            if apical_pr != axon_pr and sub_stats[apical_pr]["max_z_above"] > 5.0:
                for idx in sub_nodes_of[apical_pr]:
                    labels[idx] = 4

        # Anything else is basal/dendrite (already default)
        return labels

    return fn


# =============================================================================
# TIER 2 — Ablations
# =============================================================================

def ablation_no_stage3(eval_results_path: Path) -> dict | None:
    """Read the Stage-2-only metrics from an existing evaluation_results.json.

    Same train/test split, just bypasses Stage 3 — already computed in the
    existing `evaluate.py` run.
    """
    if not eval_results_path.exists():
        print(f"  skip — {eval_results_path.name} not found", file=sys.stderr)
        return None

    with open(eval_results_path) as f:
        d = json.load(f)

    overall = d.get("overall_stage2", {})
    pf = d.get("overall_per_file_stage2", {})
    if not overall:
        return None

    return {
        "method": "no-stage3 (Stage 2 only)",
        "accuracy": overall.get("accuracy", 0.0),
        "macro_f1": overall.get("macro_f1", 0.0),
        "neurite_macro_f1": overall.get("neurite_macro_f1", 0.0),
        "per_class_f1": {
            name: overall.get("per_label", {}).get(name, {}).get("f1", None)
            for name in ("soma", "axon", "basal/dendrite", "apical")
        },
        "per_cell_type": {
            ct: {
                "accuracy": d["per_cell_type"][ct]["stage2"]["accuracy"],
                "neurite_macro_f1": d["per_cell_type"][ct]["stage2"]["neurite_macro_f1"],
            }
            for ct in d.get("per_cell_type", {})
        },
        "per_file": {
            "mean": pf.get("mean", 0.0),
            "median": pf.get("median", 0.0),
            "p10": pf.get("p10", 0.0),
            "n_files": pf.get("n_files", 0),
        },
        "confusion": overall.get("confusion", {}),
    }


def ablation_no_soft_handoff(
    data_dir: Path,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Re-evaluate using the existing eval_tmp models but with soft-handoff disabled.

    Reuses the Stage 1 and Stage 2 models trained for the headline run
    (in `hybrid/models/eval_tmp/`), so this ablation runs in ~5 min instead
    of ~40 min. The only difference is `soft_handoff_threshold=0.0` —
    forcing a hard Stage-1 cascade.

    If the eval_tmp pickles were saved by a different sklearn version than
    the current environment, raises a RuntimeError with instructions to
    regenerate them.
    """
    s1_model = ROOT / "hybrid" / "models" / "eval_tmp" / "s1_eval.pkl"
    s2_model = ROOT / "hybrid" / "models" / "eval_tmp" / "s2_eval.pkl"
    if not s1_model.exists() or not s2_model.exists():
        raise FileNotFoundError(
            f"need eval_tmp models from a prior `python -m hybrid.evaluate` run "
            f"before this ablation can use them: {s1_model}, {s2_model}"
        )

    # Probe-load the bundle up front so we fail fast with a useful message
    # rather than mid-run after processing many files.
    try:
        with open(s2_model, "rb") as fh:
            pickle.load(fh)
    except (AttributeError, ModuleNotFoundError, ImportError, ValueError, EOFError) as e:
        raise RuntimeError(
            "Cannot load hybrid/models/eval_tmp/s2_eval.pkl — "
            "the pickle was saved by a different sklearn version than the "
            "one in your current environment.\n\n"
            f"  Underlying error: {type(e).__name__}: {e}\n\n"
            "  Fix: regenerate the eval_tmp models in this environment by running\n"
            "    python -m hybrid.evaluate\n"
            "  (~30–60 min). Then rerun this baseline command."
        ) from e

    train_files, test_files = _collect_test_files(data_dir, test_size, seed)

    def predict_fn(nodes, cell_type):
        # Bypass soft handoff by setting threshold = 0.0
        from hybrid.pipeline import run_pipeline_on_nodes
        result = run_pipeline_on_nodes(
            nodes, "", s1_model, s2_model,
            soft_handoff_threshold=0.0,
        )
        return result.node_labels

    file_rows, all_gt, all_pred, gt_by_ct, pred_by_ct = _aggregate_per_file(
        test_files, predict_fn,
    )
    return _build_result(
        "no-soft-handoff (hard Stage 1 cascade)",
        file_rows, all_gt, all_pred, gt_by_ct, pred_by_ct,
    )


def ablation_feature_subset_instructions() -> str:
    """Return guidance for the heavier feature-ablation runs.

    `no-pca`, `no-trunk`, `no-cell-type` each require a fresh ~40-min Stage 2
    retrain. Rather than embed that runtime here, this prints commands you
    can run manually and paste back into the results file.
    """
    return (
        "Heavy ablations (require fresh ~40-min retrain each):\n"
        "  no-pca           → comment out PCA features in features.py & branch_features.py,\n"
        "                      run  `python -m hybrid.evaluate`,\n"
        "                      copy `evaluation_results.json` to `eval_no_pca.json`,\n"
        "                      then `python -m hybrid.baselines ablation --import-eval eval_no_pca.json`.\n"
        "  no-trunk         → same procedure with the 4 trunk features removed.\n"
        "  no-cell-type     → train a single Stage 2 model on all data;\n"
        "                      easiest path is to set `models_by_cell_type` to one merged\n"
        "                      model in `_train_stage2`. Workflow same as above.\n"
    )


# =============================================================================
# Aggregate runner + table writer
# =============================================================================

def write_table(results: list[dict], out_path: Path) -> None:
    cols = ["method", "accuracy", "macro_f1", "neurite_f1",
            "soma", "axon", "basal", "apical",
            "per_file_mean", "per_file_p10"]
    widths = [42, 9, 9, 11, 7, 7, 7, 7, 14, 12]

    def fmt_cell(v, width):
        if v is None:
            s = "—"
        elif isinstance(v, float):
            s = f"{v:.4f}"
        else:
            s = str(v)
        return s.ljust(width)

    lines = []
    header = " ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "─" * len(header)
    lines.append(header)
    lines.append(sep)
    for r in results:
        cells = [
            r["method"],
            r.get("accuracy"),
            r.get("macro_f1"),
            r.get("neurite_macro_f1"),
            r.get("per_class_f1", {}).get("soma"),
            r.get("per_class_f1", {}).get("axon"),
            r.get("per_class_f1", {}).get("basal/dendrite"),
            r.get("per_class_f1", {}).get("apical"),
            r.get("per_file", {}).get("mean"),
            r.get("per_file", {}).get("p10"),
        ]
        lines.append(" ".join(fmt_cell(v, w) for v, w in zip(cells, widths)))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# CLI
# =============================================================================

def _cmd_floor(args) -> int:
    train_files, test_files = _collect_test_files(args.data_dir, args.test_size, args.seed)
    print(f"Train files: {sum(len(v) for v in train_files.values())}")
    print(f"Test files:  {sum(len(v) for v in test_files.values())}")

    class_freq = _gt_class_distribution(train_files)
    print(f"Train-set class frequency: { {LABEL_NAMES.get(k, k): round(v, 3) for k, v in class_freq.items()} }")

    rng = np.random.default_rng(args.seed)
    methods = []
    print("\n=== Tier 1 floor baselines ===")
    for name, predict_factory in [
        ("random",    lambda: predict_random(class_freq, rng)),
        ("majority",  lambda: predict_majority(class_freq)),
        ("heuristic", lambda: predict_heuristic()),
    ]:
        print(f"\n  Running {name} ...")
        predict_fn = predict_factory()
        file_rows, all_gt, all_pred, gt_by_ct, pred_by_ct = _aggregate_per_file(
            test_files, predict_fn,
        )
        result = _build_result(name, file_rows, all_gt, all_pred, gt_by_ct, pred_by_ct)
        print(f"    accuracy:        {result['accuracy']:.4f}")
        print(f"    neurite F1:      {result['neurite_macro_f1']:.4f}")
        print(f"    per-file mean:   {result['per_file']['mean']:.4f}")
        print(f"    per-file P10:    {result['per_file']['p10']:.4f}")
        methods.append(result)

    _save_and_print(methods, args.out, args.table)
    return 0


def _cmd_ablation(args) -> int:
    eval_path = ROOT / "hybrid" / "models" / "evaluation_results.json"

    methods = []
    targets = [args.which] if args.which else ["no-stage3", "no-soft-handoff"]

    if "no-stage3" in targets:
        print("=== no-stage3 (read from existing eval JSON) ===")
        r = ablation_no_stage3(eval_path)
        if r:
            print(f"  neurite F1:    {r['neurite_macro_f1']:.4f}")
            print(f"  per-file mean: {r['per_file']['mean']:.4f}")
            print(f"  per-file P10:  {r['per_file']['p10']:.4f}")
            methods.append(r)

    if "no-soft-handoff" in targets:
        print("\n=== no-soft-handoff (rerun with threshold=0.0 on eval_tmp models) ===")
        try:
            r = ablation_no_soft_handoff(args.data_dir, args.test_size, args.seed)
            print(f"  accuracy:      {r['accuracy']:.4f}")
            print(f"  neurite F1:    {r['neurite_macro_f1']:.4f}")
            print(f"  per-file mean: {r['per_file']['mean']:.4f}")
            print(f"  per-file P10:  {r['per_file']['p10']:.4f}")
            methods.append(r)
        except FileNotFoundError as e:
            print(f"  skipped: {e}", file=sys.stderr)

    print("\n" + ablation_feature_subset_instructions())

    if methods:
        _save_and_print(methods, args.out, args.table, append=True)
    return 0


def _cmd_all(args) -> int:
    print("=== Running all baselines ===")
    floor_args = argparse.Namespace(
        data_dir=args.data_dir, test_size=args.test_size, seed=args.seed,
        out=args.out, table=args.table,
    )
    rc1 = _cmd_floor(floor_args)
    if rc1 != 0:
        return rc1

    abl_args = argparse.Namespace(
        data_dir=args.data_dir, test_size=args.test_size, seed=args.seed,
        which=None, out=args.out, table=args.table,
    )
    rc2 = _cmd_ablation(abl_args)

    # Add the headline pipeline numbers from the existing eval JSON
    eval_path = ROOT / "hybrid" / "models" / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path) as f:
            d = json.load(f)
        overall = d.get("overall_stage23", {})
        pf = d.get("overall_per_file_stage23", {})
        headline = {
            "method": "FULL PIPELINE (Stage 1+2+3, this work)",
            "accuracy": overall.get("accuracy", 0.0),
            "macro_f1": overall.get("macro_f1", 0.0),
            "neurite_macro_f1": overall.get("neurite_macro_f1", 0.0),
            "per_class_f1": {
                name: overall.get("per_label", {}).get(name, {}).get("f1", None)
                for name in ("soma", "axon", "basal/dendrite", "apical")
            },
            "per_file": {
                "mean": pf.get("mean", 0.0),
                "p10": pf.get("p10", 0.0),
                "median": pf.get("median", 0.0),
                "n_files": pf.get("n_files", 0),
            },
            "confusion": overall.get("confusion", {}),
        }
        # Append headline to results JSON
        with open(args.out) as f:
            current = json.load(f)
        current.append(headline)
        with open(args.out, "w") as f:
            json.dump(current, f, indent=2)
        write_table(current, args.table)
        print(f"\nFinal table: {args.table}")
        print(f"Full JSON:   {args.out}")

    return rc2


def _save_and_print(methods: list[dict], out_path: Path, table_path: Path, append: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if append and out_path.exists():
        with open(out_path) as f:
            current = json.load(f)
        current.extend(methods)
    else:
        current = methods
    with open(out_path, "w") as f:
        json.dump(current, f, indent=2)
    write_table(current, table_path)
    print(f"\nSaved → {out_path}")
    print(f"Table → {table_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="paper.baselines",
        description="Floor baselines and ablations for the hybrid pipeline.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)

    sub = parser.add_subparsers(dest="cmd", required=True, metavar="<command>")

    p_floor = sub.add_parser("floor", help="Run all Tier 1 floor baselines.")
    p_floor.set_defaults(func=_cmd_floor)

    p_abl = sub.add_parser("ablation", help="Run Tier 2 ablations.")
    p_abl.add_argument("--which", choices=["no-stage3", "no-soft-handoff"],
                       default=None,
                       help="Run a single ablation (default: all available).")
    p_abl.set_defaults(func=_cmd_ablation)

    p_all = sub.add_parser("all", help="Run everything available.")
    p_all.set_defaults(func=_cmd_all)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
