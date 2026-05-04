"""Cross-dataset zero-shot evaluation.

Runs the v8 pipeline (Stage 1 + subtree-Stage2 + GNN + Stage 3) on a
flat directory of SWC files that ALL belong to one cell type, scored
against the SWC type column as ground truth. None of these files were
seen during training of any pipeline component, so this is a true
zero-shot generalization test (CONTINUATION §7's "cross-dataset
generalization" item).

Default configuration: production models (committed, trained on the full
benchmark), v8 architecture (--use-subtree-stage2 + --use-gnn).

Usage:
    # Evaluate hpf_ca1 (1377 CA1 pyramidal cells) zero-shot
    python -m paper.cross_dataset_eval \\
        --data-dir data/hpf_ca1 \\
        --cell-type pyramidal \\
        --tag hpf_ca1

    # Different model architecture mode (v7 = per-branch + GNN)
    python -m paper.cross_dataset_eval \\
        --data-dir data/hpf_ca1 --cell-type pyramidal \\
        --no-subtree-stage2

Outputs:
    paper/results/cross_dataset_<tag>.json   aggregate + per-class + per-file
    paper/results/cross_dataset_<tag>.csv    one row per file
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from hybrid.evaluate import (
    LABEL_NAMES,
    VALID_LABELS,
    _compute_metrics,
)
from hybrid.features import parse_swc
from hybrid.pipeline import run_pipeline_on_nodes

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = ROOT / "paper" / "results"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = int(round((pct / 100.0) * (len(arr) - 1)))
    return arr[idx]


def cross_eval(
    data_dir: Path,
    cell_type: str,
    stage1_path: Path | None,
    stage2_path: Path | None,
    gnn_path: Path | None,
    use_gnn: bool = True,
    use_subtree_stage2: bool = True,
    limit: int | None = None,
    progress_every: int = 50,
    require_classes: tuple[int, ...] = (),
) -> dict:
    """Zero-shot eval on a flat / batched dir of SWCs all assumed to be
    one cell type. Returns a dict suitable for JSON dump.

    `require_classes`: if non-empty, drop any file whose ground-truth
    type column lacks at least one of the listed SWC type codes. Use
    `(4,)` to limit the eval to cells that have apical labels.
    """
    valid_labels = VALID_LABELS.get(cell_type, {1, 2, 3, 4})

    swcs = sorted(data_dir.rglob("*.swc"))
    if limit:
        swcs = swcs[:limit]
    print(f"Found {len(swcs)} SWC files under {data_dir}")
    print(f"Assumed cell type: {cell_type}  (valid labels: {sorted(valid_labels)})")

    # Pre-filter on required GT classes (cheap pre-pass that just reads
    # type-column histograms via parse_swc).
    if require_classes:
        required = set(require_classes)
        kept: list[Path] = []
        skipped = 0
        for swc in swcs:
            try:
                nodes = parse_swc(swc)
            except Exception:
                skipped += 1
                continue
            types = {nd.type for nd in nodes}
            if required.issubset(types):
                kept.append(swc)
            else:
                skipped += 1
        print(f"Filter: require GT classes {sorted(required)} -> kept {len(kept)} / {len(swcs)} files (skipped {skipped})")
        swcs = kept

    # Optional GNN load (once)
    gnn_state = None
    if use_gnn:
        from paper.gnn_inference import load_gnn  # noqa: PLC0415
        from paper.gnn_apical_basal import DEFAULT_CKPT_PATH  # noqa: PLC0415
        ckpt = Path(gnn_path) if gnn_path else DEFAULT_CKPT_PATH
        print(f"Loading GNN from {ckpt} ...")
        gnn_state = load_gnn(ckpt)

    # Aggregates
    all_gt: list[int] = []
    all_pred: list[int] = []
    cell_type_correct = 0
    per_file: list[dict] = []
    cell_type_confusion: dict[str, int] = defaultdict(int)

    t_start = time.time()
    for i, swc in enumerate(swcs, 1):
        try:
            nodes = parse_swc(swc)
        except Exception as e:
            print(f"  [{i:>4}] SKIP {swc.name}: parse error {e}")
            continue
        if not nodes:
            continue

        gt = [nd.type for nd in nodes]

        try:
            result = run_pipeline_on_nodes(
                nodes, str(swc),
                stage1_model=stage1_path,
                stage2_model=stage2_path,
                gnn_state=gnn_state,
                use_subtree_stage2=use_subtree_stage2,
            )
        except Exception as e:
            print(f"  [{i:>4}] SKIP {swc.name}: pipeline error {e}")
            continue
        pred = result.node_labels
        pred_ct = result.stage1.cell_type
        cell_type_confusion[pred_ct] += 1
        if pred_ct == cell_type:
            cell_type_correct += 1

        all_gt.extend(gt)
        all_pred.extend(pred)

        m = _compute_metrics(gt, pred, valid_labels)
        per_file.append({
            "file": str(swc),
            "name": swc.name,
            "n_nodes": len(nodes),
            "predicted_cell_type": pred_ct,
            "stage1_correct": pred_ct == cell_type,
            "neurite_macro_f1": m["neurite_macro_f1"],
            "macro_f1": m["macro_f1"],
            "accuracy": m["accuracy"],
            "n_apical_gt": int(sum(1 for t in gt if t == 4)),
            "n_basal_gt": int(sum(1 for t in gt if t == 3)),
            "n_axon_gt": int(sum(1 for t in gt if t == 2)),
            "per_label": m["per_label"],
        })

        if i % progress_every == 0 or i == len(swcs):
            elapsed = time.time() - t_start
            rate = i / max(elapsed, 1e-6)
            eta = (len(swcs) - i) / rate if rate > 0 else 0
            print(f"  [{i:>4}/{len(swcs)}] elapsed={elapsed:.0f}s  rate={rate:.2f} files/s  eta={eta:.0f}s")

    # Overall metrics
    print("\nComputing aggregate metrics...")
    overall = _compute_metrics(all_gt, all_pred, valid_labels)

    # Per-file F1 distribution
    pf_f1 = sorted([f["neurite_macro_f1"] for f in per_file])
    pf_summary = {
        "n_files": len(per_file),
        "mean": round(float(np.mean(pf_f1)), 4) if pf_f1 else 0.0,
        "median": round(float(np.median(pf_f1)), 4) if pf_f1 else 0.0,
        "p10": round(_percentile(pf_f1, 10), 4),
        "p25": round(_percentile(pf_f1, 25), 4),
        "p75": round(_percentile(pf_f1, 75), 4),
        "p90": round(_percentile(pf_f1, 90), 4),
        "min": round(min(pf_f1), 4) if pf_f1 else 0.0,
        "max": round(max(pf_f1), 4) if pf_f1 else 0.0,
    }

    # 5 worst per-file scores
    worst5 = sorted(per_file, key=lambda f: f["neurite_macro_f1"])[:10]
    cell_type_acc = cell_type_correct / max(len(per_file), 1)

    payload = {
        "data_dir": str(data_dir),
        "assumed_cell_type": cell_type,
        "n_files_seen": len(per_file),
        "config": {
            "use_subtree_stage2": use_subtree_stage2,
            "use_gnn": use_gnn,
            "stage1_model": str(stage1_path) if stage1_path else "default",
            "stage2_model": str(stage2_path) if stage2_path else "default",
            "gnn_model": str(gnn_path) if gnn_path else "default",
        },
        "stage1_cell_type_accuracy": round(cell_type_acc, 4),
        "stage1_predicted_distribution": dict(cell_type_confusion),
        "overall": overall,
        "per_file_summary": pf_summary,
        "worst_files": [
            {"name": f["name"], "n_nodes": f["n_nodes"],
             "neurite_macro_f1": f["neurite_macro_f1"],
             "predicted_cell_type": f["predicted_cell_type"]}
            for f in worst5
        ],
    }
    return payload, per_file


def _save_per_file_csv(per_file: list[dict], path: Path) -> None:
    if not per_file:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "n_nodes", "predicted_cell_type", "stage1_correct",
              "neurite_macro_f1", "macro_f1", "accuracy",
              "n_apical_gt", "n_basal_gt", "n_axon_gt"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in per_file:
            w.writerow({k: r[k] for k in fields})


def _print_summary(payload: dict) -> None:
    print()
    print("=" * 70)
    print(f"CROSS-DATASET EVAL  ({payload['data_dir']})")
    print(f"assumed cell type: {payload['assumed_cell_type']}")
    print(f"files: {payload['n_files_seen']}")
    print(f"config: use_subtree_stage2={payload['config']['use_subtree_stage2']}  "
          f"use_gnn={payload['config']['use_gnn']}")
    print("=" * 70)
    print(f"\nStage 1 cell-type accuracy: {payload['stage1_cell_type_accuracy']:.4f}")
    print(f"  predicted distribution: {payload['stage1_predicted_distribution']}")

    o = payload["overall"]
    print(f"\nOverall (Stage 2+3, full pipeline):")
    print(f"  HEADLINE neurite-macro-F1:  {o['neurite_macro_f1']:.4f}")
    print(f"           neurite-bal-acc:    {o['neurite_balanced_accuracy']:.4f}")
    print(f"  (ref)    macro-F1:           {o['macro_f1']:.4f}")
    print(f"  (ref)    accuracy:           {o['accuracy']:.4f}  ({o['n']:,} nodes)")
    print(f"\nPer-class F1:")
    for lbl in ["soma", "axon", "basal/dendrite", "apical"]:
        if lbl in o["per_label"]:
            m = o["per_label"][lbl]
            print(f"  {lbl:<18s} F1={m['f1']:.4f}  (P={m['precision']:.4f}  R={m['recall']:.4f}  n={m['support']:,})")

    pf = payload["per_file_summary"]
    print(f"\nPer-file neurite-macro-F1:")
    print(f"  n={pf['n_files']}  mean={pf['mean']:.4f}  median={pf['median']:.4f}")
    print(f"  P10={pf['p10']:.4f}  P25={pf['p25']:.4f}  P75={pf['p75']:.4f}  P90={pf['p90']:.4f}")
    print(f"  min={pf['min']:.4f}  max={pf['max']:.4f}")

    print(f"\nWorst-10 files:")
    for f in payload["worst_files"]:
        print(f"  {f['name']:<55s} F1={f['neurite_macro_f1']:.4f} "
              f"n={f['n_nodes']:>6}  ct_pred={f['predicted_cell_type']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--cell-type", choices=["pyramidal", "interneuron"],
                    default="pyramidal")
    ap.add_argument("--tag", type=str, default="cross",
                    help="suffix for output files (cross_dataset_<tag>.json/csv)")
    ap.add_argument("--stage1-model", type=Path, default=None)
    ap.add_argument("--stage2-model", type=Path, default=None)
    ap.add_argument("--gnn-model", type=Path, default=None)
    ap.add_argument("--no-gnn", dest="use_gnn", action="store_false", default=True)
    ap.add_argument("--no-subtree-stage2", dest="use_subtree_stage2",
                    action="store_false", default=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N files (debugging)")
    ap.add_argument(
        "--require-classes", type=int, nargs="+", default=[],
        help="Only include files where every listed SWC type code is present in GT. "
             "E.g. `--require-classes 4` keeps only cells with at least one apical "
             "node, dropping CA1 partial reconstructions.",
    )
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = ap.parse_args()

    payload, per_file = cross_eval(
        args.data_dir, args.cell_type,
        args.stage1_model, args.stage2_model, args.gnn_model,
        use_gnn=args.use_gnn,
        use_subtree_stage2=args.use_subtree_stage2,
        limit=args.limit,
        require_classes=tuple(args.require_classes),
    )
    _print_summary(payload)

    json_path = args.results_dir / f"cross_dataset_{args.tag}.json"
    csv_path = args.results_dir / f"cross_dataset_{args.tag}.csv"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _save_per_file_csv(per_file, csv_path)
    print(f"\nSaved {json_path}")
    print(f"Saved {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
