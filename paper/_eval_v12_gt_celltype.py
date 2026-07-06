#!/usr/bin/env python3
"""Pure v12 eval with GT cell-type override — NO training.

Loads existing v12 models from `paper/models/v12_gentle_seed<seed>/` and runs
the full pipeline on the seed's held-out test cells, BYPASSING Stage 1 by
passing each cell's ground-truth cell type as `override_cell_type`.

This isolates Stage 2 + Stage 3 (GNN) quality so the comparison against
baselines (which always receive GT cell type at inference) is fair.

Usage:
    python -m paper._eval_v12_gt_celltype --seed 42

Output:
    paper/results/v12_gt_celltype_seed<seed>.json
        single comprehensive metric report

Wall time: ~25 min for ~2,500 cells. NO retraining.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.features import parse_swc                                # noqa: E402
from hybrid.pipeline import run_pipeline_on_nodes                    # noqa: E402
from hybrid.evaluate import per_cell_neurite_f1, per_cell_per_class_f1  # noqa: E402
from hybrid.comprehensive_metrics import (                           # noqa: E402
    build_full_report, print_summary_table, print_per_class_per_cell,
    print_per_cell_type,
)
from paper.gnn_inference import load_gnn                             # noqa: E402
from paper.gnn_branch3_inference import load_branch3                 # noqa: E402

DATA_DIR = ROOT / "data" / "v12_uncurated"
# SWCAL_MODEL_DIR_SUFFIX appends to the model dir name (e.g. "_cleaned")
MODEL_DIR_SUFFIX = os.environ.get("SWCAL_MODEL_DIR_SUFFIX", "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--branch3-ckpt", type=Path, default=None,
                    help="Optional Branch3 rescue checkpoint. Defaults to "
                         "<model-dir>/gnn_branch3_rescue.pt if present.")
    ap.add_argument("--branch3-gate", type=Path, default=None,
                    help="Optional learned Branch3 accept/abstain gate joblib.")
    ap.add_argument("--no-branch3", action="store_true",
                    help="Disable automatic Branch3 rescue loading.")
    ap.add_argument("--out-suffix", default="",
                    help="Output-only suffix appended after seed, e.g. _no_branch3.")
    args = ap.parse_args()
    seed = args.seed

    model_dir = ROOT / "paper" / "models" / f"v12_gentle_seed{seed}{MODEL_DIR_SUFFIX}"
    split_json = model_dir / "train_test_split.json"
    s1_path = model_dir / "cell_type_classifier.pkl"
    s2_path = model_dir / "branch_classifier.pkl"
    gnn_path = model_dir / "gnn_apical_basal.pt"
    for p in (split_json, s1_path, s2_path, gnn_path):
        if not p.is_file():
            raise SystemExit(
                f"MISSING: {p}\n"
                f"Train v12 seed={seed} first with:\n"
                f"  python -m paper._retrain_v12_gentle_seed --seed {seed}"
            )

    out_suffix = args.out_suffix
    out_json = ROOT / "paper" / "results" / f"v12_gt_celltype_seed{seed}{MODEL_DIR_SUFFIX}{out_suffix}.json"
    out_csv = ROOT / "paper" / "results" / f"v12_gt_celltype_seed{seed}{MODEL_DIR_SUFFIX}{out_suffix}.csv"

    sp = json.loads(split_json.read_text(encoding="utf-8"))
    test_files: list[tuple[str, Path]] = []
    for ct in ("pyramidal", "interneuron"):
        for fn in sp["test"].get(ct, []):
            p = DATA_DIR / ct / "swc" / fn
            if p.is_file():
                test_files.append((ct, p))
    print(f"=== v12 seed={seed} eval with GT cell-type override ===")
    print(f"  test cells: {len(test_files)}")

    print(f"  loading models from {model_dir.name}/...")
    gnn_state = load_gnn(gnn_path)
    branch3_state = None
    branch3_path = args.branch3_ckpt
    if branch3_path is None and not args.no_branch3:
        auto_branch3 = model_dir / "gnn_branch3_rescue.pt"
        if auto_branch3.is_file():
            branch3_path = auto_branch3
    if branch3_path is not None:
        branch3_path = branch3_path.resolve()
        if not branch3_path.is_file():
            raise SystemExit(f"MISSING: {branch3_path}")
        branch3_state = load_branch3(branch3_path, gate_path=args.branch3_gate)
        print(f"  branch3: {branch3_path.relative_to(ROOT)}")
        if args.branch3_gate is not None:
            print(f"  branch3 gate: {args.branch3_gate.resolve().relative_to(ROOT)}")

    cell_records: list[dict] = []
    per_cell_rows: list[dict] = []
    gt_pool: list[int] = []
    pred_pool: list[int] = []
    t0 = time.perf_counter()
    for i, (ct_gt, p) in enumerate(test_files):
        try:
            nodes = parse_swc(p)
            if not nodes:
                continue
            gt = [n.type for n in nodes]
        except Exception as exc:
            print(f"  WARN {p.name}: {exc}")
            continue
        pr = run_pipeline_on_nodes(
            nodes, file_path="", stage1_model=s1_path, stage2_model=s2_path,
            gnn_state=gnn_state, branch3_state=branch3_state,
            use_subtree_stage2=True,
            override_cell_type=ct_gt,
        )
        pred = list(pr.node_labels)
        n_correct = sum(1 for g, q in zip(gt, pred) if g == q)
        per_class_f1 = per_cell_per_class_f1(gt, pred, ct_gt)
        cell_records.append({
            "cell_type": ct_gt, "n_nodes": len(gt),
            "gt": gt, "pred": pred,
        })
        per_cell_rows.append({
            "file": p.name,
            "cell_type_gt": ct_gt,
            "n_nodes": len(gt),
            "accuracy": n_correct / max(1, len(gt)),
            "neurite_macro_f1": per_cell_neurite_f1(gt, pred, ct_gt),
            "soma_f1": per_class_f1.get("soma"),
            "axon_f1": per_class_f1.get("axon"),
            "basal_f1": per_class_f1.get("basal/dendrite"),
            "apical_f1": per_class_f1.get("apical"),
            "pred_soma": sum(1 for q in pred if q == 1),
            "pred_axon": sum(1 for q in pred if q == 2),
            "pred_basal": sum(1 for q in pred if q == 3),
            "pred_apical": sum(1 for q in pred if q == 4),
        })
        gt_pool.extend(gt); pred_pool.extend(pred)
        if (i + 1) % 200 == 0:
            print(f"    ... {i+1}/{len(test_files)} ({(time.perf_counter()-t0)/60:.1f} min)")
    elapsed = (time.perf_counter() - t0) / 60.0
    print(f"  inference: {elapsed:.1f} min")

    report = build_full_report(
        method=(f"v12_branch3_gt_celltype_seed{seed}" if branch3_state is not None
                else f"v12_gt_celltype_seed{seed}"),
        gt_pool=gt_pool, pred_pool=pred_pool,
        cell_records=cell_records, inference_min=elapsed,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if per_cell_rows:
        import csv

        fields = list(per_cell_rows[0].keys())
        with out_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for row in per_cell_rows:
                out = row.copy()
                for k, v in out.items():
                    if isinstance(v, float):
                        out[k] = f"{v:.6f}"
                    elif v is None:
                        out[k] = ""
                writer.writerow(out)
    out_json.write_text(json.dumps({
        "note":            "v12 Stage 2 + GNN evaluated with GT cell-type override "
                           "(Stage 1 bypassed). Apples-to-apples with baselines.",
        "model_dir":       str(model_dir.relative_to(ROOT)),
        "branch3_enabled":  branch3_state is not None,
        "branch3_ckpt":     str(branch3_path.relative_to(ROOT)) if branch3_path is not None else None,
        "branch3_gate":     str(args.branch3_gate.resolve().relative_to(ROOT)) if args.branch3_gate is not None else None,
        "out_suffix":       out_suffix,
        "n_test_cells":    len(cell_records),
        "inference_min":   elapsed,
        "per_cell_csv":     str(out_csv.relative_to(ROOT)) if per_cell_rows else None,
        "report":          report,
    }, indent=2), encoding="utf-8")

    print()
    print("=" * 120)
    print(f"  V12 seed={seed} with GT cell-type override")
    print("=" * 120)
    print_summary_table([report])
    print()
    print("=" * 120)
    print(f"  PER-CLASS PER-CELL F1 DISTRIBUTIONS")
    print("=" * 120)
    print_per_class_per_cell([report])
    print()
    print("=" * 120)
    print(f"  PER-CELL-TYPE BREAKDOWN")
    print("=" * 120)
    print_per_cell_type([report])
    print(f"\n  JSON -> {out_json}")
    if per_cell_rows:
        print(f"  CSV  -> {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
