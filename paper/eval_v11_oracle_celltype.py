#!/usr/bin/env python3
"""Fairness ablation: evaluate v11_final pipeline with ORACLE cell type.

Bypasses Stage 1's cell-type prediction by feeding the ground-truth cell type
(derived from the test-file's parent directory: ``interneuron/`` or
``pyramidal/``) directly to Stage 2 + GNN + Stage 3. This matches the
information available to the external baselines, which receive
ground-truth cell type as an input one-hot feature.

Reuses v11_final's trained Stage 2 + GNN — no retraining needed.

Inputs:
    hybrid/models/eval_tmp/s1_eval.pkl          (Stage 1; ignored but loaded for compatibility)
    hybrid/models/eval_tmp/s2_eval.pkl          (Stage 2 trained for v11_final)
    paper/models/gnn_apical_basal_v11_final.pt  (GNN trained for v11_final)
    hybrid/models/eval_split.json               (test file list)

Outputs:
    paper/results/snapshots/eval_v11_final_oracle_celltype.json
    paper/results/snapshots/eval_v11_final_oracle_celltype_per_file.csv

Usage::

    python -m paper.eval_v11_oracle_celltype
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid.features import SWCNode, parse_swc                                   # noqa: E402
from hybrid.evaluate import _compute_metrics, _summarize_file_scores             # noqa: E402
from hybrid.cell_type_detector import (                                          # noqa: E402
    CellTypeResult,
    CELL_TYPE_LABEL_SETS,
)
import hybrid.pipeline as pipeline                                               # noqa: E402

DATA_DIR = Path("D:/Desktop/SWC-Studio/data/v10_dedup_dataset")
STAGE1_PATH = ROOT / "hybrid" / "models" / "eval_tmp" / "s1_eval.pkl"
STAGE2_PATH = ROOT / "hybrid" / "models" / "eval_tmp" / "s2_eval.pkl"
GNN_PATH = ROOT / "paper" / "models" / "gnn_apical_basal_v11_final.pt"
SPLIT_PATH = ROOT / "hybrid" / "models" / "eval_split.json"
SNAPSHOT_DIR = ROOT / "paper" / "results" / "snapshots"


def _oracle_stage1_factory(gt_cell_type: str):
    """Build a fake detect_cell_type_from_nodes that returns the ground-truth
    cell type with confidence 1.0. Used to monkeypatch hybrid.pipeline so the
    rest of the pipeline runs unchanged."""
    label_set = set(CELL_TYPE_LABEL_SETS.get(gt_cell_type, {1, 2, 3, 4}))
    probs = {gt_cell_type: 1.0}
    def _fake_detect(nodes, model_path=None):
        return CellTypeResult(
            cell_type=gt_cell_type,
            confidence=1.0,
            probabilities=probs,
            label_set=label_set,
            structure_flags={"oracle_celltype": True},
            features={},
        )
    return _fake_detect


def _load_gnn(gnn_path: Path):
    """Load the GraphSAGE checkpoint using the proper wrapper so it has
    the .metadata / .model / .scaler attributes the pipeline expects."""
    from paper.gnn_inference import load_gnn  # noqa: PLC0415
    return load_gnn(gnn_path)


def main():
    print(f"Loading test split from {SPLIT_PATH} ...")
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    test_files_by_ct: dict[str, list[Path]] = {}
    for ct, fnames in split.get("test_files", {}).items():
        ct_dir = DATA_DIR / ct / "swc"
        test_files_by_ct[ct] = [ct_dir / name for name in fnames if (ct_dir / name).is_file()]
        print(f"  {ct}: {len(test_files_by_ct[ct])} files")

    print(f"\nLoading GNN: {GNN_PATH}")
    gnn_state = _load_gnn(GNN_PATH)

    # Run inference with ORACLE cell type per-file
    file_rows: list[dict] = []
    all_gt: list[int] = []
    all_pred: list[int] = []
    gt_by_ct: dict[str, list[int]] = defaultdict(list)
    pred_by_ct: dict[str, list[int]] = defaultdict(list)

    total = sum(len(v) for v in test_files_by_ct.values())
    t0 = time.perf_counter()
    seen = 0
    for ct, files in test_files_by_ct.items():
        print(f"\nRunning oracle-celltype inference on {len(files)} {ct} files ...")
        # Monkeypatch the cell-type detector to return ground truth for this batch
        original_detect = pipeline.detect_cell_type_from_nodes
        pipeline.detect_cell_type_from_nodes = _oracle_stage1_factory(ct)
        try:
            for i, f in enumerate(files):
                if not f.is_file():
                    continue
                nodes = parse_swc(f)
                if not nodes:
                    continue
                gt = [nd.type for nd in nodes]
                try:
                    result = pipeline.run_pipeline_on_nodes(
                        nodes,
                        file_path=str(f),
                        stage1_model=STAGE1_PATH,
                        stage2_model=STAGE2_PATH,
                        gnn_state=gnn_state,
                        use_subtree_stage2=True,
                    )
                    pred = result.node_labels
                except Exception as exc:
                    print(f"  skip {f.name}: {exc}")
                    continue

                assert len(pred) == len(gt), f"{f.name}: pred {len(pred)} != gt {len(gt)}"
                valid = set(CELL_TYPE_LABEL_SETS.get(ct, {1, 2, 3, 4}))
                metrics = _compute_metrics(gt, pred, valid)
                file_rows.append({
                    "path": str(f),
                    "cell_type": ct,
                    "n_nodes": len(nodes),
                    "neurite_macro_f1_stage23": metrics.get("neurite_macro_f1", 0.0),
                    "macro_f1_stage23": metrics.get("macro_f1", 0.0),
                    "accuracy_stage23": metrics.get("accuracy", 0.0),
                })
                all_gt.extend(gt)
                all_pred.extend(pred)
                gt_by_ct[ct].extend(gt)
                pred_by_ct[ct].extend(pred)

                seen += 1
                if seen % 50 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = seen / elapsed if elapsed > 0 else 0
                    eta = (total - seen) / rate if rate > 0 else 0
                    print(f"  [{seen:4d}/{total}] elapsed={elapsed:.0f}s rate={rate:.2f} files/s eta={eta:.0f}s")
        finally:
            pipeline.detect_cell_type_from_nodes = original_detect

    # Overall metrics
    overall = _compute_metrics(all_gt, all_pred, {1, 2, 3, 4})
    per_ct_metrics = {
        ct: _compute_metrics(
            gt_by_ct[ct], pred_by_ct[ct],
            set(CELL_TYPE_LABEL_SETS.get(ct, {1, 2, 3, 4})),
        )
        for ct in gt_by_ct
    }
    per_file_summary = _summarize_file_scores([
        {"path": r["path"],
         "neurite_macro_f1": r["neurite_macro_f1_stage23"],
         "n_nodes": r["n_nodes"]}
        for r in file_rows
    ])

    payload = {
        "method": "v11_final_oracle_celltype",
        "n_test_files": len(file_rows),
        "note": (
            "Fairness ablation: v11_final pipeline run with ground-truth cell type "
            "in place of Stage 1 prediction. Matches the cell-type information "
            "available to external baselines."
        ),
        "overall_stage23": overall,
        "overall_per_file_stage23": per_file_summary,
        "per_cell_type": per_ct_metrics,
    }
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = SNAPSHOT_DIR / "eval_v11_final_oracle_celltype.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")

    # Per-file CSV
    out_csv = SNAPSHOT_DIR / "eval_v11_final_oracle_celltype_per_file.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "cell_type", "n_nodes",
                                            "neurite_macro_f1_stage23",
                                            "macro_f1_stage23", "accuracy_stage23"])
        w.writeheader()
        for r in file_rows:
            w.writerow(r)
    print(f"Wrote {out_csv}")

    # Print headline
    print()
    print("=" * 60)
    print("v11_final WITH ORACLE CELL TYPE — held-out test split")
    print("=" * 60)
    print(f"  n_files                 = {len(file_rows)}")
    print(f"  accuracy                = {overall.get('accuracy', 0):.4f}")
    print(f"  macro_f1                = {overall.get('macro_f1', 0):.4f}")
    print(f"  neurite_macro_f1        = {overall.get('neurite_macro_f1', 0):.4f}")
    print(f"  per-file mean F1        = {per_file_summary.get('mean', 0):.4f}")
    print(f"  per-file median F1      = {per_file_summary.get('median', 0):.4f}")
    print(f"  per-file P10            = {per_file_summary.get('p10', 0):.4f}")
    print()
    print("Compare to v11_final (Stage-1-predicted cell type):")
    print("  accuracy=0.9934  neurite_F1=0.9673  pf_mean=0.9586  P10=0.8998")


if __name__ == "__main__":
    sys.exit(main())
