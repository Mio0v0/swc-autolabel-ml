#!/usr/bin/env python3
"""One-off: eval seed-123 model in BRANCH mode (use_subtree_stage2=False),
to compare the per-branch 64-feature Stage-2 classifier against the
default subtree-mode Stage 2. Same test split, same GNN + Branch3.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hybrid.features import parse_swc
from hybrid.pipeline import run_pipeline_on_nodes
from hybrid.evaluate import per_cell_neurite_f1, per_cell_per_class_f1, _compute_metrics
from paper.gnn_inference import load_gnn
from paper.gnn_branch3_inference import load_branch3

DATA = ROOT / "data" / "v12_uncurated"
M = ROOT / "paper" / "models" / "v12_gentle_seed123"


def main() -> int:
    sp = json.loads((M / "train_test_split.json").read_text(encoding="utf-8"))
    test = []
    for ct in ("pyramidal", "interneuron"):
        for fn in sp["test"].get(ct, []):
            p = DATA / ct / "swc" / fn
            if p.is_file():
                test.append((ct, p))
    print(f"branch-mode eval: {len(test)} test cells")
    gnn = load_gnn(M / "gnn_apical_basal.pt")
    b3 = load_branch3(M / "gnn_branch3_rescue.pt", gate_path=None)
    s1, s2 = M / "cell_type_classifier.pkl", M / "branch_classifier.pkl"

    gt_pool, pred_pool, per_cell = [], [], []
    t0 = time.perf_counter()
    for i, (ct, p) in enumerate(test):
        nodes = parse_swc(p)
        if not nodes:
            continue
        gt = [n.type for n in nodes]
        pr = run_pipeline_on_nodes(
            nodes, file_path="", stage1_model=s1, stage2_model=s2,
            gnn_state=gnn, branch3_state=b3,
            use_subtree_stage2=False,          # <-- BRANCH MODE (64-feat classifier)
            override_cell_type=ct,
        )
        pred = list(pr.node_labels)
        per_cell.append(per_cell_neurite_f1(gt, pred, ct))
        gt_pool.extend(gt); pred_pool.extend(pred)
        if (i + 1) % 400 == 0:
            print(f"  ... {i+1}/{len(test)} ({(time.perf_counter()-t0)/60:.1f} min)")
    m = _compute_metrics(gt_pool, pred_pool, {1, 2, 3, 4})
    pl = m.get("per_label", {})
    arr = np.array(per_cell)
    out = {
        "mode": "branch (use_subtree_stage2=False)",
        "n_test": len(per_cell),
        "node_accuracy": round(m["accuracy"], 4),
        "neurite_macro_f1": round(m["neurite_macro_f1"], 4),
        "axon_f1": round(pl.get("axon", {}).get("f1", 0.0), 4),
        "basal_f1": round(pl.get("basal/dendrite", {}).get("f1", 0.0), 4),
        "apical_f1": round(pl.get("apical", {}).get("f1", 0.0), 4),
        "per_cell_f1_mean": round(float(arr.mean()), 4),
        "per_cell_f1_p10": round(float(np.quantile(arr, 0.10)), 4),
    }
    (ROOT / "paper" / "results" / "branch_mode_seed123.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
