#!/usr/bin/env python3
"""Head-to-head: subtree-mode Stage 2 vs branch-mode (64-feature) Stage 2,
on the SAME deterministic subsample of the seed-123 test set. Same GNN +
Branch3 in both. Answers: does the per-branch classifier help accuracy?
"""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hybrid.features import parse_swc
from hybrid.pipeline import run_pipeline_on_nodes
from hybrid.evaluate import per_cell_neurite_f1, _compute_metrics
from paper.gnn_inference import load_gnn
from paper.gnn_branch3_inference import load_branch3

DATA = ROOT / "data" / "v12_uncurated"
M = ROOT / "paper" / "models" / "v12_gentle_seed123"
N_SUB = 400


def _bucket(name): return int(hashlib.md5(f"cmp:{name}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _metrics(gt_pool, pred_pool, per_cell):
    m = _compute_metrics(gt_pool, pred_pool, {1, 2, 3, 4}); pl = m.get("per_label", {})
    a = np.array(per_cell)
    return {
        "node_accuracy": round(m["accuracy"], 4),
        "neurite_macro_f1": round(m["neurite_macro_f1"], 4),
        "axon_f1": round(pl.get("axon", {}).get("f1", 0.0), 4),
        "basal_f1": round(pl.get("basal/dendrite", {}).get("f1", 0.0), 4),
        "apical_f1": round(pl.get("apical", {}).get("f1", 0.0), 4),
        "per_cell_f1_mean": round(float(a.mean()), 4),
        "per_cell_f1_p10": round(float(np.quantile(a, 0.10)), 4),
    }


def main() -> int:
    sp = json.loads((M / "train_test_split.json").read_text(encoding="utf-8"))
    test = []
    for ct in ("pyramidal", "interneuron"):
        for fn in sp["test"].get(ct, []):
            p = DATA / ct / "swc" / fn
            if p.is_file():
                test.append((ct, p))
    test.sort(key=lambda t: _bucket(t[1].name))
    sub = test[:N_SUB]
    print(f"subsample: {len(sub)} of {len(test)} seed-123 test cells")

    gnn = load_gnn(M / "gnn_apical_basal.pt")
    b3 = load_branch3(M / "gnn_branch3_rescue.pt", gate_path=None)
    s1, s2 = M / "cell_type_classifier.pkl", M / "branch_classifier.pkl"

    # Pre-parse once.
    parsed = []
    for ct, p in sub:
        nodes = parse_swc(p)
        if nodes:
            parsed.append((ct, [n.type for n in nodes], nodes))

    results = {}
    for mode, use_sub in (("subtree", True), ("branch", False)):
        gt_pool, pred_pool, per_cell = [], [], []
        t0 = time.perf_counter()
        for ct, gt, nodes in parsed:
            pr = run_pipeline_on_nodes(
                nodes, file_path="", stage1_model=s1, stage2_model=s2,
                gnn_state=gnn, branch3_state=b3,
                use_subtree_stage2=use_sub, override_cell_type=ct,
            )
            pred = list(pr.node_labels)
            per_cell.append(per_cell_neurite_f1(gt, pred, ct))
            gt_pool.extend(gt); pred_pool.extend(pred)
        results[mode] = _metrics(gt_pool, pred_pool, per_cell)
        results[mode]["minutes"] = round((time.perf_counter() - t0) / 60.0, 1)
        print(f"  {mode:8s} done in {results[mode]['minutes']} min: "
              f"macroF1={results[mode]['neurite_macro_f1']} "
              f"apical={results[mode]['apical_f1']} P10={results[mode]['per_cell_f1_p10']}")

    (ROOT / "paper" / "results" / "stage2_mode_compare_seed123.json").write_text(
        json.dumps({"n_subsample": len(parsed), **results}, indent=2), encoding="utf-8")

    keys = ["node_accuracy", "neurite_macro_f1", "axon_f1", "basal_f1",
            "apical_f1", "per_cell_f1_mean", "per_cell_f1_p10"]
    print(f"\n{'metric':20s} {'subtree':>10} {'branch(64)':>12} {'delta':>9}")
    for k in keys:
        s, b = results["subtree"][k], results["branch"][k]
        print(f"{k:20s} {s:>10.4f} {b:>12.4f} {b-s:>+9.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
