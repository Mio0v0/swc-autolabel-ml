#!/usr/bin/env python3
"""§5.7 full-pipeline confirmation: does the curator-feedback labeling gain
survive when the WHOLE labeler (Stage 2 + GNN + Stage 3 topology) is retrained,
not just the Stage-2 classifier?

Compute-bounded endpoint design (a full multi-round curve would be ~days):
three models trained on a fixed seed set plus different additions, all
evaluated on the SAME held-out test set (GT cell-type config):

    seed     — Stage 2 + GNN trained on the seed set only
    +flag    — seed ∪ K cells selected by the seed model's flag/uncertainty signal
    +random  — seed ∪ K random cells

Each model runs the full pipeline at eval (Stage 2 → GNN → Stage 3 topology
refinement). Branch3 rescue is omitted for tractability (it is a fixed
correction head; noted in the paper). Stage 1 is bypassed via GT cell type,
matching §5.1/§5.7.

Usage:
    python -m paper._active_learning_full_pipeline --n-max 4000 --k 1000 --seed 123
    python -m paper._active_learning_full_pipeline --smoke   # tiny, fast

Outputs (incremental — safe to resume):
    paper/results/active_learning_full_pipeline.json
    paper/results/active_learning_full_pipeline.txt
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SWCAL_CLASS_BALANCE_POWER", "1.25")
os.environ.setdefault("SWCAL_GNN_CLASS_WEIGHT", "inverse_sqrt")
os.environ.setdefault("SWCAL_GNN_FOCAL_GAMMA", "2.0")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.features import parse_swc                               # noqa: E402
from hybrid.branch_features import extract_branches                 # noqa: E402
from hybrid.evaluate import _train_stage2, per_cell_neurite_f1      # noqa: E402
from hybrid.pipeline import run_pipeline_on_nodes                    # noqa: E402
from hybrid.cell_type_detector import CELL_TYPE_LABEL_SETS          # noqa: E402
from hybrid.comprehensive_metrics import build_full_report          # noqa: E402
from paper.gnn_inference import load_gnn                            # noqa: E402
from paper._active_learning_sim import _load_qc_cells, _hash_bucket  # noqa: E402

DATA_DIR = ROOT / "data" / "v12_uncurated"
MODELS = ROOT / "paper" / "models"
RESULTS = ROOT / "paper" / "results"
FACTORY_STAGE1 = MODELS / "v12_gentle_seed123" / "cell_type_classifier.pkl"
PYTHON = Path("D:/Desktop/SWC-Studio/.venv/Scripts/python.exe")


def _files_by_ct(cells) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {"pyramidal": [], "interneuron": []}
    for _src, ct, p in cells:
        out.setdefault(ct, []).append(p)
    return out


def _write_gnn_split(model_dir: Path, train_by_ct, test_by_ct):
    payload = {
        "seed": 0, "test_size": 0.2,
        "train_files": {ct: sorted(p.name for p in ps) for ct, ps in train_by_ct.items()},
        "test_files":  {ct: sorted(p.name for p in ps) for ct, ps in test_by_ct.items()},
    }
    (model_dir / "eval_split_for_gnn.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _train_endpoint(name, train_cells, test_by_ct, seed, gnn_epochs, gnn_quick) -> Path:
    import shutil
    mdir = MODELS / f"al_full_{name}_seed{seed}"
    mdir.mkdir(parents=True, exist_ok=True)
    train_by_ct = _files_by_ct(train_cells)
    n_tr = sum(len(v) for v in train_by_ct.values())
    print(f"    [{name}] Stage 2 on {n_tr} cells...")
    shutil.copy(FACTORY_STAGE1, mdir / "cell_type_classifier.pkl")
    _train_stage2(train_by_ct, mdir / "branch_classifier.pkl")
    _write_gnn_split(mdir, train_by_ct, test_by_ct)
    print(f"    [{name}] GNN...")
    cmd = [str(PYTHON), "-u", "-m", "paper.gnn_apical_basal",
           "--data-dir", str(DATA_DIR),
           "--eval-split", str(mdir / "eval_split_for_gnn.json"),
           "--ckpt", str(mdir / "gnn_apical_basal.pt"),
           "--seed", str(seed), "--epochs", str(gnn_epochs)]
    if gnn_quick:
        cmd.append("--quick")
    rc = subprocess.call(cmd, cwd=str(ROOT), env=os.environ.copy())
    if rc != 0:
        raise RuntimeError(f"GNN training failed for {name} (rc={rc})")
    return mdir


def _eval_endpoint(mdir: Path, test_cells) -> dict:
    gnn = load_gnn(mdir / "gnn_apical_basal.pt")
    s1 = mdir / "cell_type_classifier.pkl"
    s2 = mdir / "branch_classifier.pkl"
    per_cell, gt_pool, pred_pool, records = [], [], [], []
    for _src, ct, p in test_cells:
        nodes = parse_swc(p)
        if not nodes:
            continue
        gt = [n.type for n in nodes]
        pr = run_pipeline_on_nodes(
            nodes, file_path="", stage1_model=s1, stage2_model=s2,
            gnn_state=gnn, branch3_state=None, use_subtree_stage2=True,
            override_cell_type=ct,
        )
        pred = list(pr.node_labels)
        per_cell.append(per_cell_neurite_f1(gt, pred, ct))
        gt_pool.extend(gt); pred_pool.extend(pred)
        records.append({"cell_type": ct, "n_nodes": len(gt), "gt": gt, "pred": pred})
    rep = build_full_report("al_full", gt_pool, pred_pool, records, 0.0)
    arr = np.array(per_cell)
    pc = rep["per_class"] if "per_class" in rep else rep["corpus"]["per_class"]
    return {
        "node_accuracy": round(rep["corpus"]["accuracy"], 4),
        "neurite_macro_f1": round(rep["corpus"]["neurite_macro_f1"], 4),
        "axon_f1": round(pc.get("axon", {}).get("f1", 0.0), 4),
        "basal_f1": round(pc.get("basal/dendrite", {}).get("f1", 0.0), 4),
        "apical_f1": round(pc.get("apical", {}).get("f1", 0.0), 4),
        "per_cell_f1_mean": round(float(arr.mean()), 4),
        "per_cell_f1_p10": round(float(np.quantile(arr, 0.10)), 4),
        "n_test": len(per_cell),
    }


def _score_uncertainty(seed_mdir: Path, pool_cells) -> dict[str, float]:
    """Per-cell mean (1 - node confidence) from the seed model; higher = flagged.

    Runs the real Stage-2 path (via the pipeline, which builds the
    owner-augmented features internally) with GNN/Branch3 disabled for speed;
    the acquisition only needs Stage-2 confidence.
    """
    s1 = seed_mdir / "cell_type_classifier.pkl"
    s2 = seed_mdir / "branch_classifier.pkl"
    scores: dict[str, float] = {}
    for _src, ct, p in pool_cells:
        try:
            nodes = parse_swc(p)
            pr = run_pipeline_on_nodes(
                nodes, file_path="", stage1_model=s1, stage2_model=s2,
                gnn_state=None, branch3_state=None, use_subtree_stage2=True,
                override_cell_type=ct,
            )
            conf = pr.node_confidences
            scores[p.name] = (1.0 - float(np.mean(conf))) if len(conf) else 0.0
        except Exception:
            scores[p.name] = 0.0
    return scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-max", type=int, default=4000)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--seed-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--gnn-epochs", type=int, default=120)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_max, args.k, args.gnn_epochs = 250, 30, 12

    cells = _load_qc_cells(args.n_max)
    seed_cells, pool_cells, test_cells = [], [], []
    for c in cells:
        b = _hash_bucket(c[2].name, f"alfull_split_{args.seed}")
        if b < args.test_frac:
            test_cells.append(c)
        elif b < args.test_frac + args.seed_frac:
            seed_cells.append(c)
        else:
            pool_cells.append(c)
    test_by_ct = _files_by_ct(test_cells)
    print(f"=== full-pipeline AL (seed={args.seed}) ===")
    print(f"  seed={len(seed_cells)} pool={len(pool_cells)} test={len(test_cells)} k={args.k}")

    out_json = RESULTS / "active_learning_full_pipeline.json"
    results = {"seed": args.seed, "n_max": args.n_max, "k": args.k,
               "n_seed": len(seed_cells), "n_pool": len(pool_cells), "n_test": len(test_cells),
               "note": "Full pipeline (Stage2+GNN+Stage3 topology; Branch3 omitted). "
                       "GT cell-type config. Endpoints trained on seed / seed+flagK / seed+randomK.",
               "endpoints": {}}
    if out_json.is_file():
        try:
            results = json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            pass
    results.setdefault("endpoints", {})

    def _save():
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    t0 = time.perf_counter()
    # --- Endpoint: seed ---
    if "seed" not in results["endpoints"]:
        mdir = _train_endpoint("seed", seed_cells, test_by_ct, args.seed, args.gnn_epochs, args.smoke)
        results["endpoints"]["seed"] = _eval_endpoint(mdir, test_cells); _save()
        print(f"  seed: {results['endpoints']['seed']}")

    # --- Acquisition with seed model ---
    seed_mdir = MODELS / f"al_full_seed_seed{args.seed}"
    print("  scoring pool uncertainty with seed model...")
    scores = _score_uncertainty(seed_mdir, pool_cells)
    order = sorted(pool_cells, key=lambda c: scores.get(c[2].name, 0.0), reverse=True)
    flag_add = order[:args.k]
    rng = np.random.RandomState(args.seed)
    rand_idx = rng.permutation(len(pool_cells))[:args.k]
    rand_add = [pool_cells[i] for i in rand_idx]

    # --- Endpoint: +flag ---
    if "flag" not in results["endpoints"]:
        mdir = _train_endpoint("flag", seed_cells + flag_add, test_by_ct, args.seed, args.gnn_epochs, args.smoke)
        results["endpoints"]["flag"] = _eval_endpoint(mdir, test_cells); _save()
        print(f"  +flag: {results['endpoints']['flag']}")

    # --- Endpoint: +random ---
    if "random" not in results["endpoints"]:
        mdir = _train_endpoint("random", seed_cells + rand_add, test_by_ct, args.seed, args.gnn_epochs, args.smoke)
        results["endpoints"]["random"] = _eval_endpoint(mdir, test_cells); _save()
        print(f"  +random: {results['endpoints']['random']}")

    # --- Summary table ---
    e = results["endpoints"]
    lines = ["Full-pipeline active learning (Stage2+GNN+Stage3, GT cell type)",
             "=" * 66, "",
             f"seed={args.seed}  seed_train={len(seed_cells)} +K={args.k} corrections  test={len(test_cells)}",
             "",
             f"{'model':>10} {'macroF1':>9} {'apical':>8} {'basal':>8} {'axon':>8} {'pcF1mean':>9} {'pcF1p10':>8}"]
    for key in ("seed", "random", "flag"):
        if key in e:
            m = e[key]
            lines.append(f"{key:>10} {m['neurite_macro_f1']:>9.4f} {m['apical_f1']:>8.4f} "
                         f"{m['basal_f1']:>8.4f} {m['axon_f1']:>8.4f} "
                         f"{m['per_cell_f1_mean']:>9.4f} {m['per_cell_f1_p10']:>8.4f}")
    if "seed" in e and "flag" in e:
        d_seed = e["flag"]["neurite_macro_f1"] - e["seed"]["neurite_macro_f1"]
        lines += ["", f"flag vs seed:   macroF1 {d_seed:+.4f}   apical {e['flag']['apical_f1']-e['seed']['apical_f1']:+.4f}"]
    if "random" in e and "flag" in e:
        d_rand = e["flag"]["neurite_macro_f1"] - e["random"]["neurite_macro_f1"]
        lines += [f"flag vs random: macroF1 {d_rand:+.4f}   apical {e['flag']['apical_f1']-e['random']['apical_f1']:+.4f}"]
    (RESULTS / "active_learning_full_pipeline.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  total {(time.perf_counter()-t0)/60:.1f} min")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
