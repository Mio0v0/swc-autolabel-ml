#!/usr/bin/env python3
"""Pyramidal-only Stage 2 + GNN retrain on a 50/50 split.

Experiment design:
    * Stage 1 (cell-type classifier)        -- INHERIT from v12_gentle_seed42 (frozen)
    * Stage 2 INTERNEURON sub-model         -- INHERIT from v12_gentle_seed42 (frozen)
    * Stage 2 PYRAMIDAL  sub-model          -- RETRAIN on the 50% pyramidal train
    * Stage 3 GNN (apical-vs-basal)         -- RETRAIN on the 50% pyramidal train

Important: v12's Stage 2 pyramidal sub-model is ALREADY trained on
pyramidals alone (interneurons live in a separate, independent
sub-model). So "freezing interneuron" is functionally a no-op for the
pyramidal weights. The real change here is the SPLIT: 50/50 instead
of 80/20, giving a much larger test set (~4500 pyramidals vs ~1750)
for cleaner statistical comparison at the cost of ~38% less training
data.

Output:
    paper/models/v12_pyramidal_only_seed<seed>/
        cell_type_classifier.pkl         (copied from source)
        branch_classifier.pkl            (merged: pyramidal=new, interneuron=source)
        gnn_apical_basal.pt              (new)
        qc_gate.pkl                      (copied from source)
        train_test_split.json
        eval_split_for_gnn.json
        training_metrics.json
    paper/results/pyramidal_only_eval_seed<seed>.csv      per-cell F1 on the 50% test
    paper/results/pyramidal_only_eval_seed<seed>.json     summary

Usage:
    python -m paper._train_pyramidal_only --seed 2024
        [--test-size 0.5]
        [--source-seed 42]          # which existing seed to inherit Stage 1 + interneuron from
        [--skip-eval]               # train only, don't evaluate

Wall time: ~90-130 min (Stage 2 ~30 min + GNN ~50 min + eval ~30 min).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Set focal-gamma default like the regular retrain
os.environ.setdefault("SWCAL_GNN_FOCAL_GAMMA", "2.0")
os.environ.setdefault("SWCAL_CLASS_BALANCE_POWER", "1.25")
os.environ.setdefault("SWCAL_GNN_CLASS_WEIGHT", "inverse_sqrt")

from hybrid.evaluate import _train_stage2, _file_in_test_bucket          # noqa: E402
from hybrid.features import parse_swc                                    # noqa: E402
from hybrid.pipeline import run_pipeline_on_nodes                        # noqa: E402
from hybrid.evaluate import per_cell_neurite_f1                          # noqa: E402
from paper.gnn_inference import load_gnn                                 # noqa: E402

QC_CSV   = ROOT / "paper" / "results" / "corpus_qc_v12_uncurated.csv"
DATA_DIR = ROOT / "data" / "v12_uncurated"
PYTHON   = Path("D:/Desktop/SWC-Studio/.venv/Scripts/python.exe")


def _load_pyramidals_from_qc() -> list[Path]:
    """Return all QC-passed pyramidal file paths."""
    out: list[Path] = []
    with QC_CSV.open("r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("qc_pass", "").strip().lower() not in ("true", "1"):
                continue
            if r["cell_type"] != "pyramidal":
                continue
            p = Path(r["path"])
            if p.is_file():
                out.append(p)
    return sorted(out)


def _all_pyramidals_in_data_dir() -> list[Path]:
    """ALL pyramidal .swc files on disk (including the 64 dropped by T2).

    Used to know which files the GNN's build_dataset will pick up so we
    can list dropped cells in test_files (preventing them from contaminating training).
    """
    return sorted((DATA_DIR / "pyramidal" / "swc").glob("*.swc"))


def _split_pyramidals(files: list[Path], seed: int, test_size: float):
    train, test = [], []
    for p in sorted(files):
        (test if _file_in_test_bucket(p.name, seed, test_size) else train).append(p)
    return train, test


def _merge_stage2_dicts(source_pkl: Path, new_pkl: Path, out_pkl: Path) -> dict:
    """Take pyramidal entries from `new_pkl`, interneuron from `source_pkl`,
    save merged dict to `out_pkl`. Returns a summary."""
    with source_pkl.open("rb") as fh: source = pickle.load(fh)
    with new_pkl.open("rb")    as fh: new    = pickle.load(fh)
    merged = dict(source)  # start from source (gives us interneuron weights + metadata)

    src_models = dict(source.get("models_by_cell_type", {}))
    new_models = dict(new.get("models_by_cell_type", {}))
    src_models["pyramidal"] = new_models.get("pyramidal", src_models.get("pyramidal"))
    merged["models_by_cell_type"] = src_models

    src_defaults = dict(source.get("default_labels_by_cell_type", {}))
    new_defaults = dict(new.get("default_labels_by_cell_type", {}))
    if "pyramidal" in new_defaults:
        src_defaults["pyramidal"] = new_defaults["pyramidal"]
    elif "pyramidal" in src_defaults and "pyramidal" in src_models:
        src_defaults.pop("pyramidal", None)
    merged["default_labels_by_cell_type"] = src_defaults

    src_owners = dict(source.get("subtree_owner_models_by_cell_type", {}))
    new_owners = dict(new.get("subtree_owner_models_by_cell_type", {}))
    if "pyramidal" in new_owners:
        src_owners["pyramidal"] = new_owners["pyramidal"]
    merged["subtree_owner_models_by_cell_type"] = src_owners
    merged["pyramidal_subtree_owner_model"] = src_owners.get("pyramidal")

    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as fh: pickle.dump(merged, fh)

    return {
        "merged_keys": sorted(src_models.keys()),
        "source_interneuron_present": "interneuron" in src_models,
        "new_pyramidal_present":      "pyramidal" in src_models,
    }


def _eval_held_out(
    test_files: list[Path],
    s1_path: Path, s2_path: Path, gnn_path: Path,
    out_csv: Path, out_json: Path,
):
    print(f"\n=== Evaluating on {len(test_files)} held-out pyramidals ===")
    gnn_state = load_gnn(gnn_path)
    rows: list[dict] = []
    t0 = time.perf_counter()
    n_failed = 0
    for i, p in enumerate(test_files):
        try:
            nodes = parse_swc(p)
            if not nodes:
                n_failed += 1; continue
            gt = [n.type for n in nodes]
        except Exception as exc:
            print(f"  WARN parse {p.name}: {exc}")
            n_failed += 1; continue
        try:
            pr = run_pipeline_on_nodes(
                nodes, file_path="",
                stage1_model=s1_path, stage2_model=s2_path,
                gnn_state=gnn_state, use_subtree_stage2=True,
            )
        except Exception as exc:
            print(f"  WARN pipeline {p.name}: {exc}")
            n_failed += 1; continue
        pred = list(pr.node_labels)

        pc_f1 = per_cell_neurite_f1(gt, pred, "pyramidal")
        n_correct = sum(1 for g, q in zip(gt, pred) if g == q)
        pc_acc = n_correct / max(1, len(gt))

        y_t = np.asarray(gt, dtype=int); y_p = np.asarray(pred, dtype=int)
        def _f1(cls: int) -> float:
            if not (y_t == cls).any(): return float("nan")
            return float(f1_score((y_t == cls).astype(int), (y_p == cls).astype(int), zero_division=0))
        axon_f1   = _f1(2)
        basal_f1  = _f1(3)
        apical_f1 = _f1(4)
        pred_counter = Counter(pred)

        rows.append({
            "file":          p.name,
            "n_nodes":       len(gt),
            "held_out_F1":   f"{pc_f1:.4f}",
            "acc":           f"{pc_acc:.4f}",
            "axon_F1":       f"{axon_f1:.4f}"   if not np.isnan(axon_f1)   else "",
            "basal_F1":      f"{basal_f1:.4f}"  if not np.isnan(basal_f1)  else "",
            "apical_F1":     f"{apical_f1:.4f}" if not np.isnan(apical_f1) else "",
            "stage1_pred":   pr.stage1.cell_type,
            "stage1_conf":   f"{pr.stage1.confidence:.4f}",
            "pred_axon":     pred_counter.get(2, 0),
            "pred_basal":    pred_counter.get(3, 0),
            "pred_apical":   pred_counter.get(4, 0),
        })
        if (i + 1) % 200 == 0:
            elapsed = (time.perf_counter() - t0) / 60.0
            eta = elapsed * (len(test_files) - i - 1) / max(1, i + 1)
            print(f"  ... {i+1}/{len(test_files)}  ({elapsed:.1f} min, ETA {eta:.0f} min)", flush=True)

    elapsed = (time.perf_counter() - t0) / 60.0
    print(f"  eval done: {len(rows)} scored, {n_failed} failed  ({elapsed:.1f} min)")

    rows.sort(key=lambda r: float(r["held_out_F1"]))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow(r)

    f1s = np.array([float(r["held_out_F1"]) for r in rows])
    summary = {
        "n_test":     len(rows),
        "n_failed":   n_failed,
        "eval_min":   elapsed,
        "F1_mean":    float(f1s.mean()),
        "F1_median":  float(np.median(f1s)),
        "F1_p10":     float(np.percentile(f1s, 10)),
        "F1_p25":     float(np.percentile(f1s, 25)),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  F1 mean={summary['F1_mean']:.4f}  median={summary['F1_median']:.4f}  "
          f"P10={summary['F1_p10']:.4f}  P25={summary['F1_p25']:.4f}")
    print(f"  CSV  -> {out_csv}")
    print(f"  JSON -> {out_json}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed",        type=int,   default=2024)
    ap.add_argument("--test-size",   type=float, default=0.5)
    ap.add_argument("--source-seed", type=int,   default=42,
                    help="Seed of existing v12 model to inherit Stage 1 + interneuron from.")
    ap.add_argument("--skip-eval",   action="store_true",
                    help="Train only, skip held-out eval at the end.")
    ap.add_argument("--suffix",      type=str,   default="",
                    help="Suffix appended to model dir + result-file names "
                         "(e.g. '_lmeasure'). Lets multiple variants coexist.")
    args = ap.parse_args()

    src_dir = ROOT / "paper" / "models" / f"v12_gentle_seed{args.source_seed}"
    if not (src_dir / "branch_classifier.pkl").is_file():
        raise SystemExit(f"Source model missing: {src_dir}")

    out_dir = ROOT / "paper" / "models" / f"v12_pyramidal_only_seed{args.seed}{args.suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    overall_t0 = time.perf_counter()

    print(f"=== Pyramidal-only retrain  seed={args.seed}  test_size={args.test_size} ===")
    print(f"  source model dir: {src_dir.name}")
    print(f"  output model dir: {out_dir.name}")
    print(f"  SWCAL_GNN_FOCAL_GAMMA      = {os.environ.get('SWCAL_GNN_FOCAL_GAMMA')}")
    print(f"  SWCAL_CLASS_BALANCE_POWER  = {os.environ.get('SWCAL_CLASS_BALANCE_POWER')}")
    print(f"  SWCAL_GNN_CLASS_WEIGHT     = {os.environ.get('SWCAL_GNN_CLASS_WEIGHT')}")

    # 1) Build the 50/50 pyramidal split (QC-passed only).
    qc_pyrs = _load_pyramidals_from_qc()
    print(f"\n[Split] QC-passed pyramidals: {len(qc_pyrs)}")
    train_pyrs, test_pyrs = _split_pyramidals(qc_pyrs, args.seed, args.test_size)
    print(f"  train={len(train_pyrs)}  test={len(test_pyrs)}")

    # 2) Save the split JSON files.
    (out_dir / "train_test_split.json").write_text(json.dumps({
        "seed": args.seed,
        "test_size": args.test_size,
        "convention": "pyramidal-only 50/50; interneuron Stage 2 inherited from source",
        "source_seed": args.source_seed,
        "train": {"pyramidal": sorted(p.name for p in train_pyrs), "interneuron": []},
        "test":  {"pyramidal": sorted(p.name for p in test_pyrs),  "interneuron": []},
        "counts": {"train_pyramidal": len(train_pyrs), "test_pyramidal": len(test_pyrs)},
    }, indent=2), encoding="utf-8")

    # 3) GNN eval-split must list:
    #    test_files.pyramidal = test half + ALL dropped pyramidals on disk
    #    so the GNN doesn't accidentally pick up the 64 dropped cells in training.
    train_basenames    = {p.name for p in train_pyrs}
    disk_pyrs          = _all_pyramidals_in_data_dir()
    test_or_excluded   = sorted({p.name for p in disk_pyrs if p.name not in train_basenames})
    print(f"  GNN test_files.pyramidal = {len(test_or_excluded)} files  "
          f"(test half + {len(disk_pyrs) - len(train_basenames) - len(test_pyrs)} dropped)")
    (out_dir / "eval_split_for_gnn.json").write_text(json.dumps({
        "seed":       args.seed,
        "test_size":  args.test_size,
        "test_files": {"pyramidal": test_or_excluded, "interneuron": []},
    }, indent=2), encoding="utf-8")

    # 4) Copy frozen artifacts from source.
    for fname in ("cell_type_classifier.pkl", "qc_gate.pkl"):
        src = src_dir / fname
        if src.is_file():
            shutil.copy2(src, out_dir / fname)
            print(f"  copied frozen artifact: {fname}")

    # 5) Train Stage 2 -- ONLY pyramidal training data.
    s2_tmp = out_dir / "branch_classifier.pyramidal_only.pkl"
    train_files_dict = {"pyramidal": train_pyrs, "interneuron": []}
    print(f"\n[Stage 2] training pyramidal sub-model on {len(train_pyrs)} files...")
    t0 = time.perf_counter()
    _train_stage2(train_files_dict, s2_tmp)
    s2_min = (time.perf_counter() - t0) / 60.0
    print(f"  Stage 2 elapsed: {s2_min:.1f} min")

    # 6) Merge: keep source interneuron entries, use new pyramidal entries.
    merged_pkl = out_dir / "branch_classifier.pkl"
    summary_merge = _merge_stage2_dicts(src_dir / "branch_classifier.pkl", s2_tmp, merged_pkl)
    print(f"  merged Stage 2 -> {merged_pkl}  ({summary_merge})")
    s2_tmp.unlink(missing_ok=True)

    # 7) Train GNN.
    gnn_out = out_dir / "gnn_apical_basal.pt"
    gnn_type = os.environ.get("SWCAL_GNN_ARCH", "sage").lower()
    print(f"\n[GNN] training (seed={args.seed}, gnn-type={gnn_type})...")
    t0 = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    cmd = [
        str(PYTHON), "-u", "-m", "paper.gnn_apical_basal",
        "--data-dir",   str(DATA_DIR),
        "--eval-split", str(out_dir / "eval_split_for_gnn.json"),
        "--ckpt",       str(gnn_out),
        "--seed",       str(args.seed),
        "--gnn-type",   gnn_type,
    ]
    print(f"  CMD: {' '.join(cmd)}")
    rc = subprocess.call(cmd, env=env, cwd=str(ROOT))
    gnn_min = (time.perf_counter() - t0) / 60.0
    print(f"  GNN elapsed: {gnn_min:.1f} min (rc={rc})")
    if rc != 0:
        print(f"  GNN training failed; aborting.")
        return rc

    # 8) Write training metrics.
    (out_dir / "training_metrics.json").write_text(json.dumps({
        "seed":               args.seed,
        "source_seed":        args.source_seed,
        "test_size":          args.test_size,
        "n_train_pyramidal":  len(train_pyrs),
        "n_test_pyramidal":   len(test_pyrs),
        "stage2_min":         s2_min,
        "gnn_min":            gnn_min,
        "total_min":          (time.perf_counter() - overall_t0) / 60.0,
        "swcal_class_balance_power": os.environ.get("SWCAL_CLASS_BALANCE_POWER"),
        "swcal_gnn_class_weight":    os.environ.get("SWCAL_GNN_CLASS_WEIGHT"),
        "swcal_gnn_focal_gamma":     os.environ.get("SWCAL_GNN_FOCAL_GAMMA"),
        "merge_summary": summary_merge,
    }, indent=2), encoding="utf-8")

    # 9) Held-out eval on test half (unless skipped).
    if args.skip_eval:
        print("\n--skip-eval set; not running held-out eval.")
    else:
        out_csv  = ROOT / "paper" / "results" / f"pyramidal_only_eval_seed{args.seed}{args.suffix}.csv"
        out_json = ROOT / "paper" / "results" / f"pyramidal_only_eval_seed{args.seed}{args.suffix}.json"
        _eval_held_out(
            test_files=test_pyrs,
            s1_path=out_dir / "cell_type_classifier.pkl",
            s2_path=merged_pkl,
            gnn_path=gnn_out,
            out_csv=out_csv,
            out_json=out_json,
        )

    total_min = (time.perf_counter() - overall_t0) / 60.0
    print(f"\nTOTAL ELAPSED: {total_min:.1f} min")
    print(f"Model -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
