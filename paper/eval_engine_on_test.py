#!/usr/bin/env python3
"""Run the auto-typing engine on the v9 test split and report metrics.

Lightweight inference-only evaluator. Reads the held-out test files
from ``paper/results/snapshots/v9_eval_split.json``, runs the engine
on each, and writes per-file + summary CSVs / JSONs. No retraining,
no caching — just inference using whatever models are currently
bundled / configured.

Used by the ablation rows that don't need a full retrain (e.g.
``no-soft-handoff``, controlled via the ``SWCAL_NO_SOFT_HANDOFF`` env
var read by ``swcstudio.core.auto_typing.pipeline``). Set the env var
*before* invoking::

    set SWCAL_NO_SOFT_HANDOFF=1
    python -m paper.eval_engine_on_test --tag no_soft_handoff

Outputs (under ``paper/results/snapshots/``):
    eval_<tag>.json
    eval_<tag>_per_file.csv

Usage::

    python -m paper.eval_engine_on_test --tag v9_inference_only
    python -m paper.eval_engine_on_test --tag no_soft_handoff
    python -m paper.eval_engine_on_test --tag no_pca --limit 100   # smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = ROOT / "paper" / "results" / "snapshots"
DEFAULT_DATA_DIR = Path("D:/Desktop/SWC-Studio/data/v9_merged_dataset")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")


def _per_class_f1(gt: list[int], pred: list[int]) -> dict:
    """Macro-F1, neurite-macro-F1, accuracy, per-class F1 from a list pair."""
    classes = (1, 2, 3, 4)
    per_label: dict[int, dict] = {}
    for lbl in classes:
        tp = sum(1 for g, p in zip(gt, pred) if g == lbl and p == lbl)
        fp = sum(1 for g, p in zip(gt, pred) if g != lbl and p == lbl)
        fn = sum(1 for g, p in zip(gt, pred) if g == lbl and p != lbl)
        sup = sum(1 for g in gt if g == lbl)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_label[lbl] = {"precision": prec, "recall": rec, "f1": f1, "support": sup}
    accuracy = sum(1 for g, p in zip(gt, pred) if g == p) / max(1, len(gt))
    f1s_present = [per_label[lbl]["f1"] for lbl in classes if per_label[lbl]["support"] > 0]
    macro = float(np.mean(f1s_present)) if f1s_present else 0.0
    neurite_f1s = [
        per_label[lbl]["f1"] for lbl in (2, 3, 4) if per_label[lbl]["support"] > 0
    ]
    neurite_macro = float(np.mean(neurite_f1s)) if neurite_f1s else 0.0
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro),
        "neurite_macro_f1": float(neurite_macro),
        "per_label": {
            {1: "soma", 2: "axon", 3: "basal", 4: "apical"}[lbl]: v
            for lbl, v in per_label.items()
        },
    }


def _summary(file_f1s: list[float]) -> dict:
    if not file_f1s:
        return {"n_files": 0}
    arr = np.array(file_f1s, dtype=np.float64)
    return {
        "n_files": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--eval-split", type=Path,
        default=SNAPSHOT_DIR / "v9_eval_split.json",
    )
    parser.add_argument("--tag", required=True, help="Run label, used in output filenames.")
    parser.add_argument("--limit", type=int, default=0, help="Cap files (0=all).")
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.eval_split.is_file():
        print(f"ERROR: split file not found: {args.eval_split}", file=sys.stderr)
        return 2
    if not args.data_dir.is_dir():
        print(f"ERROR: data dir not found: {args.data_dir}", file=sys.stderr)
        return 2

    # IMPORTANT: import only after env var is settled — pipeline.py reads
    # SWCAL_NO_SOFT_HANDOFF at module load.
    from swcstudio.core.auto_typing import (  # noqa: PLC0415
        BatchOptions,
        is_available,
        run_file,
    )

    print(f"Tag: {args.tag}")
    env_keys = ("SWCAL_NO_SOFT_HANDOFF", "SWCAL_NO_PCA", "SWCAL_NO_TRUNK")
    print("Env flags:", {k: os.environ.get(k, "") for k in env_keys})

    ok, reason = is_available(model_dir=str(args.model_dir) if args.model_dir else None)
    if not ok:
        print(f"ERROR: engine unavailable: {reason}", file=sys.stderr)
        return 2

    split = json.loads(args.eval_split.read_text(encoding="utf-8"))
    files: list[Path] = []
    for ct in ("pyramidal", "interneuron"):
        ct_dir = args.data_dir / ct / "swc"
        if not ct_dir.is_dir():
            ct_dir = args.data_dir / ct
        for name in split["test_files"].get(ct, []):
            p = ct_dir / name
            if p.is_file():
                files.append(p)
    if args.limit > 0:
        files = files[: args.limit]
    print(f"Test files: {len(files)}")

    opts = BatchOptions()
    rows: list[dict] = []
    file_f1s: list[float] = []
    all_gt: list[int] = []
    all_pred: list[int] = []
    timings: list[float] = []
    t_start_total = time.perf_counter()

    for idx, f in enumerate(files, 1):
        t0 = time.perf_counter()
        try:
            res = run_file(
                str(f),
                opts,
                write_output=False,
                write_log=False,
                model_dir=str(args.model_dir) if args.model_dir else None,
            )
            elapsed = time.perf_counter() - t0
            gt = [int(r["type"]) for r in res.rows]
            pred = list(res.types)
            metrics = _per_class_f1(gt, pred)
            file_f1s.append(metrics["neurite_macro_f1"])
            all_gt.extend(gt)
            all_pred.extend(pred)
            timings.append(elapsed)
            rows.append({
                "path": str(f),
                "cell_type": "pyramidal" if "pyramidal" in str(f) else "interneuron",
                "n_nodes": len(gt),
                "neurite_macro_f1": metrics["neurite_macro_f1"],
                "macro_f1": metrics["macro_f1"],
                "elapsed_sec": elapsed,
            })
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            rows.append({
                "path": str(f),
                "cell_type": "pyramidal" if "pyramidal" in str(f) else "interneuron",
                "n_nodes": 0,
                "neurite_macro_f1": 0.0,
                "macro_f1": 0.0,
                "elapsed_sec": elapsed,
                "error": str(exc),
            })
            print(f"  [{idx:>3d}/{len(files)}] {f.name}: ERROR {exc}")
            continue
        if idx % 50 == 0 or idx == len(files):
            print(
                f"  [{idx:>3d}/{len(files)}] mean F1 so far = "
                f"{float(np.mean(file_f1s)):.4f}, "
                f"avg time = {float(np.mean(timings)):.2f}s"
            )

    t_total = time.perf_counter() - t_start_total
    overall = _per_class_f1(all_gt, all_pred)
    file_summary = _summary(file_f1s)

    summary = {
        "tag": args.tag,
        "data_dir": str(args.data_dir),
        "n_files": len(files),
        "env": {k: os.environ.get(k, "") for k in env_keys},
        "overall": overall,
        "per_file": file_summary,
        "timing": {
            "total_sec": float(t_total),
            "median_per_file_sec": float(np.median(timings)) if timings else None,
            "p95_per_file_sec": float(np.percentile(timings, 95)) if timings else None,
            "max_per_file_sec": float(max(timings)) if timings else None,
        },
    }

    out_json = SNAPSHOT_DIR / f"eval_{args.tag}.json"
    out_csv = SNAPSHOT_DIR / f"eval_{args.tag}_per_file.csv"
    out_json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    if rows:
        cols = list(rows[0].keys())
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"\n{args.tag}: acc={overall['accuracy']:.4f}  "
          f"neurite_F1={overall['neurite_macro_f1']:.4f}  "
          f"per-file mean={file_summary.get('mean', 0):.4f}  "
          f"P10={file_summary.get('p10', 0):.4f}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
