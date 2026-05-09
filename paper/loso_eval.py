#!/usr/bin/env python3
"""Leave-one-source-out cross-corpus evaluation.

For each of the 3 real sources in v10_dedup_dataset (in_house = hpf_ca1,
neuromorpho, allen), train Stage 1 + Stage 2 + GNN on the OTHER two
sources, then evaluate on the held-out source. This is the gold-standard
generalization test: "does the pipeline work on a corpus it never saw?"

Source classification is by SWC filename prefix:
  hpf_ca1__*.swc       -&gt; in_house
  neuromorpho__*.swc   -&gt; neuromorpho
  allen__*.swc         -&gt; allen
  anything else        -&gt; other (treated as in_house if it ends up needed)

Outputs (per held-out source):
  paper/results/snapshots/eval_loso_<src>.json
  paper/results/snapshots/eval_loso_<src>_per_file.csv
  paper/models/loso_<src>/{s1.pkl, s2.pkl, gnn.pt}

Usage::

    # all 3 LOSO experiments
    python -m paper.loso_eval

    # one specific experiment
    python -m paper.loso_eval --hold-out neuromorpho
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = ROOT / "paper" / "results" / "snapshots"
LOSO_MODEL_DIR = ROOT / "paper" / "models" / "loso"
LOG_PATH = ROOT / "paper" / "results" / "loso_eval.log"
DEFAULT_DATA_DIR = Path("D:/Desktop/SWC-Studio/data/v10_dedup_dataset")
DEFAULT_PYTHON = Path("D:/Desktop/SWC-Studio/.venv/Scripts/python.exe")

SOURCES = ("in_house", "neuromorpho", "allen")


def _classify_source(filename: str) -> str:
    n = filename.lower()
    if n.startswith("hpf_ca1__") or n.startswith("lab__"):
        return "in_house"
    if n.startswith("neuromorpho__"):
        return "neuromorpho"
    if n.startswith("allen__"):
        return "allen"
    return "other"


def _split_by_source(
    data_dir: Path,
    held_out: str,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """Build (train_files, test_files), keyed by cell_type.
    train = cells from sources != held_out.  test = cells from held_out source.
    """
    from hybrid.evaluate import VALID_LABELS  # noqa
    train: dict[str, list[Path]] = {}
    test: dict[str, list[Path]] = {}
    for ct_dir in sorted(data_dir.iterdir()):
        if not ct_dir.is_dir():
            continue
        ct = ct_dir.name.lower()
        if ct not in VALID_LABELS:
            continue
        swc_dir = ct_dir / "swc" if (ct_dir / "swc").is_dir() else ct_dir
        train.setdefault(ct, [])
        test.setdefault(ct, [])
        for f in sorted(swc_dir.glob("*.swc")):
            src = _classify_source(f.name)
            if src == held_out:
                test[ct].append(f)
            else:
                train[ct].append(f)
    return train, test


def _write_eval_split(
    test_files: dict[str, list[Path]],
    out_path: Path,
    *,
    held_out: str,
    data_dir: Path,
) -> None:
    """Write a custom eval_split.json that paper.gnn_apical_basal
    will use to know which cells are 'test' (= held-out source).
    Everything not listed is training data."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({
            "loso_held_out": held_out,
            "data_dir": str(data_dir),
            "test_files": {ct: sorted(p.name for p in files) for ct, files in test_files.items()},
        }, f, indent=2)


def _run_subprocess(cmd: list[str], log_fh, env: dict[str, str] | None = None) -> int:
    """Tee a subprocess to console + log."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    proc = subprocess.Popen(
        cmd,
        env=full_env,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        log_fh.write(line)
        log_fh.flush()
    proc.wait()
    return proc.returncode


def loso_one(held_out: str, data_dir: Path, python_exe: Path, log_fh) -> bool:
    """Run one LOSO experiment. Returns True on success."""
    from hybrid.evaluate import _train_stage1, _train_stage2  # noqa

    banner = (
        f"\n{'=' * 78}\n"
        f"LOSO held-out source: {held_out}\n"
        f"  data_dir: {data_dir}\n"
        f"  started:  {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'=' * 78}\n"
    )
    print(banner, end="", flush=True)
    log_fh.write(banner)
    log_fh.flush()

    t0 = time.perf_counter()

    # --- 1. Split files by source ---
    train_files, test_files = _split_by_source(data_dir, held_out)
    n_train = sum(len(v) for v in train_files.values())
    n_test = sum(len(v) for v in test_files.values())
    msg = f"\n  train: {n_train} cells (sources != {held_out})\n  test:  {n_test} cells (source = {held_out})\n"
    for ct in sorted(train_files):
        msg += f"    {ct}:  train={len(train_files[ct])}  test={len(test_files[ct])}\n"
    print(msg, end="", flush=True)
    log_fh.write(msg)
    log_fh.flush()

    if n_test == 0:
        msg = f"  ERROR: no test files for held-out source '{held_out}'. Aborting.\n"
        print(msg, end="")
        log_fh.write(msg)
        return False

    # --- 2. Set up output directories ---
    out_dir = LOSO_MODEL_DIR / held_out
    out_dir.mkdir(parents=True, exist_ok=True)
    s1_path = out_dir / "s1.pkl"
    s2_path = out_dir / "s2.pkl"
    gnn_path = out_dir / "gnn.pt"
    eval_split_path = out_dir / "eval_split.json"

    # --- 3. Train Stage 1+2 on non-held-out cells ---
    if s1_path.is_file():
        msg = f"\n  reusing existing Stage 1 at {s1_path}\n"
    else:
        msg = f"\n  Training Stage 1 on {n_train} cells...\n"
    print(msg, end="", flush=True)
    log_fh.write(msg)
    log_fh.flush()
    if not s1_path.is_file():
        ts = time.time()
        _train_stage1(train_files, s1_path)
        msg = f"    done in {time.time() - ts:.1f}s\n"
        print(msg, end="", flush=True)
        log_fh.write(msg)
        log_fh.flush()

    if s2_path.is_file():
        msg = f"  reusing existing Stage 2 at {s2_path}\n"
    else:
        msg = f"  Training Stage 2 on {n_train} cells...\n"
    print(msg, end="", flush=True)
    log_fh.write(msg)
    log_fh.flush()
    if not s2_path.is_file():
        ts = time.time()
        _train_stage2(train_files, s2_path)
        msg = f"    done in {time.time() - ts:.1f}s\n"
        print(msg, end="", flush=True)
        log_fh.write(msg)
        log_fh.flush()

    # --- 4. Train GNN on non-held-out cells ---
    # Write custom eval_split.json that gnn_apical_basal will read; the
    # script trains on everything NOT in test_files["pyramidal"].
    _write_eval_split(test_files, eval_split_path, held_out=held_out, data_dir=data_dir)

    if gnn_path.is_file():
        msg = f"\n  reusing existing GNN at {gnn_path}\n"
        print(msg, end="", flush=True)
        log_fh.write(msg)
    else:
        msg = f"\n  Training GNN on non-held-out pyramidals...\n"
        print(msg, end="", flush=True)
        log_fh.write(msg)
        log_fh.flush()
        cmd = [
            str(python_exe), "-u", "-m", "paper.gnn_apical_basal",
            "--data-dir", str(data_dir),
            "--eval-split", str(eval_split_path),
            "--ckpt", str(gnn_path),
        ]
        rc = _run_subprocess(cmd, log_fh)
        if rc != 0:
            msg = f"  ERROR: GNN training failed (exit={rc})\n"
            print(msg, end="")
            log_fh.write(msg)
            return False

    # --- 5. Build a held-out-source test directory and run cross_dataset_eval ---
    # cross_dataset_eval needs all files of one cell type per call. Run it
    # twice (interneuron, pyramidal) and merge results in step 6.
    results_per_ct: dict[str, dict] = {}
    for ct, files in test_files.items():
        if not files:
            continue
        # Build a temp directory for this cell type's test files
        ct_dir = out_dir / f"test_{ct}"
        ct_dir.mkdir(exist_ok=True)
        # Symlink test files (avoid copying)
        for f in files:
            link = ct_dir / f.name
            if link.exists() or link.is_symlink():
                link.unlink()
            try:
                link.symlink_to(f)
            except (OSError, NotImplementedError):
                import shutil
                shutil.copy2(f, link)

        msg = f"\n  Running cross_dataset_eval on {len(files)} {ct} cells...\n"
        print(msg, end="", flush=True)
        log_fh.write(msg)
        log_fh.flush()

        out_tag = f"loso_{held_out}_{ct}"
        cmd = [
            str(python_exe), "-u", "-m", "paper.cross_dataset_eval",
            "--data-dir", str(ct_dir),
            "--cell-type", ct,
            "--tag", out_tag,
            "--stage1-model", str(s1_path),
            "--stage2-model", str(s2_path),
            "--gnn-model", str(gnn_path),
        ]
        rc = _run_subprocess(cmd, log_fh)
        if rc != 0:
            msg = f"  ERROR: cross_dataset_eval failed for {ct} (exit={rc})\n"
            print(msg, end="")
            log_fh.write(msg)
            return False

        # Read the result JSON
        result_path = ROOT / "paper" / "results" / f"cross_dataset_{out_tag}.json"
        if result_path.is_file():
            results_per_ct[ct] = json.loads(result_path.read_text(encoding="utf-8"))

    # --- 6. Merge per-cell-type results into a single LOSO snapshot ---
    snapshot = {
        "loso_held_out": held_out,
        "data_dir": str(data_dir),
        "n_train_cells": n_train,
        "n_test_cells": n_test,
        "stage1_model": str(s1_path),
        "stage2_model": str(s2_path),
        "gnn_model": str(gnn_path),
        "per_cell_type": results_per_ct,
    }
    snapshot_path = SNAPSHOT_DIR / f"eval_loso_{held_out}.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    msg = f"\n  snapshot: {snapshot_path}\n"
    print(msg, end="", flush=True)
    log_fh.write(msg)

    # Build a unified per-file CSV
    csv_path = SNAPSHOT_DIR / f"eval_loso_{held_out}_per_file.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cell_type", "path", "n_nodes", "neurite_macro_f1_stage23", "macro_f1_stage23"])
        for ct, payload in results_per_ct.items():
            for r in payload.get("per_file", []):
                writer.writerow([
                    ct,
                    r.get("path", ""),
                    r.get("n_nodes", 0),
                    r.get("neurite_macro_f1", 0.0),
                    r.get("macro_f1", 0.0),
                ])
    msg = f"  per-file CSV: {csv_path}\n"
    print(msg, end="", flush=True)
    log_fh.write(msg)

    elapsed = time.perf_counter() - t0
    msg = f"\n----- LOSO {held_out} finished — elapsed {elapsed/60:.1f} min -----\n"
    print(msg, end="", flush=True)
    log_fh.write(msg)
    log_fh.flush()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--hold-out", default="all", choices=("all",) + SOURCES)
    parser.add_argument("--log-path", type=Path, default=LOG_PATH)
    args = parser.parse_args()

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    sources = SOURCES if args.hold_out == "all" else (args.hold_out,)

    print(f"LOSO experiments: {', '.join(sources)}")
    print(f"Log: {args.log_path}")

    overall_t0 = time.perf_counter()
    summary: list[tuple[str, bool, float]] = []
    with args.log_path.open("a", encoding="utf-8") as log_fh:
        log_fh.write(f"\n\n=== loso_eval start: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        for src in sources:
            t0 = time.perf_counter()
            ok = loso_one(src, args.data_dir, args.python, log_fh)
            summary.append((src, ok, time.perf_counter() - t0))

    overall = time.perf_counter() - overall_t0
    print("\n" + "=" * 78)
    print(f"LOSO COMPLETE — total {overall/3600:.2f} hr")
    print("=" * 78)
    for src, ok, elapsed in summary:
        status = "OK " if ok else "FAIL"
        print(f"  [{status}]  loso_{src:<14s}  {elapsed/60:>6.1f} min")
    return 0 if all(ok for _, ok, _ in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
