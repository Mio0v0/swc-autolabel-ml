#!/usr/bin/env python3
"""Overnight queue: full ablation grid + multi-seed + v9 cross-dataset.

This is the long-running paper-completion script. Launch it once and
walk away — total wall-clock ~12-20 hr depending on whether the GNN
retrain is on CPU or GPU.

Stages, run sequentially:

    1. no-PCA            full retrain (Stage 1 + Stage 2 + GNN with 7 PCA
                         features removed from cell-level + 5 from branch-level
                         feature vectors). ~3 hr CPU.
    2. no-trunk          full retrain with 4 trunk-detection features
                         removed from branch-level vector. ~3 hr CPU.
    3. multi-seed-123    full retrain at seed=123. ~3 hr CPU.
    4. multi-seed-456    full retrain at seed=456. ~3 hr CPU.
    5. v9-cross-dataset  zero-shot eval of the production v9 models on
                         the hpf_ca1 corpus (1377 cells). ~3 hr CPU
                         (~1 hr if the merged dataset has been
                         partially seen, but we eval on the cross-set).

Each stage writes its own snapshot under
``paper/results/snapshots/`` so a partial run still produces useful
output. Stages run as subprocesses with the right env vars set, so
the ablation hooks in hybrid/features.py and hybrid/branch_features.py
take effect.

Cheap ablations already run by other scripts (no-soft-handoff,
no-subtree-stage2, no-gnn) are NOT re-run here. Multi-seed seed=42
is already covered by v9_final.

Usage::

    python -m paper.overnight_queue            # full queue
    python -m paper.overnight_queue --skip no_pca,no_trunk  # multi-seed only
    python -m paper.overnight_queue --only multi_seed_123   # single stage

Logs land in ``paper/results/overnight_queue.log`` so the user can
tail it from another shell.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = ROOT / "paper" / "results" / "snapshots"
LOG_PATH = ROOT / "paper" / "results" / "overnight_queue.log"
DEFAULT_DATA_DIR = Path("D:/Desktop/SWC-Studio/data/v9_merged_dataset")


# Each entry: (stage_name, env_overrides, command_template).
# The command template uses {python} and {data_dir} placeholders.
STAGES: list[tuple[str, dict[str, str], list[str]]] = [
    # 1. no-PCA full retrain
    (
        "no_pca",
        {"SWCAL_NO_PCA": "1"},
        # Train Stage 1+2+GNN, then evaluate. Reuse the existing
        # `python -m hybrid.evaluate` which retrains as needed.
        [
            "{python}", "-u", "-m", "hybrid.evaluate",
            "--data-dir", "{data_dir}",
            "--use-gnn", "--use-subtree-stage2",
        ],
    ),
    # 2. no-trunk full retrain
    (
        "no_trunk",
        {"SWCAL_NO_TRUNK": "1"},
        [
            "{python}", "-u", "-m", "hybrid.evaluate",
            "--data-dir", "{data_dir}",
            "--use-gnn", "--use-subtree-stage2",
        ],
    ),
    # 3. multi-seed retrain — seed 123
    (
        "multi_seed_123",
        {},
        [
            "{python}", "-u", "-m", "hybrid.evaluate",
            "--data-dir", "{data_dir}",
            "--seed", "123",
            "--use-gnn", "--use-subtree-stage2",
        ],
    ),
    # 4. multi-seed retrain — seed 456
    (
        "multi_seed_456",
        {},
        [
            "{python}", "-u", "-m", "hybrid.evaluate",
            "--data-dir", "{data_dir}",
            "--seed", "456",
            "--use-gnn", "--use-subtree-stage2",
        ],
    ),
    # 5. v9 cross-dataset eval on hpf_ca1
    (
        "v9_cross_dataset_hpf_ca1",
        {},
        [
            "{python}", "-u", "-m", "paper.cross_dataset_eval",
            "--data-dir", str(Path("D:/Desktop/SWC-Studio/data/hpf_ca1")),
            "--cell-type", "pyramidal",
            "--tag", "v9_hpf_ca1",
        ],
    ),
]


def _resolve_command(template: list[str], python_exe: str, data_dir: Path) -> list[str]:
    out: list[str] = []
    for tok in template:
        out.append(
            tok
            .replace("{python}", python_exe)
            .replace("{data_dir}", str(data_dir))
        )
    return out


def _run_stage(
    name: str,
    env_overrides: dict[str, str],
    command: list[str],
    log_fh,
) -> tuple[bool, float]:
    """Run one stage as a subprocess. Returns (ok, elapsed_seconds)."""
    env = os.environ.copy()
    env.update(env_overrides)

    banner = (
        f"\n{'=' * 78}\n"
        f"STAGE: {name}\n"
        f"  env: {env_overrides}\n"
        f"  cmd: {' '.join(shlex.quote(c) for c in command)}\n"
        f"  started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'=' * 78}\n"
    )
    print(banner, end="", flush=True)
    log_fh.write(banner)
    log_fh.flush()

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        command,
        env=env,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        # Tee to console + log file.
        sys.stdout.write(line)
        sys.stdout.flush()
        log_fh.write(line)
        log_fh.flush()
    proc.wait()
    elapsed = time.perf_counter() - t0

    footer = (
        f"\n----- {name} finished: exit_code={proc.returncode} "
        f"elapsed={elapsed/60:.1f} min -----\n"
    )
    print(footer, end="", flush=True)
    log_fh.write(footer)
    log_fh.flush()
    return proc.returncode == 0, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--python", default=str(Path("D:/Desktop/SWC-Studio/.venv/Scripts/python.exe")),
        help="Python interpreter to use for each stage.",
    )
    parser.add_argument(
        "--only", default="",
        help="Comma-separated stage names to run (default: all). "
             "Available: " + ", ".join(s[0] for s in STAGES),
    )
    parser.add_argument(
        "--skip", default="",
        help="Comma-separated stage names to skip.",
    )
    parser.add_argument(
        "--log-path", type=Path, default=LOG_PATH,
        help="Combined log file (tee'd to console).",
    )
    args = parser.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    selected = [s for s in STAGES if (not only or s[0] in only) and s[0] not in skip]
    if not selected:
        print("ERROR: no stages selected", file=sys.stderr)
        return 2

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Selected stages ({len(selected)}): {', '.join(s[0] for s in selected)}")
    print(f"Log: {args.log_path}")

    summary: list[tuple[str, bool, float]] = []
    overall_t0 = time.perf_counter()
    with args.log_path.open("a", encoding="utf-8") as log_fh:
        log_fh.write(f"\n\n=== overnight_queue start: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        for name, env, cmd_template in selected:
            cmd = _resolve_command(cmd_template, args.python, args.data_dir)
            ok, elapsed = _run_stage(name, env, cmd, log_fh)
            summary.append((name, ok, elapsed))

    overall_elapsed = time.perf_counter() - overall_t0
    print("\n" + "=" * 78)
    print(f"OVERNIGHT QUEUE COMPLETE — total {overall_elapsed/3600:.2f} hr")
    print("=" * 78)
    for name, ok, elapsed in summary:
        status = "OK " if ok else "FAIL"
        print(f"  [{status}]  {name:<28s}  {elapsed/60:>6.1f} min")
    return 0 if all(ok for _, ok, _ in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
