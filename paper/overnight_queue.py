#!/usr/bin/env python3
"""Overnight queue: full ablation grid + multi-seed + v9 cross-dataset.

This is the long-running paper-completion script. Launch it once and
walk away — total wall-clock ~14-20 hr depending on whether the GNN
retrain is on CPU or GPU.

Stages, run sequentially:

    1. no-PCA            full retrain (Stage 1 + Stage 2 + GNN with 7 PCA
                         features removed from cell-level + 5 from branch-level
                         feature vectors). The GNN MUST be retrained because
                         the cached production GNN expects the full 61-dim
                         branch feature vector. ~4-5 hr CPU.
    2. no-trunk          full retrain with 4 trunk-detection features
                         removed from branch-level vector. Same GNN-retrain
                         requirement. ~4-5 hr CPU.
    3. multi-seed-123    Stage 1+2 retrain at seed=123 (GNN reused from
                         production — the claim is "Stage 1+2 seed
                         sensitivity at fixed GNN"). ~3 hr CPU.
    4. multi-seed-456    same at seed=456. ~3 hr CPU.
    5. v9-cross-dataset  zero-shot eval of the production v9 models on
                         the hpf_ca1 corpus. ~1 hr CPU.

Each stage writes its own snapshot under
``paper/results/snapshots/eval_<stage>.json`` so a partial run still
produces useful output. After every stage, the queue copies
``hybrid/models/evaluation_results.json`` and ``per_file_scores.csv``
into the snapshot dir so the next stage doesn't clobber them.

For the no-PCA / no-trunk ablations the queue first retrains a fresh
GNN under the env var (writing to a per-stage path) and then passes
``--gnn-model-path`` to ``hybrid.evaluate`` so the production GNN is
left untouched.

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
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = ROOT / "paper" / "results" / "snapshots"
GNN_MODEL_DIR = ROOT / "paper" / "models"
HYBRID_MODELS = ROOT / "hybrid" / "models"
LOG_PATH = ROOT / "paper" / "results" / "overnight_queue.log"
DEFAULT_DATA_DIR = Path("D:/Desktop/SWC-Studio/data/v9_merged_dataset")


def _gnn_retrain_cmd(stage_name: str, data_dir: Path) -> list[str]:
    """Retrain the apical-vs-basal GNN under whatever env the stage
    sets. The fresh checkpoint is saved to a stage-specific path so
    we never clobber the production GNN. The downstream evaluate
    step picks it up via ``--gnn-model-path``."""
    ckpt = GNN_MODEL_DIR / f"gnn_apical_basal_{stage_name}.pt"
    return [
        "{python}", "-u", "-m", "paper.gnn_apical_basal",
        "--data-dir", str(data_dir),
        "--ckpt", str(ckpt),
    ]


def _evaluate_cmd(
    data_dir: Path,
    *,
    seed: int | None = None,
    gnn_ckpt: Path | None = None,
) -> list[str]:
    cmd = [
        "{python}", "-u", "-m", "hybrid.evaluate",
        "--data-dir", str(data_dir),
        "--use-gnn", "--use-subtree-stage2",
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if gnn_ckpt is not None:
        cmd += ["--gnn-model-path", str(gnn_ckpt)]
    return cmd


# Each stage: (stage_name, env_overrides, command_templates, snapshot_results)
# command_templates is a list of subprocess argv lists. They run in order;
# any failure aborts the stage.
# snapshot_results means: after the last command, copy
#   hybrid/models/evaluation_results.json -> snapshots/eval_<name>.json
#   hybrid/models/per_file_scores.csv     -> snapshots/eval_<name>_per_file.csv
def _build_stages(data_dir: Path) -> list[tuple[str, dict[str, str], list[list[str]], bool]]:
    return [
        # 1. no-PCA: retrain GNN under env, then evaluate against the new GNN
        (
            "no_pca",
            {"SWCAL_NO_PCA": "1"},
            [
                _gnn_retrain_cmd("no_pca", data_dir),
                _evaluate_cmd(
                    data_dir,
                    gnn_ckpt=GNN_MODEL_DIR / "gnn_apical_basal_no_pca.pt",
                ),
            ],
            True,
        ),
        # 2. no-trunk: same shape as no-PCA
        (
            "no_trunk",
            {"SWCAL_NO_TRUNK": "1"},
            [
                _gnn_retrain_cmd("no_trunk", data_dir),
                _evaluate_cmd(
                    data_dir,
                    gnn_ckpt=GNN_MODEL_DIR / "gnn_apical_basal_no_trunk.pt",
                ),
            ],
            True,
        ),
        # 3-4. multi-seed: Stage 1+2 retrain only; GNN reused from production.
        (
            "multi_seed_123",
            {},
            [_evaluate_cmd(data_dir, seed=123)],
            True,
        ),
        (
            "multi_seed_456",
            {},
            [_evaluate_cmd(data_dir, seed=456)],
            True,
        ),
        # 5. v9 cross-dataset eval — separate script, writes its own snapshot
        (
            "v9_cross_dataset_hpf_ca1",
            {},
            [
                [
                    "{python}", "-u", "-m", "paper.cross_dataset_eval",
                    "--data-dir", str(Path("D:/Desktop/SWC-Studio/data/hpf_ca1")),
                    "--cell-type", "pyramidal",
                    "--tag", "v9_hpf_ca1",
                ],
            ],
            False,
        ),
    ]


def _resolve_command(template: list[str], python_exe: str) -> list[str]:
    return [tok.replace("{python}", python_exe) for tok in template]


def _snapshot_results(stage_name: str, log_fh) -> None:
    """Copy hybrid.evaluate's outputs into the snapshot dir under a
    stage-specific name so the next stage doesn't overwrite them."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = [
        (HYBRID_MODELS / "evaluation_results.json",
         SNAPSHOT_DIR / f"eval_{stage_name}.json"),
        (HYBRID_MODELS / "per_file_scores.csv",
         SNAPSHOT_DIR / f"eval_{stage_name}_per_file.csv"),
    ]
    for src, dst in pairs:
        if not src.is_file():
            msg = f"  WARN: snapshot source missing: {src}\n"
            print(msg, end="", flush=True)
            log_fh.write(msg)
            continue
        shutil.copy2(src, dst)
        msg = f"  snapshot: {src.name} -> {dst}\n"
        print(msg, end="", flush=True)
        log_fh.write(msg)
    log_fh.flush()


def _run_subprocess(command: list[str], env: dict[str, str], log_fh) -> int:
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
        sys.stdout.write(line)
        sys.stdout.flush()
        log_fh.write(line)
        log_fh.flush()
    proc.wait()
    return proc.returncode


def _run_stage(
    name: str,
    env_overrides: dict[str, str],
    commands: list[list[str]],
    snapshot_results: bool,
    log_fh,
) -> tuple[bool, float]:
    """Run all subcommands of a stage in order. Stop on first non-zero exit.
    Returns (ok, elapsed_seconds)."""
    env = os.environ.copy()
    env.update(env_overrides)

    banner = (
        f"\n{'=' * 78}\n"
        f"STAGE: {name}\n"
        f"  env: {env_overrides}\n"
        f"  steps: {len(commands)}\n"
        f"  started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'=' * 78}\n"
    )
    print(banner, end="", flush=True)
    log_fh.write(banner)
    log_fh.flush()

    # Track result-file mtimes BEFORE the stage so we can detect "the
    # stage produced fresh results, then crashed on a print" — the
    # metrics are still valid and worth snapshotting.
    eval_json = HYBRID_MODELS / "evaluation_results.json"
    eval_csv = HYBRID_MODELS / "per_file_scores.csv"
    pre_mtimes: dict[Path, float] = {
        p: (p.stat().st_mtime if p.is_file() else 0.0)
        for p in (eval_json, eval_csv)
    }

    t0 = time.perf_counter()
    ok = True
    for i, cmd in enumerate(commands, start=1):
        step_banner = (
            f"\n--- step {i}/{len(commands)}: "
            f"{' '.join(shlex.quote(c) for c in cmd)} ---\n"
        )
        print(step_banner, end="", flush=True)
        log_fh.write(step_banner)
        log_fh.flush()

        rc = _run_subprocess(cmd, env, log_fh)
        if rc != 0:
            err = f"\n  step {i} FAILED with exit_code={rc} — aborting stage\n"
            print(err, end="", flush=True)
            log_fh.write(err)
            ok = False
            break

    elapsed = time.perf_counter() - t0

    # Snapshot if requested AND fresh results landed during this stage,
    # regardless of exit code. Late-stage prints (Unicode chars on
    # cp1252 Windows stdout) can crash hybrid.evaluate AFTER it has
    # already written valid metrics to disk — we still want those.
    if snapshot_results:
        result_was_updated = any(
            p.is_file() and p.stat().st_mtime > pre_mtimes.get(p, 0.0)
            for p in (eval_json, eval_csv)
        )
        if result_was_updated:
            _snapshot_results(name, log_fh)
        else:
            msg = f"  no fresh results detected (mtimes unchanged); skipping snapshot\n"
            print(msg, end="", flush=True)
            log_fh.write(msg)

    footer = (
        f"\n----- {name} finished: ok={ok} "
        f"elapsed={elapsed/60:.1f} min -----\n"
    )
    print(footer, end="", flush=True)
    log_fh.write(footer)
    log_fh.flush()
    return ok, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--python", default=str(Path("D:/Desktop/SWC-Studio/.venv/Scripts/python.exe")),
        help="Python interpreter to use for each stage.",
    )
    stages_all = _build_stages(Path("X"))  # placeholder for help text only
    parser.add_argument(
        "--only", default="",
        help="Comma-separated stage names to run (default: all). "
             "Available: " + ", ".join(s[0] for s in stages_all),
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

    stages = _build_stages(args.data_dir)
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    selected = [s for s in stages if (not only or s[0] in only) and s[0] not in skip]
    if not selected:
        print("ERROR: no stages selected", file=sys.stderr)
        return 2

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    GNN_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Selected stages ({len(selected)}): {', '.join(s[0] for s in selected)}")
    print(f"Log: {args.log_path}")

    summary: list[tuple[str, bool, float]] = []
    overall_t0 = time.perf_counter()
    with args.log_path.open("a", encoding="utf-8") as log_fh:
        log_fh.write(f"\n\n=== overnight_queue start: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        for name, env, cmd_templates, snapshot in selected:
            resolved = [_resolve_command(c, args.python) for c in cmd_templates]
            ok, elapsed = _run_stage(name, env, resolved, snapshot, log_fh)
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
