#!/usr/bin/env python3
"""Run SWC-Studio auto-labeling over CIC CA1 no-label batches.

Current behavior:
- input files are read directly from each ``batch_XX/no_label`` folder
- labels are not stripped here; the dataset already contains the no-label copies
- the current auto-label algorithm decides automatically whether to emit 3-class
  or 4-class output based on apical detection
- output SWCs are written to each matching ``batch_XX/auto_label_output`` folder
- no accuracy computation is performed in this script
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swcstudio.api import RuleBatchOptions, validation_auto_label_file

DEFAULT_DATASET_ROOT = ROOT / "data" / "CIC_CA1_Dataset"
DEFAULT_OPTIONS = RuleBatchOptions(
    soma=True,
    axon=True,
    apic=False,
    basal=True,
    rad=False,
    zip_output=False,
)


def _iter_batch_dirs(dataset_root: Path):
    for path in sorted(dataset_root.iterdir()):
        if path.is_dir() and path.name.startswith("batch_"):
            yield path


def _process_one_file(input_path: str, output_path: str) -> tuple[str, bool, str | None]:
    try:
        validation_auto_label_file(
            input_path,
            options=DEFAULT_OPTIONS,
            config_overrides=None,
            output_path=output_path,
            write_output=True,
            write_log=False,
        )
        return input_path, True, None
    except Exception as exc:  # noqa: BLE001
        return input_path, False, str(exc)


def _run_one_batch(
    batch_dir: Path,
    *,
    workers: int,
    skip_existing: bool,
) -> tuple[int, int, int, list[str]]:
    input_dir = batch_dir / "no_label"
    output_dir = batch_dir / "auto_label_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        return 0, 0, 0, [f"missing input folder: {input_dir}"]

    input_files = sorted(p for p in input_dir.glob("*.swc") if p.is_file())
    failures: list[str] = []
    processed = 0
    skipped = 0

    existing_outputs = {p.name for p in output_dir.glob("*.swc") if p.is_file()}
    input_names = {p.name for p in input_files}
    for stale in sorted(existing_outputs - input_names):
        (output_dir / stale).unlink()

    jobs: list[tuple[str, str]] = []
    for input_path in input_files:
        output_path = output_dir / input_path.name
        if skip_existing and output_path.exists():
            skipped += 1
            continue
        jobs.append((str(input_path), str(output_path)))

    if workers <= 1:
        for input_path, output_path in jobs:
            _, ok, error = _process_one_file(input_path, output_path)
            if ok:
                processed += 1
            else:
                failures.append(f"{Path(input_path).name}: {error}")
        return len(input_files), processed, skipped, failures

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_process_one_file, input_path, output_path) for input_path, output_path in jobs]
        for future in as_completed(futures):
            input_path, ok, error = future.result()
            if ok:
                processed += 1
            else:
                failures.append(f"{Path(input_path).name}: {error}")

    return len(input_files), processed, skipped, failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Run auto-labeling on CIC CA1 no-label batches.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root containing batch_XX folders",
    )
    parser.add_argument(
        "--batch",
        action="append",
        default=[],
        help="Optional batch name(s) like batch_01; can be passed multiple times",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 1))),
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs even when the output SWC already exists",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")

    requested = set(args.batch or [])
    batches = [b for b in _iter_batch_dirs(dataset_root) if not requested or b.name in requested]
    if not batches:
        raise RuntimeError("no matching batch folders found")

    workers = max(1, int(args.workers))
    skip_existing = not bool(args.overwrite)

    total_inputs = 0
    total_processed = 0
    total_skipped = 0
    total_failures: list[str] = []

    print(f"dataset_root: {dataset_root}", flush=True)
    print(f"workers: {workers}", flush=True)
    print(f"skip_existing: {skip_existing}", flush=True)
    for batch_dir in batches:
        total, processed, skipped, failures = _run_one_batch(
            batch_dir,
            workers=workers,
            skip_existing=skip_existing,
        )
        total_inputs += total
        total_processed += processed
        total_skipped += skipped
        total_failures.extend(f"{batch_dir.name}/{msg}" for msg in failures)
        print(
            f"{batch_dir.name}: inputs={total} processed={processed} skipped={skipped} "
            f"failed={len(failures)} output_dir={batch_dir / 'auto_label_output'}",
            flush=True,
        )

    print(f"total_inputs: {total_inputs}", flush=True)
    print(f"total_processed: {total_processed}", flush=True)
    print(f"total_skipped: {total_skipped}", flush=True)
    print(f"total_failed: {len(total_failures)}", flush=True)
    if total_failures:
        print("failures:", flush=True)
        for row in total_failures:
            print(f"  - {row}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
