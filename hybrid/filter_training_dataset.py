#!/usr/bin/env python3
"""Create a strict filtered training dataset without remapping labels.

Files are copied unchanged only if their existing SWC type set already
matches the canonical label policy:

- pyramidal: any subset of {1,2,3,4} that includes 1
- interneuron: exactly one of {1,2}, {1,3}, {1,2,3}
- purkinje: exactly {1,3}

All other files are excluded.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "training_morphologies"
DEFAULT_OUTPUT = ROOT / "data" / "training_morphologies_filtered"


def _type_set(path: Path) -> tuple[int, ...]:
    types: set[int] = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            types.add(int(float(parts[1])))
    return tuple(sorted(types))


def _keep_pyramidal(ts: tuple[int, ...]) -> bool:
    s = set(ts)
    return 1 in s and s.issubset({1, 2, 3, 4})


def _keep_interneuron(ts: tuple[int, ...]) -> bool:
    return ts in {(1, 2), (1, 3), (1, 2, 3)}


def _keep_purkinje(ts: tuple[int, ...]) -> bool:
    return ts == (1, 3)


KEEP_RULES = {
    "pyramidal": _keep_pyramidal,
    "interneuron": _keep_interneuron,
    "purkinje": _keep_purkinje,
}


def filter_dataset(input_dir: Path, output_dir: Path) -> dict:
    summary: dict[str, dict] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for cell_type, keep_fn in KEEP_RULES.items():
        src_dir = input_dir / cell_type
        if not src_dir.is_dir():
            continue

        dst_dir = output_dir / cell_type
        dst_dir.mkdir(parents=True, exist_ok=True)

        kept_sets = Counter()
        dropped_sets = Counter()
        kept_files: list[str] = []
        dropped_files: list[dict] = []

        for src in sorted(src_dir.glob("*.swc")):
            ts = _type_set(src)
            if keep_fn(ts):
                shutil.copy2(src, dst_dir / src.name)
                kept_sets[ts] += 1
                kept_files.append(src.name)
            else:
                dropped_sets[ts] += 1
                dropped_files.append({"file": src.name, "types": list(ts)})

        summary[cell_type] = {
            "n_kept": len(kept_files),
            "n_dropped": len(dropped_files),
            "kept_type_sets": {str(list(k)): v for k, v in sorted(kept_sets.items())},
            "dropped_type_sets": {str(list(k)): v for k, v in sorted(dropped_sets.items())},
            "kept_files": kept_files,
            "dropped_files": dropped_files,
        }

    (output_dir / "filter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a strict filtered SWC training dataset")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary = filter_dataset(args.input_dir, args.output_dir)
    print(f"Filtered dataset written to {args.output_dir}")
    for ct, info in summary.items():
        print(f"\n{ct}:")
        print(f"  kept: {info['n_kept']}")
        print(f"  dropped: {info['n_dropped']}")
        print(f"  kept sets: {info['kept_type_sets']}")
    print(f"\nSummary JSON: {args.output_dir / 'filter_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
