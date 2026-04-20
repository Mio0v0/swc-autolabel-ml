#!/usr/bin/env python3
"""Create a canonicalized training dataset for the hybrid pipeline.

This script copies ``data/training_morphologies`` into a new output folder
and normalizes SWC type labels per cell type:

- pyramidal: keep only 1/2/3/4, remap everything else to 3
- interneuron: keep only 1/2/3, remap everything else to 3
- purkinje: keep only 1/3, remap everything else to 3

Additionally, if a file has no soma label (type 1), all root nodes are
relabeled to 1 so the cleaned file has a canonical soma anchor.

This is a pragmatic dataset-cleaning pass for training/evaluation. It does
not claim that every remapped non-canonical label is biologically identical
to dendrite; it simply forces the raw dataset into the label ontology used by
the current hybrid model.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "training_morphologies"
DEFAULT_OUTPUT = ROOT / "data" / "training_morphologies_canonical"

ALLOWED_BY_CELL_TYPE: dict[str, set[int]] = {
    "pyramidal": {1, 2, 3, 4},
    "interneuron": {1, 2, 3},
    "purkinje": {1, 3},
}


def _parse_type_and_parent(parts: list[str]) -> tuple[int, int]:
    return int(float(parts[1])), int(float(parts[6]))


def _clean_file(src: Path, dst: Path, cell_type: str) -> dict:
    allowed = ALLOWED_BY_CELL_TYPE[cell_type]
    root_rows: list[int] = []
    raw_types = Counter()
    cleaned_types = Counter()
    remapped = Counter()

    lines = src.read_text(encoding="utf-8", errors="ignore").splitlines()
    parsed_rows: list[tuple[int, list[str]] | None] = []

    for idx, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("#"):
            parsed_rows.append(None)
            continue
        parts = s.split()
        if len(parts) < 7:
            parsed_rows.append(None)
            continue
        t, parent = _parse_type_and_parent(parts)
        raw_types[t] += 1
        if parent == -1:
            root_rows.append(idx)
        parsed_rows.append((idx, parts))

    has_soma = raw_types.get(1, 0) > 0
    cleaned_lines: list[str] = []

    for idx, line in enumerate(lines):
        row = parsed_rows[idx]
        if row is None:
            cleaned_lines.append(line)
            continue

        _, parts = row
        raw_type, parent = _parse_type_and_parent(parts)

        if raw_type in allowed:
            new_type = raw_type
        else:
            new_type = 3
            remapped[raw_type] += 1

        # If the raw file has no explicit soma, promote root nodes to soma.
        if not has_soma and parent == -1:
            if new_type != 1:
                remapped[new_type] += 1
            new_type = 1

        parts[1] = str(new_type)
        cleaned_types[new_type] += 1
        cleaned_lines.append(" ".join(parts))

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(cleaned_lines) + "\n", encoding="utf-8")

    cleaned_set = tuple(sorted(cleaned_types))
    raw_set = tuple(sorted(raw_types))
    return {
        "file": src.name,
        "raw_types": list(raw_set),
        "cleaned_types": list(cleaned_set),
        "raw_counts": dict(sorted(raw_types.items())),
        "cleaned_counts": dict(sorted(cleaned_types.items())),
        "remapped_counts": dict(sorted(remapped.items())),
        "had_soma": has_soma,
        "n_root_rows": len(root_rows),
    }


def clean_dataset(input_dir: Path, output_dir: Path) -> dict:
    summary: dict[str, dict] = {}

    for cell_type, allowed in ALLOWED_BY_CELL_TYPE.items():
        src_dir = input_dir / cell_type
        if not src_dir.is_dir():
            continue

        per_file: list[dict] = []
        raw_agg = Counter()
        clean_agg = Counter()
        raw_sets = Counter()
        clean_sets = Counter()
        remap_agg = Counter()

        for src in sorted(src_dir.glob("*.swc")):
            dst = output_dir / cell_type / src.name
            info = _clean_file(src, dst, cell_type)
            per_file.append(info)
            raw_agg.update(info["raw_counts"])
            clean_agg.update(info["cleaned_counts"])
            raw_sets[tuple(info["raw_types"])] += 1
            clean_sets[tuple(info["cleaned_types"])] += 1
            remap_agg.update(info["remapped_counts"])

        summary[cell_type] = {
            "allowed_labels": sorted(allowed),
            "n_files": len(per_file),
            "raw_aggregate_counts": dict(sorted(raw_agg.items())),
            "cleaned_aggregate_counts": dict(sorted(clean_agg.items())),
            "raw_type_sets": {str(list(k)): v for k, v in sorted(raw_sets.items())},
            "cleaned_type_sets": {str(list(k)): v for k, v in sorted(clean_sets.items())},
            "remapped_label_counts": dict(sorted(remap_agg.items())),
            "files": per_file,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cleaning_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Canonicalize SWC training labels")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary = clean_dataset(args.input_dir, args.output_dir)

    print(f"Cleaned dataset written to {args.output_dir}")
    for cell_type, info in summary.items():
        print(f"\n{cell_type}:")
        print(f"  files: {info['n_files']}")
        print(f"  allowed labels: {info['allowed_labels']}")
        print(f"  raw aggregate: {info['raw_aggregate_counts']}")
        print(f"  cleaned aggregate: {info['cleaned_aggregate_counts']}")
        print(f"  remapped labels: {info['remapped_label_counts']}")

    print(f"\nSummary JSON: {args.output_dir / 'cleaning_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
