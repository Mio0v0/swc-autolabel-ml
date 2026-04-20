#!/usr/bin/env python3
"""Quick test: run Stage 1 cell-type detection on sample files.

Usage:
    python -m hybrid.test_stage1 data/training_morphologies/pyramidal/*.swc
    python -m hybrid.test_stage1 data/training_morphologies/interneuron/*.swc
    python -m hybrid.test_stage1 path/to/any.swc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid.cell_type_detector import detect_cell_type


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Stage 1 on SWC files")
    parser.add_argument("files", nargs="+", type=Path, help="SWC files to classify")
    parser.add_argument("--model", type=Path, default=None, help="Trained model path")
    args = parser.parse_args()

    results: list[tuple[str, str, float, str]] = []
    for swc_path in args.files:
        if not swc_path.exists():
            print(f"skip: {swc_path} not found")
            continue
        result = detect_cell_type(swc_path, model_path=args.model)
        top3 = sorted(result.probabilities.items(), key=lambda x: -x[1])[:3]
        prob_str = ", ".join(f"{k}={v:.3f}" for k, v in top3)
        labels = sorted(result.label_set)
        flag_str = " ".join(
            k for k, v in result.structure_flags.items() if v
        )
        results.append((swc_path.name, result.cell_type, result.confidence, str(labels)))
        print(
            f"{swc_path.name}\n"
            f"  type={result.cell_type}  conf={result.confidence:.3f}  "
            f"labels={labels}\n"
            f"  probs: {prob_str}\n"
            f"  flags: {flag_str}\n"
        )

    # Summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"Summary: {len(results)} files")
        from collections import Counter
        counts = Counter(r[1] for r in results)
        for ct, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {ct}: {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
