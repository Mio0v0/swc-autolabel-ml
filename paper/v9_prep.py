"""Build the v9 merged dataset by symlinking hpf_ca1 cells with apical labels
into the existing benchmark pyramidal pool.

After this runs, `data/v9_merged_dataset/` looks like:

    pyramidal/
        swc/
            <932 existing benchmark cells, symlinked from qc_diag_pruned>
            <1,012 hpf_ca1 cells with apical labels, symlinked>
    interneuron/
        swc/
            <805 existing benchmark cells, symlinked>

The hpf_ca1 cells are renamed `hpf_ca1__<original>.swc` so the hash-bucket
test split treats them as new files (and they don't collide with any
existing names). Only files that have at least one apical-labeled (type=4)
node in their GT are included — partial reconstructions without apical
are excluded.

Usage:
    python -m paper.v9_prep             # build v9_merged_dataset (symlinks)
    python -m paper.v9_prep --copy      # actually copy files (3.3 GB)
    python -m paper.v9_prep --check     # only print what would be done

Once built, retrain with:
    rm -rf hybrid/models/eval_tmp/*.pkl  # force fresh stage-1/2 train
    rm hybrid/models/eval_split.json     # force fresh hash-bucket split

    python -m hybrid.evaluate --data-dir data/v9_merged_dataset
    # ^ retrains stage 1 + stage 2 on the merged train split

    python -m paper.gnn_apical_basal --hidden 128 --n-layers 3 --dropout 0.0 \\
        --data-dir data/v9_merged_dataset
    # ^ retrains the GNN on the merged train pyramidals

    python -m hybrid.evaluate --data-dir data/v9_merged_dataset \\
        --use-gnn --use-subtree-stage2
    # ^ final v9 evaluation (stage1+2 cached, just inference + scoring)
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from hybrid.features import parse_swc

ROOT = Path(__file__).resolve().parent.parent
SRC_BENCHMARK = ROOT / "data" / "benchmark_pyramidal_interneuron_v1_qc_diag_pruned"
SRC_HPF_CA1 = ROOT / "data" / "hpf_ca1"
DEST = ROOT / "data" / "v9_merged_dataset"


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    """Make `dst` point to `src` content. Create parent dirs as needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copyfile(src, dst)
    else:
        # Windows symlinks need special handling; use junction (file copy
        # fallback) since SWC files are small enough that copying is fine.
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copyfile(src, dst)


def _has_apical(swc_path: Path) -> bool:
    """Return True if the SWC's type-column has at least one type=4 node."""
    try:
        nodes = parse_swc(swc_path)
    except Exception:
        return False
    return any(n.type == 4 for n in nodes)


def build(dest: Path, copy: bool = False, check_only: bool = False) -> dict:
    if check_only:
        print(f"[CHECK MODE] would build {dest}")
    elif dest.exists():
        print(f"WARNING: {dest} already exists. Skipping existing entries.")
    else:
        dest.mkdir(parents=True)

    counts = {"benchmark_pyr": 0, "benchmark_inter": 0,
              "hpf_ca1_with_apical": 0, "hpf_ca1_without_apical": 0,
              "hpf_ca1_skipped_collision": 0}

    # === existing benchmark ===
    print("\n--- Linking existing benchmark ---")
    for ct in ("pyramidal", "interneuron"):
        src_swc = SRC_BENCHMARK / ct / "swc"
        if not src_swc.is_dir():
            print(f"  WARN: {src_swc} not found — skipping")
            continue
        files = sorted(src_swc.glob("*.swc"))
        for f in files:
            dst = dest / ct / "swc" / f.name
            if not check_only:
                _link_or_copy(f, dst, copy)
            counts["benchmark_pyr" if ct == "pyramidal" else "benchmark_inter"] += 1
        # Also link manifest.csv if present
        for extra in ("manifest.csv", "README.md"):
            src = SRC_BENCHMARK / ct / extra
            if src.is_file() and not check_only:
                _link_or_copy(src, dest / ct / extra, copy)
        print(f"  {ct:<12s}: {counts['benchmark_pyr' if ct == 'pyramidal' else 'benchmark_inter']} files")

    # === hpf_ca1 with apical ===
    print("\n--- Linking hpf_ca1 cells with apical labels ---")
    pyr_dst = dest / "pyramidal" / "swc"
    existing_names = {f.name for f in pyr_dst.glob("*.swc")} if pyr_dst.exists() else set()

    hpf_files = sorted(SRC_HPF_CA1.rglob("*.swc"))
    print(f"  scanning {len(hpf_files)} hpf_ca1 SWCs for apical labels...")
    for i, f in enumerate(hpf_files):
        if (i + 1) % 200 == 0:
            print(f"    ... checked {i+1}/{len(hpf_files)}")
        if not _has_apical(f):
            counts["hpf_ca1_without_apical"] += 1
            continue
        # Rename with hpf_ca1__ prefix to avoid collisions and preserve
        # provenance in the file name.
        new_name = f"hpf_ca1__{f.name}"
        if new_name in existing_names:
            counts["hpf_ca1_skipped_collision"] += 1
            continue
        existing_names.add(new_name)
        dst = pyr_dst / new_name
        if not check_only:
            _link_or_copy(f, dst, copy)
        counts["hpf_ca1_with_apical"] += 1

    print(f"  added: {counts['hpf_ca1_with_apical']} files")
    print(f"  excluded (no apical): {counts['hpf_ca1_without_apical']}")
    print(f"  skipped (name collision): {counts['hpf_ca1_skipped_collision']}")

    print("\n--- Summary ---")
    total_pyr = counts["benchmark_pyr"] + counts["hpf_ca1_with_apical"]
    print(f"  total pyramidal pool: {total_pyr}  "
          f"(benchmark={counts['benchmark_pyr']}, hpf_ca1={counts['hpf_ca1_with_apical']})")
    print(f"  total interneuron pool: {counts['benchmark_inter']}")
    print(f"  v9 dataset at: {dest}")
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--copy", action="store_true",
                    help="copy files instead of symlinking (uses 3.3 GB extra disk)")
    ap.add_argument("--check", dest="check_only", action="store_true",
                    help="only report what would be done; do not modify disk")
    args = ap.parse_args()
    build(args.dest, copy=args.copy, check_only=args.check_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
