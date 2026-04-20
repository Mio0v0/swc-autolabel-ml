#!/usr/bin/env python3
"""Download Allen Cell Types morphology data for training.

Downloads SWC reconstructions from the Allen Cell Types Database API,
organized by cell type into the training directory structure expected
by hybrid/train.py.

Available data (full reconstructions only):
- spiny (pyramidal): ~120 cells
- aspiny (interneuron): ~203 cells
- sparsely spiny (interneuron): ~44 cells

Usage:
    python -m hybrid.download_allen_data
    python -m hybrid.download_allen_data --output-dir data/training_morphologies
    python -m hybrid.download_allen_data --max-per-type 50   # limit downloads
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "data" / "training_morphologies"

ALLEN_API_BASE = "https://api.brain-map.org/api/v2"

# Query templates for the Allen Cell Types API
# ApiCellTypesSpecimenDetail is a flattened view that includes specimen tags
QUERY_TEMPLATE = (
    "{base}/data/query.json?"
    "criteria=model::ApiCellTypesSpecimenDetail,"
    "rma::criteria,"
    "[tag__dendrite_type$eq%27{dendrite_type}%27],"
    "[nr__reconstruction_type$eq%27full%27],"
    "rma::options[num_rows$eq{num_rows}][start_row$eq{start_row}]"
    "&only=specimen__id,nrwkf__id,tag__dendrite_type,line_name,"
    "structure__acronym,structure__layer,specimen__name"
)

DOWNLOAD_URL_TEMPLATE = "{base}/well_known_file_download/{nrwkf_id}"

# Map Allen dendrite_type tags to our cell type categories
DENDRITE_TYPE_TO_CELL_TYPE = {
    "spiny": "pyramidal",
    "aspiny": "interneuron",
    "sparsely spiny": "interneuron",  # grouped with aspiny as interneuron
}


def _fetch_json(url: str) -> dict:
    """Fetch JSON from a URL with retries."""
    for attempt in range(3):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Failed to fetch {url}: {e}") from e


def _download_file(url: str, dest: Path) -> bool:
    """Download a file with retries. Returns True on success."""
    for attempt in range(3):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            dest.write_bytes(data)
            return True
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            print(f"  FAILED: {dest.name}: {e}")
            return False


def query_specimens(
    dendrite_type: str,
    max_rows: int = 500,
) -> list[dict]:
    """Query Allen API for specimens of a given dendrite type."""
    all_rows: list[dict] = []
    start_row = 0
    batch_size = 50

    while start_row < max_rows:
        n = min(batch_size, max_rows - start_row)
        url = QUERY_TEMPLATE.format(
            base=ALLEN_API_BASE,
            dendrite_type=dendrite_type.replace(" ", "%20"),
            num_rows=n,
            start_row=start_row,
        )
        data = _fetch_json(url)
        rows = data.get("msg", [])
        if not rows:
            break
        all_rows.extend(rows)
        total = data.get("total_rows", 0)
        start_row += len(rows)
        if start_row >= total:
            break

    return all_rows


def download_cell_type(
    dendrite_type: str,
    output_dir: Path,
    max_files: int | None = None,
    skip_existing: bool = True,
) -> tuple[int, int, list[dict]]:
    """Download all SWC files for a dendrite type.

    Returns (downloaded_count, skipped_count, metadata_rows).
    """
    cell_type = DENDRITE_TYPE_TO_CELL_TYPE.get(dendrite_type, "other")
    cell_dir = output_dir / cell_type
    cell_dir.mkdir(parents=True, exist_ok=True)

    max_query = max_files if max_files else 500
    print(f"Querying Allen API for '{dendrite_type}' specimens...")
    specimens = query_specimens(dendrite_type, max_rows=max_query)
    print(f"  Found {len(specimens)} specimens")

    if max_files:
        specimens = specimens[:max_files]

    downloaded = 0
    skipped = 0
    metadata: list[dict] = []

    for i, spec in enumerate(specimens):
        nrwkf_id = spec.get("nrwkf__id")
        specimen_id = spec.get("specimen__id")
        specimen_name = spec.get("specimen__name", str(specimen_id))
        if not nrwkf_id:
            continue

        filename = f"{specimen_name}_{specimen_id}.swc"
        # Clean filename
        filename = filename.replace(";", "_").replace(" ", "_").replace("/", "_")
        dest = cell_dir / filename

        meta_row = {
            "cell_type": cell_type,
            "dendrite_type": dendrite_type,
            "specimen_id": specimen_id,
            "specimen_name": specimen_name,
            "nrwkf_id": nrwkf_id,
            "line_name": spec.get("line_name", ""),
            "structure": spec.get("structure__acronym", ""),
            "layer": spec.get("structure__layer", ""),
            "file_name": filename,
        }
        metadata.append(meta_row)

        if skip_existing and dest.exists():
            skipped += 1
            continue

        url = DOWNLOAD_URL_TEMPLATE.format(base=ALLEN_API_BASE, nrwkf_id=nrwkf_id)
        ok = _download_file(url, dest)
        if ok:
            downloaded += 1
            if (downloaded + skipped) % 10 == 0:
                print(f"  progress: {downloaded} downloaded, {skipped} skipped / {len(specimens)} total")
        time.sleep(0.2)  # Be polite to the API

    return downloaded, skipped, metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Allen Cell Types morphologies for training"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: data/training_morphologies)",
    )
    parser.add_argument(
        "--max-per-type",
        type=int,
        default=None,
        help="Max files to download per dendrite type (default: all)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip files that already exist (default: True)",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=["spiny", "aspiny", "sparsely spiny"],
        help="Dendrite types to download",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    all_metadata: list[dict] = []

    for dtype in args.types:
        print(f"\n{'='*60}")
        print(f"Downloading: {dtype}")
        print(f"{'='*60}")

        downloaded, skipped, metadata = download_cell_type(
            dendrite_type=dtype,
            output_dir=output_dir,
            max_files=args.max_per_type,
            skip_existing=args.skip_existing,
        )
        all_metadata.extend(metadata)
        print(f"  Done: {downloaded} downloaded, {skipped} skipped")

    # Write combined metadata CSV
    if all_metadata:
        meta_path = output_dir / "metadata.csv"
        fieldnames = list(all_metadata[0].keys())
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metadata)
        print(f"\nMetadata written to {meta_path}")

    # Summary
    print(f"\nDirectory structure:")
    for subdir in sorted(output_dir.iterdir()):
        if subdir.is_dir():
            n = len(list(subdir.glob("*.swc")))
            print(f"  {subdir.name}/: {n} SWC files")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
