#!/usr/bin/env python3
"""Download morphologies from NeuroMorpho.Org for training.

Downloads SWC reconstructions organized by cell type.

Available (approximate counts):
- Purkinje: ~1768 cells
- granule: ~500+ cells
- interneuron: ~2000+ cells

Usage:
    python -m hybrid.download_neuromorpho_data
    python -m hybrid.download_neuromorpho_data --types purkinje granule
    python -m hybrid.download_neuromorpho_data --max-per-type 50
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

NEUROMORPHO_API = "https://neuromorpho.org/api"

# Cell type queries and their target folder
CELL_TYPE_QUERIES = {
    "purkinje": {
        "query": "cell_type:Purkinje",
        "folder": "purkinje",
    },
    "granule": {
        "query": "cell_type:granule",
        "folder": "granule",
    },
    "interneuron": {
        "query": "cell_type:interneuron",
        "folder": "interneuron",
    },
}


def _fetch_json(url: str) -> dict:
    """Fetch JSON from NeuroMorpho API with retries."""
    for attempt in range(3):
        try:
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            raise RuntimeError(f"Failed to fetch {url}: {e}") from e


def _download_swc(archive: str, neuron_name: str, dest: Path) -> bool:
    """Download an SWC file from NeuroMorpho CNG version."""
    # NeuroMorpho SWC URL pattern (archive must be lowercase, spaces URL-encoded)
    archive_enc = archive.lower().replace(" ", "%20")
    neuron_enc = neuron_name.replace(" ", "%20")
    url = (
        f"https://neuromorpho.org/dableFiles/"
        f"{archive_enc}/CNG%20version/{neuron_enc}.CNG.swc"
    )
    for attempt in range(3):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            # Verify it looks like SWC (not an error page)
            text = data.decode("utf-8", errors="ignore")
            if "<html" in text.lower()[:200]:
                return False
            dest.write_bytes(data)
            return True
        except (urllib.error.URLError, TimeoutError):
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            return False


def query_neurons(cell_type_query: str, max_count: int) -> list[dict]:
    """Query NeuroMorpho API for neurons matching a cell type."""
    all_neurons: list[dict] = []
    page = 0
    page_size = 50

    while len(all_neurons) < max_count:
        remaining = max_count - len(all_neurons)
        size = min(page_size, remaining)
        url = f"{NEUROMORPHO_API}/neuron/select?q={cell_type_query}&size={size}&page={page}"
        try:
            data = _fetch_json(url)
        except RuntimeError as e:
            print(f"  API error at page {page}: {e}")
            break

        neurons = data.get("_embedded", {}).get("neuronResources", [])
        if not neurons:
            break
        all_neurons.extend(neurons)

        total_pages = data.get("page", {}).get("totalPages", 0)
        page += 1
        if page >= total_pages:
            break
        time.sleep(0.5)  # Be polite

    return all_neurons[:max_count]


def download_cell_type(
    type_key: str,
    output_dir: Path,
    max_files: int = 100,
    skip_existing: bool = True,
) -> tuple[int, int, list[dict]]:
    """Download SWC files for one cell type.

    Returns (downloaded, skipped, metadata).
    """
    config = CELL_TYPE_QUERIES[type_key]
    cell_dir = output_dir / config["folder"]
    cell_dir.mkdir(parents=True, exist_ok=True)

    print(f"Querying NeuroMorpho for '{type_key}'...")
    neurons = query_neurons(config["query"], max_count=max_files)
    print(f"  Found {len(neurons)} neurons")

    downloaded = 0
    skipped = 0
    failed = 0
    metadata: list[dict] = []

    for i, neuron in enumerate(neurons):
        name = neuron.get("neuron_name", "")
        archive = neuron.get("archive", "")
        if not name or not archive:
            continue

        filename = f"{name}.swc"
        dest = cell_dir / filename

        meta_row = {
            "cell_type": config["folder"],
            "source": "neuromorpho",
            "neuron_name": name,
            "archive": archive,
            "species": neuron.get("species", ""),
            "brain_region": str(neuron.get("brain_region", "")),
            "file_name": filename,
        }
        metadata.append(meta_row)

        if skip_existing and dest.exists():
            skipped += 1
            continue

        ok = _download_swc(archive, name, dest)
        if ok:
            downloaded += 1
        else:
            failed += 1

        if (downloaded + skipped + failed) % 10 == 0:
            print(
                f"  progress: {downloaded} downloaded, "
                f"{skipped} skipped, {failed} failed / {len(neurons)} total"
            )
        time.sleep(0.3)

    return downloaded, skipped, metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download NeuroMorpho.Org morphologies for training"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--max-per-type",
        type=int,
        default=100,
        help="Max files per cell type (default: 100)",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=["purkinje", "granule"],
        choices=list(CELL_TYPE_QUERIES.keys()),
        help="Cell types to download (default: purkinje granule)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    all_metadata: list[dict] = []
    for type_key in args.types:
        print(f"\n{'='*60}")
        print(f"Downloading: {type_key}")
        print(f"{'='*60}")
        downloaded, skipped, metadata = download_cell_type(
            type_key, output_dir,
            max_files=args.max_per_type,
        )
        all_metadata.extend(metadata)
        print(f"  Done: {downloaded} downloaded, {skipped} skipped")

    # Append to metadata CSV
    if all_metadata:
        meta_path = output_dir / "metadata_neuromorpho.csv"
        fieldnames = list(all_metadata[0].keys())
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metadata)
        print(f"\nMetadata: {meta_path}")

    # Summary
    print(f"\nDirectory structure:")
    for subdir in sorted(output_dir.iterdir()):
        if subdir.is_dir():
            n = len(list(subdir.glob("*.swc")))
            print(f"  {subdir.name}/: {n} SWC files")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
