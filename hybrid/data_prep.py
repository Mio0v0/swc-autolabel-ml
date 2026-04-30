#!/usr/bin/env python3
"""Data preparation toolbox for the hybrid pipeline.

A single entry point that bundles the dataset-construction utilities used
upstream of the QC pipeline:

    download-allen         Download Allen Cell Types morphologies.
    download-neuromorpho   Download morphologies from NeuroMorpho.Org.
    build-source-pool      Filter raw downloads + lab data into a clean
                           "usable" source pool with canonical labels.
    build-benchmark        Assemble a provenance-tracked benchmark from
                           the source pool with per-type sampling.

Usage:
    python -m hybrid.data_prep download-allen
    python -m hybrid.data_prep download-neuromorpho --types purkinje granule
    python -m hybrid.data_prep build-source-pool
    python -m hybrid.data_prep build-benchmark --per-type 1000

Public Python API:
    from hybrid.data_prep import (
        download_allen, download_neuromorpho,
        build_source_pool, build_benchmark,
    )
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CELL_TYPES = ("pyramidal", "interneuron")


# =============================================================================
# DOWNLOAD-ALLEN  (was download_allen_data.py)
# =============================================================================

ALLEN_API_BASE = "https://api.brain-map.org/api/v2"

_ALLEN_QUERY_TEMPLATE = (
    "{base}/data/query.json?"
    "criteria=model::ApiCellTypesSpecimenDetail,"
    "rma::criteria,"
    "[tag__dendrite_type$eq%27{dendrite_type}%27],"
    "[nr__reconstruction_type$eq%27full%27],"
    "rma::options[num_rows$eq{num_rows}][start_row$eq{start_row}]"
    "&only=specimen__id,nrwkf__id,tag__dendrite_type,line_name,"
    "structure__acronym,structure__layer,specimen__name"
)

_ALLEN_DOWNLOAD_URL_TEMPLATE = "{base}/well_known_file_download/{nrwkf_id}"

ALLEN_DENDRITE_TYPE_TO_CELL_TYPE = {
    "spiny": "pyramidal",
    "aspiny": "interneuron",
    "sparsely spiny": "interneuron",
}


def _allen_fetch_json(url: str) -> dict:
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


def _allen_download_file(url: str, dest: Path) -> bool:
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


def _allen_query_specimens(dendrite_type: str, max_rows: int = 500) -> list[dict]:
    all_rows: list[dict] = []
    start_row = 0
    batch_size = 50
    while start_row < max_rows:
        n = min(batch_size, max_rows - start_row)
        url = _ALLEN_QUERY_TEMPLATE.format(
            base=ALLEN_API_BASE,
            dendrite_type=dendrite_type.replace(" ", "%20"),
            num_rows=n,
            start_row=start_row,
        )
        data = _allen_fetch_json(url)
        rows = data.get("msg", [])
        if not rows:
            break
        all_rows.extend(rows)
        total = data.get("total_rows", 0)
        start_row += len(rows)
        if start_row >= total:
            break
    return all_rows


def download_allen(
    dendrite_type: str,
    output_dir: Path,
    max_files: int | None = None,
    skip_existing: bool = True,
) -> tuple[int, int, list[dict]]:
    """Download SWC files for one Allen dendrite type."""
    cell_type = ALLEN_DENDRITE_TYPE_TO_CELL_TYPE.get(dendrite_type, "other")
    cell_dir = output_dir / cell_type
    cell_dir.mkdir(parents=True, exist_ok=True)

    max_query = max_files if max_files else 500
    print(f"Querying Allen API for '{dendrite_type}' specimens...")
    specimens = _allen_query_specimens(dendrite_type, max_rows=max_query)
    print(f"  Found {len(specimens)} specimens")

    if max_files:
        specimens = specimens[:max_files]

    downloaded = 0
    skipped = 0
    metadata: list[dict] = []

    for spec in specimens:
        nrwkf_id = spec.get("nrwkf__id")
        specimen_id = spec.get("specimen__id")
        specimen_name = spec.get("specimen__name", str(specimen_id))
        if not nrwkf_id:
            continue

        filename = f"{specimen_name}_{specimen_id}.swc"
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

        url = _ALLEN_DOWNLOAD_URL_TEMPLATE.format(base=ALLEN_API_BASE, nrwkf_id=nrwkf_id)
        ok = _allen_download_file(url, dest)
        if ok:
            downloaded += 1
            if (downloaded + skipped) % 10 == 0:
                print(f"  progress: {downloaded} downloaded, {skipped} skipped / {len(specimens)} total")
        time.sleep(0.2)

    return downloaded, skipped, metadata


# =============================================================================
# DOWNLOAD-NEUROMORPHO  (was download_neuromorpho_data.py)
# =============================================================================

NEUROMORPHO_API = "https://neuromorpho.org/api"

NEUROMORPHO_CELL_TYPE_QUERIES = {
    "pyramidal":   {"query": "cell_type:pyramidal",   "folder": "pyramidal"},
    "purkinje":    {"query": "cell_type:Purkinje",    "folder": "purkinje"},
    "granule":     {"query": "cell_type:granule",     "folder": "granule"},
    "interneuron": {"query": "cell_type:interneuron", "folder": "interneuron"},
}


def _nm_fetch_json(url: str) -> dict:
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


def _nm_download_swc(archive: str, neuron_name: str, dest: Path) -> bool:
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


def _nm_query_neurons(cell_type_query: str, max_count: int) -> list[dict]:
    all_neurons: list[dict] = []
    page = 0
    page_size = 50
    while len(all_neurons) < max_count:
        remaining = max_count - len(all_neurons)
        size = min(page_size, remaining)
        url = f"{NEUROMORPHO_API}/neuron/select?q={cell_type_query}&size={size}&page={page}"
        try:
            data = _nm_fetch_json(url)
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
        time.sleep(0.5)
    return all_neurons[:max_count]


def download_neuromorpho(
    type_key: str,
    output_dir: Path,
    max_files: int = 100,
    skip_existing: bool = True,
) -> tuple[int, int, list[dict]]:
    """Download SWC files for one NeuroMorpho cell-type query."""
    config = NEUROMORPHO_CELL_TYPE_QUERIES[type_key]
    cell_dir = output_dir / config["folder"]
    cell_dir.mkdir(parents=True, exist_ok=True)

    print(f"Querying NeuroMorpho for '{type_key}'...")
    neurons = _nm_query_neurons(config["query"], max_count=max_files)
    print(f"  Found {len(neurons)} neurons")

    downloaded = 0
    skipped = 0
    failed = 0
    metadata: list[dict] = []

    for neuron in neurons:
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

        ok = _nm_download_swc(archive, name, dest)
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


# =============================================================================
# BUILD-SOURCE-POOL  (was build_usable_source_pool.py)
# =============================================================================

_POOL_TRAIN = ROOT / "data" / "training_morphologies"
_POOL_LAB = ROOT / "data" / "CIC_CA1_Dataset"
_POOL_PYRAMIDAL_TYPES = {(1, 2, 3), (1, 2, 3, 4)}
_POOL_INTERNEURON_TYPES = {(1, 2), (1, 3), (1, 2, 3)}


def _pool_df_to_structured(df):
    import numpy as np
    DTYPE = np.dtype([
        ("id", np.int64), ("type", np.int64),
        ("x", np.float64), ("y", np.float64), ("z", np.float64),
        ("radius", np.float64), ("parent", np.int64),
    ])
    arr = np.empty(len(df), dtype=DTYPE)
    for col in ["id", "type", "x", "y", "z", "radius", "parent"]:
        arr[col] = df[col].to_numpy()
    return arr


def _pool_process_swc(path: Path):
    import numpy as np
    from swcstudio.core.swc_io import parse_swc_text_preserve_tokens
    from swcstudio.core.validation_engine import (
        consolidate_complex_somas_array, _array_to_swc_text,
    )

    text = path.read_text(encoding="utf-8", errors="ignore")
    df = parse_swc_text_preserve_tokens(text)
    arr = _pool_df_to_structured(df)
    type_set = tuple(sorted(set(int(v) for v in arr["type"].tolist())))
    soma_count = int(np.sum(arr["type"] == 1))
    if soma_count == 0:
        return type_set, False, None, {
            "original_soma_count": 0, "cleaned_soma_count": 0, "collapsed": False,
        }

    res = consolidate_complex_somas_array(arr)
    final_arr = np.array(res.get("array", arr), copy=True)
    cleaned_type_set = tuple(sorted(set(int(v) for v in final_arr["type"].tolist())))
    swc_text = _array_to_swc_text(final_arr)
    return cleaned_type_set, True, swc_text, {
        "original_soma_count": soma_count,
        "cleaned_soma_count": int(np.sum(final_arr["type"] == 1)),
        "collapsed": bool(res.get("changed", False)),
    }


def _pool_slug(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return out.strip("_") or "na"


def _pool_copied_name(source: str, path: Path, meta: dict | None = None) -> str:
    meta = meta or {}
    if source == "neuromorpho":
        archive = _pool_slug(str(meta.get("archive", "na")))
        return f"{source}__{archive}__{path.name}"
    if source == "allen":
        specimen = _pool_slug(str(meta.get("specimen_id", "na")))
        return f"{source}__{specimen}__{path.name}"
    batch = _pool_slug(str(meta.get("meta_batch", meta.get("batch", "na"))))
    return f"{source}__{batch}__{path.name}"


def _pool_copy_entry(src: Path, dest_dir: Path, copied_name: str, swc_text: str) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / copied_name).write_text(swc_text, encoding="utf-8")


def build_source_pool(output_dir: Path) -> dict:
    """Filter raw downloads + lab data into a usable source pool."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    allen_meta: dict[str, dict] = {}
    nm_meta: dict[str, dict] = {}
    with open(_POOL_TRAIN / "metadata.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            allen_meta[row["file_name"]] = row
    with open(_POOL_TRAIN / "metadata_neuromorpho.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            nm_meta[row["file_name"]] = row

    summary: dict[str, dict] = {
        "pyramidal":   {"source_counts": Counter(), "type_sets": Counter(), "selected": 0},
        "interneuron": {"source_counts": Counter(), "type_sets": Counter(), "selected": 0},
    }
    manifests: dict[str, list[dict]] = {"pyramidal": [], "interneuron": []}

    # Allen and NeuroMorpho from training_morphologies
    for cell_type in ("pyramidal", "interneuron"):
        folder = _POOL_TRAIN / cell_type
        keep = _POOL_PYRAMIDAL_TYPES if cell_type == "pyramidal" else _POOL_INTERNEURON_TYPES
        for path in sorted(folder.glob("*.swc")):
            source = None
            meta = None
            if path.name in allen_meta and allen_meta[path.name].get("cell_type") == cell_type:
                source = "allen"
                meta = allen_meta[path.name]
            elif path.name in nm_meta and nm_meta[path.name].get("cell_type") == cell_type:
                source = "neuromorpho"
                meta = nm_meta[path.name]
            if source is None:
                continue
            cleaned_type_set, has_soma, swc_text, info = _pool_process_swc(path)
            if not has_soma or cleaned_type_set not in keep:
                continue
            copied_name = _pool_copied_name(source, path, meta)
            _pool_copy_entry(path, output_dir / cell_type / "swc", copied_name, swc_text)
            manifests[cell_type].append({
                "cell_type": cell_type, "source": source,
                "copied_file": copied_name, "original_path": str(path),
                "type_set": " ".join(str(v) for v in cleaned_type_set),
                "original_soma_count": info["original_soma_count"],
                "cleaned_soma_count": info["cleaned_soma_count"],
                "collapsed_multi_soma": int(info["collapsed"]),
                **{f"meta_{k}": v for k, v in meta.items()},
            })
            summary[cell_type]["source_counts"][source] += 1
            summary[cell_type]["type_sets"][cleaned_type_set] += 1
            summary[cell_type]["selected"] += 1

    # Lab pyramidal from CIC_CA1_Dataset
    for path in sorted(_POOL_LAB.glob("batch_*/original/*.swc")):
        cleaned_type_set, has_soma, swc_text, info = _pool_process_swc(path)
        if not has_soma or cleaned_type_set not in _POOL_PYRAMIDAL_TYPES:
            continue
        copied_name = _pool_copied_name("lab", path, {"batch": path.parent.parent.name})
        _pool_copy_entry(path, output_dir / "pyramidal" / "swc", copied_name, swc_text)
        manifests["pyramidal"].append({
            "cell_type": "pyramidal", "source": "lab",
            "copied_file": copied_name, "original_path": str(path),
            "type_set": " ".join(str(v) for v in cleaned_type_set),
            "original_soma_count": info["original_soma_count"],
            "cleaned_soma_count": info["cleaned_soma_count"],
            "collapsed_multi_soma": int(info["collapsed"]),
            "meta_lab_dataset": "CIC_CA1_Dataset",
            "meta_batch": path.parent.parent.name,
            "meta_subset": path.parent.name,
        })
        summary["pyramidal"]["source_counts"]["lab"] += 1
        summary["pyramidal"]["type_sets"][cleaned_type_set] += 1
        summary["pyramidal"]["selected"] += 1

    for cell_type, rows in manifests.items():
        ct_dir = output_dir / cell_type
        ct_dir.mkdir(parents=True, exist_ok=True)
        with open(ct_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
            if rows:
                fieldnames: list[str] = []
                seen: set[str] = set()
                for row in rows:
                    for key in row.keys():
                        if key not in seen:
                            seen.add(key)
                            fieldnames.append(key)
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    out = {
        "description": (
            "Usable source pools only: canonical labels only, soma required, "
            "connected multi-node soma groups collapsed to one soma anchor."
        ),
        "summary": {
            ct: {
                "selected": summary[ct]["selected"],
                "source_counts": dict(summary[ct]["source_counts"]),
                "type_sets": {str(list(k)): v for k, v in sorted(summary[ct]["type_sets"].items())},
            }
            for ct in ("pyramidal", "interneuron")
        },
    }
    (output_dir / "source_pool_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Usable Source Pools\n\n"
        "Only usable source data are kept here: canonical labels only, soma required, "
        "and connected multi-node soma groups collapsed to one soma node.\n",
        encoding="utf-8",
    )
    return out


# =============================================================================
# BUILD-BENCHMARK  (was build_combined_benchmark.py)
# =============================================================================

_BENCH_TRAINING_DIR = ROOT / "data" / "training_morphologies"
_BENCH_LAB_DIR = ROOT / "data" / "CIC_CA1_Dataset"

_BENCH_SOURCE_PRIORITY = {
    "pyramidal":   ["allen", "lab", "neuromorpho"],
    "interneuron": ["allen", "neuromorpho", "lab"],
}
_BENCH_SOURCE_TARGETS = {
    "pyramidal": {"allen": 104, "neuromorpho": 400},
}


def _bench_type_set(path: Path) -> tuple[int, ...]:
    ts: set[int] = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            ts.add(int(float(parts[1])))
    return tuple(sorted(ts))


def _bench_keep_pyramidal(ts: tuple[int, ...]) -> bool:
    return ts in {(1, 2, 3), (1, 2, 3, 4)}


def _bench_keep_interneuron(ts: tuple[int, ...]) -> bool:
    return ts in {(1, 2), (1, 3), (1, 2, 3)}


_BENCH_KEEP_RULES = {
    "pyramidal": _bench_keep_pyramidal,
    "interneuron": _bench_keep_interneuron,
}


def _bench_load_metadata():
    allen_meta: dict[str, dict] = {}
    nm_meta: dict[str, dict] = {}
    allen_csv = _BENCH_TRAINING_DIR / "metadata.csv"
    nm_csv = _BENCH_TRAINING_DIR / "metadata_neuromorpho.csv"
    if allen_csv.exists():
        with open(allen_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                allen_meta[row["file_name"]] = row
    if nm_csv.exists():
        with open(nm_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                nm_meta[row["file_name"]] = row
    return allen_meta, nm_meta


def _bench_gather_candidates():
    allen_meta, nm_meta = _bench_load_metadata()
    out: dict[str, list[dict]] = {ct: [] for ct in CELL_TYPES}

    for ct in CELL_TYPES:
        folder = _BENCH_TRAINING_DIR / ct
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.swc")):
            name = path.name
            source = None
            meta: dict[str, str] = {}
            if name in allen_meta:
                source = "allen"; meta = allen_meta[name]
            elif name in nm_meta:
                source = "neuromorpho"; meta = nm_meta[name]
            else:
                if ct == "interneuron":
                    source = "lab"
            if source is None:
                continue
            ts = _bench_type_set(path)
            if not _BENCH_KEEP_RULES[ct](ts):
                continue
            out[ct].append({
                "cell_type": ct, "source": source,
                "path": str(path), "file_name": name,
                "type_set": list(ts), "metadata": meta,
            })

    for path in sorted(_BENCH_LAB_DIR.glob("batch_*/original/*.swc")):
        ts = _bench_type_set(path)
        if not _bench_keep_pyramidal(ts):
            continue
        out["pyramidal"].append({
            "cell_type": "pyramidal", "source": "lab",
            "path": str(path), "file_name": path.name,
            "type_set": list(ts),
            "metadata": {
                "lab_dataset": "CIC_CA1_Dataset",
                "batch": path.parent.parent.name,
                "subset": path.parent.name,
            },
        })
    return out


def _bench_select_examples(candidates: list[dict], cell_type: str, limit: int) -> list[dict]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        by_source[row["source"]].append(row)

    selected: list[dict] = []
    seen_paths: set[str] = set()

    for source, target in _BENCH_SOURCE_TARGETS.get(cell_type, {}).items():
        for row in by_source.get(source, []):
            if len(selected) >= limit:
                break
            if row["path"] in seen_paths:
                continue
            if sum(1 for x in selected if x["source"] == source) >= target:
                break
            selected.append(row); seen_paths.add(row["path"])

    for source in _BENCH_SOURCE_PRIORITY[cell_type]:
        for row in by_source.get(source, []):
            if len(selected) >= limit:
                break
            if row["path"] in seen_paths:
                continue
            selected.append(row); seen_paths.add(row["path"])
        if len(selected) >= limit:
            break
    return selected


def build_benchmark(output_dir: Path, per_type: int = 1000) -> dict:
    """Assemble a provenance-tracked benchmark from candidate SWC sources."""
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _bench_gather_candidates()
    summary: dict[str, dict] = {}

    for ct in CELL_TYPES:
        selected = _bench_select_examples(candidates[ct], ct, per_type)
        ct_dir = output_dir / ct
        swc_dir = ct_dir / "swc"
        swc_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows: list[dict] = []
        source_counts: Counter = Counter()
        type_sets: Counter = Counter()

        for idx, row in enumerate(selected, start=1):
            src = Path(row["path"])
            dest_name = f"{row['source']}__{src.name}"
            shutil.copy2(src, swc_dir / dest_name)
            source_counts[row["source"]] += 1
            type_sets[tuple(row["type_set"])] += 1
            manifest_rows.append({
                "benchmark_id": f"{ct}_{idx:04d}",
                "cell_type": ct, "source": row["source"],
                "copied_file": dest_name,
                "original_path": row["path"],
                "type_set": " ".join(map(str, row["type_set"])),
                **{f"meta_{k}": v for k, v in row["metadata"].items()},
            })

        with open(ct_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
            if manifest_rows:
                fieldnames: list[str] = []
                seen: set[str] = set()
                for row in manifest_rows:
                    for key in row.keys():
                        if key not in seen:
                            seen.add(key)
                            fieldnames.append(key)
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(manifest_rows)
            else:
                f.write("benchmark_id,cell_type,source,copied_file,original_path,type_set\n")

        summary[ct] = {
            "requested": per_type,
            "available_canonical": len(candidates[ct]),
            "selected": len(selected),
            "shortfall": max(0, per_type - len(selected)),
            "source_counts": dict(source_counts),
            "type_sets": {str(list(k)): v for k, v in sorted(type_sets.items())},
        }

    provenance = {
        "description": (
            "Canonical pyramidal/interneuron benchmark built from local Allen, "
            "lab, and NeuroMorpho sources. Files are copied unchanged and only "
            "included if their existing SWC type sets satisfy the canonical task."
        ),
        "cell_types": list(CELL_TYPES),
        "per_type_target": per_type,
        "source_priority": _BENCH_SOURCE_PRIORITY,
        "source_targets": _BENCH_SOURCE_TARGETS,
        "summary": summary,
    }
    (output_dir / "benchmark_summary.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write("# Canonical Pyramidal/Interneuron Benchmark\n\n")
        f.write("This benchmark contains copied SWC files with provenance manifests.\n")
        f.write("Files were included only if their raw SWC type sets already matched the canonical rules.\n\n")
        for ct in CELL_TYPES:
            s = summary[ct]
            f.write(f"## {ct}\n")
            f.write(f"- requested: {s['requested']}\n")
            f.write(f"- available canonical: {s['available_canonical']}\n")
            f.write(f"- selected: {s['selected']}\n")
            f.write(f"- shortfall: {s['shortfall']}\n")
            f.write(f"- source counts: {s['source_counts']}\n\n")
    return provenance


# =============================================================================
# CLI
# =============================================================================

def _cmd_download_allen(args) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    all_metadata: list[dict] = []
    for dtype in args.types:
        print(f"\n{'='*60}")
        print(f"Downloading: {dtype}")
        print(f"{'='*60}")
        downloaded, skipped, metadata = download_allen(
            dendrite_type=dtype,
            output_dir=output_dir,
            max_files=args.max_per_type,
            skip_existing=args.skip_existing,
        )
        all_metadata.extend(metadata)
        print(f"  Done: {downloaded} downloaded, {skipped} skipped")

    if all_metadata:
        meta_path = output_dir / "metadata.csv"
        fieldnames = list(all_metadata[0].keys())
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metadata)
        print(f"\nMetadata written to {meta_path}")

    print(f"\nDirectory structure:")
    for subdir in sorted(output_dir.iterdir()):
        if subdir.is_dir():
            n = len(list(subdir.glob("*.swc")))
            print(f"  {subdir.name}/: {n} SWC files")
    return 0


def _cmd_download_neuromorpho(args) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    all_metadata: list[dict] = []
    for type_key in args.types:
        print(f"\n{'='*60}")
        print(f"Downloading: {type_key}")
        print(f"{'='*60}")
        downloaded, skipped, metadata = download_neuromorpho(
            type_key, output_dir, max_files=args.max_per_type,
        )
        all_metadata.extend(metadata)
        print(f"  Done: {downloaded} downloaded, {skipped} skipped")

    if all_metadata:
        meta_path = output_dir / "metadata_neuromorpho.csv"
        merged: dict[str, dict] = {}
        if meta_path.exists():
            with open(meta_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    merged[row["file_name"]] = row
        for row in all_metadata:
            merged[row["file_name"]] = row
        fieldnames = list(all_metadata[0].keys())
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted(merged.values(), key=lambda r: (r["cell_type"], r["file_name"])))
        print(f"\nMetadata: {meta_path}")

    print(f"\nDirectory structure:")
    for subdir in sorted(output_dir.iterdir()):
        if subdir.is_dir():
            n = len(list(subdir.glob("*.swc")))
            print(f"  {subdir.name}/: {n} SWC files")
    return 0


def _cmd_build_source_pool(args) -> int:
    out = build_source_pool(args.output_dir)
    print(json.dumps(out, indent=2))
    return 0


def _cmd_build_benchmark(args) -> int:
    summary = build_benchmark(args.output_dir, args.per_type)
    print(f"Benchmark written to {args.output_dir}")
    for ct in CELL_TYPES:
        s = summary["summary"][ct]
        print(f"\n{ct}:")
        print(f"  requested: {s['requested']}")
        print(f"  available canonical: {s['available_canonical']}")
        print(f"  selected: {s['selected']}")
        print(f"  shortfall: {s['shortfall']}")
        print(f"  source counts: {s['source_counts']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="hybrid.data_prep",
        description="Data preparation toolbox (download / build-source-pool / build-benchmark).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="<command>")

    # --- download-allen ---
    pa = sub.add_parser("download-allen",
                        help="Download Allen Cell Types morphologies.")
    pa.add_argument("--output-dir", type=Path,
                    default=ROOT / "data" / "training_morphologies")
    pa.add_argument("--max-per-type", type=int, default=None,
                    help="Max files per dendrite type (default: all)")
    pa.add_argument("--skip-existing", action="store_true", default=True,
                    help="Skip files that already exist (default: True)")
    pa.add_argument("--types", nargs="+",
                    default=["spiny", "aspiny", "sparsely spiny"],
                    help="Allen dendrite_type tags to download")
    pa.set_defaults(func=_cmd_download_allen)

    # --- download-neuromorpho ---
    pn = sub.add_parser("download-neuromorpho",
                        help="Download morphologies from NeuroMorpho.Org.")
    pn.add_argument("--output-dir", type=Path,
                    default=ROOT / "data" / "training_morphologies")
    pn.add_argument("--max-per-type", type=int, default=100,
                    help="Max files per cell type (default: 100)")
    pn.add_argument("--types", nargs="+",
                    default=["purkinje", "granule"],
                    choices=list(NEUROMORPHO_CELL_TYPE_QUERIES.keys()),
                    help="Cell types to download")
    pn.set_defaults(func=_cmd_download_neuromorpho)

    # --- build-source-pool ---
    pp = sub.add_parser("build-source-pool",
                        help="Filter raw downloads + lab data into a usable source pool.")
    pp.add_argument("--output-dir", type=Path,
                    default=ROOT / "data" / "source_pools_usable")
    pp.set_defaults(func=_cmd_build_source_pool)

    # --- build-benchmark ---
    pb = sub.add_parser("build-benchmark",
                        help="Assemble a provenance-tracked benchmark from candidate sources.")
    pb.add_argument("--output-dir", type=Path,
                    default=ROOT / "data" / "benchmark_pyramidal_interneuron_v1")
    pb.add_argument("--per-type", type=int, default=1000)
    pb.set_defaults(func=_cmd_build_benchmark)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
