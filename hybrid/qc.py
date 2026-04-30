#!/usr/bin/env python3
"""Dataset quality-control toolbox for the hybrid pipeline.

A single entry point that bundles all dataset-cleanup utilities:

    diagnose     Inspect why a low-F1 file scored poorly. Verdict is one of
                 BAD GT, BORDERLINE, SUSPICIOUS GT, or APPEARS HEALTHY.
    clean-soma   Consolidate multi-node somas and drop files without soma.
                 Produces ``..._soma_clean`` from the raw benchmark.
    filter       Conservative morphology-QC filter that drops files looking
                 structurally out-of-domain (no soma, single subtree, extreme
                 extent, etc.). Produces ``..._qc_filtered``.
    prune        Drop files with diagnostic verdicts in --drop-classes from a
                 benchmark, copying survivors to a new directory.

Usage:
    python -m hybrid.qc diagnose path/to/file.swc
    python -m hybrid.qc diagnose --from-csv hybrid/models/per_file_scores.csv --top 15
    python -m hybrid.qc clean-soma --input-dir ...  --output-dir ...
    python -m hybrid.qc filter --input-dir ... --output-dir ...
    python -m hybrid.qc prune --source ... --scores ... --out ... --apply

Public Python API:
    from hybrid.qc import diagnose, clean_benchmark, filter_benchmark, prune_benchmark
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid.features import extract_global_features, parse_swc  # noqa: E402

CELL_TYPES = ("pyramidal", "interneuron")


# =============================================================================
# DIAGNOSE  (was diagnose_bad_file.py)
# =============================================================================

LABEL_NAMES = {1: "soma", 2: "axon", 3: "basal", 4: "apical"}

HEALTHY = {
    "pyramidal": {
        "axon":   (0.30, 0.90),
        "basal":  (0.05, 0.70),
        "apical": (0.05, 0.40),
    },
    "interneuron": {
        "axon":   (0.30, 0.95),
        "basal":  (0.05, 0.50),
    },
}


def diagnose(file_path: Path, cell_type_hint: str | None = None) -> dict:
    """Return a diagnostic dict with flags, verdict, and recommendation."""
    nodes = parse_swc(file_path)
    if not nodes:
        return {"file": str(file_path), "verdict": "empty file"}

    counts = Counter(n.type for n in nodes)
    total = sum(counts.values())
    fracs = {LABEL_NAMES.get(t, str(t)): counts.get(t, 0) / total for t in (1, 2, 3, 4)}

    if cell_type_hint is None:
        p = str(file_path).lower()
        if "pyramidal" in p:
            cell_type_hint = "pyramidal"
        elif "interneuron" in p:
            cell_type_hint = "interneuron"

    flags: list[str] = []

    if fracs["axon"] < 0.05:
        flags.append(f"axon_fraction={fracs['axon']*100:.1f}% (<5%) → dendrite-only reconstruction")
    if fracs["axon"] > 0.95:
        flags.append(f"axon_fraction={fracs['axon']*100:.1f}% (>95%) → axon-only reconstruction")

    neurite_classes_present = sum(1 for k in ("axon", "basal", "apical") if fracs[k] > 0.01)
    if neurite_classes_present <= 1:
        flags.append(f"only {neurite_classes_present} neurite class present → partial reconstruction")

    soma_nodes = [n for n in nodes if n.type == 1]
    if soma_nodes and counts.get(4, 0) >= 10:
        sz = soma_nodes[0].z
        apical_nodes = [n for n in nodes if n.type == 4]
        apical_mean_z = statistics.mean(n.z - sz for n in apical_nodes)
        if apical_mean_z < 0:
            flags.append(
                f"apical mean z-offset = {apical_mean_z:+.1f} (expected > 0) → "
                f"z-axis may be inverted OR apical/basal swapped"
            )

    if cell_type_hint and cell_type_hint in HEALTHY:
        for label, (lo, hi) in HEALTHY[cell_type_hint].items():
            if not (lo <= fracs[label] <= hi):
                flags.append(
                    f"{label}_fraction={fracs[label]*100:.1f}% outside healthy "
                    f"{cell_type_hint} range [{lo*100:.0f}%, {hi*100:.0f}%]"
                )

    if not flags:
        verdict = "APPEARS HEALTHY — low F1 likely reflects real model failure"
        recommendation = "investigate model: add features, tune Stage 3, look at features of this file vs similar healthy ones"
    elif any("dendrite-only" in f or "axon-only" in f or "partial reconstruction" in f for f in flags):
        verdict = "BAD GT — incomplete reconstruction"
        recommendation = "DROP from benchmark (or move to excluded folder)"
    elif any("inverted" in f for f in flags):
        verdict = "SUSPICIOUS GT — coordinate frame or label convention issue"
        recommendation = "inspect visually; possibly drop or relabel"
    else:
        verdict = "BORDERLINE — GT is internally valid but atypical for cell type"
        recommendation = "inspect visually before deciding"

    return {
        "file": str(file_path),
        "n_nodes": total,
        "cell_type_hint": cell_type_hint,
        "composition": {k: f"{v*100:.1f}%" for k, v in fracs.items()},
        "flags": flags,
        "verdict": verdict,
        "recommendation": recommendation,
    }


def _diagnose_print(d: dict) -> None:
    print(f"\nFile: {Path(d['file']).name}")
    print(f"  cell_type_hint: {d.get('cell_type_hint', '?')}")
    print(f"  n_nodes: {d.get('n_nodes', '?')}")
    if "composition" in d:
        print(f"  composition: {d['composition']}")
    if d.get("flags"):
        print(f"  flags:")
        for f in d["flags"]:
            print(f"    - {f}")
    print(f"  VERDICT: {d['verdict']}")
    print(f"  recommendation: {d.get('recommendation', '')}")


# =============================================================================
# CLEAN-SOMA  (was clean_benchmark_soma.py)
# =============================================================================

_CLEAN_DTYPE = np.dtype([
    ("id", np.int64),
    ("type", np.int64),
    ("x", np.float64),
    ("y", np.float64),
    ("z", np.float64),
    ("radius", np.float64),
    ("parent", np.int64),
])


def _clean_df_to_structured(df) -> np.ndarray:
    arr = np.empty(len(df), dtype=_CLEAN_DTYPE)
    for col in ["id", "type", "x", "y", "z", "radius", "parent"]:
        arr[col] = df[col].to_numpy()
    return arr


def _clean_type_set_from_array(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted({int(v) for v in arr["type"].tolist()}))


def _clean_process_file(src: Path) -> tuple[str, dict, str | None]:
    # Lazy imports — swcstudio is only needed for clean-soma
    from swcstudio.core.swc_io import parse_swc_text_preserve_tokens
    from swcstudio.core.validation_engine import (
        consolidate_complex_somas_array, _array_to_swc_text,
    )

    text = src.read_text(encoding="utf-8", errors="ignore")
    df = parse_swc_text_preserve_tokens(text)
    arr = _clean_df_to_structured(df)
    original_soma_count = int(np.sum(arr["type"] == 1))
    original_type_set = _clean_type_set_from_array(arr)

    if original_soma_count == 0:
        return "drop_no_soma", {
            "original_soma_count": 0,
            "cleaned_soma_count": 0,
            "removed_soma_nodes": 0,
            "group_count": 0,
            "collapsed_multi_soma": False,
            "original_type_set": list(original_type_set),
            "cleaned_type_set": list(original_type_set),
        }, None

    result = consolidate_complex_somas_array(arr)
    final_arr = np.array(result.get("array", arr), copy=True)
    cleaned_soma_count = int(np.sum(final_arr["type"] == 1))
    cleaned_type_set = _clean_type_set_from_array(final_arr)
    swc_text = _array_to_swc_text(final_arr)

    meta = {
        "original_soma_count": original_soma_count,
        "cleaned_soma_count": cleaned_soma_count,
        "removed_soma_nodes": max(0, original_soma_count - cleaned_soma_count),
        "group_count": int(result.get("group_count", 0)),
        "collapsed_multi_soma": bool(result.get("changed", False)),
        "original_type_set": list(original_type_set),
        "cleaned_type_set": list(cleaned_type_set),
    }
    action = "collapsed_multi_soma" if bool(result.get("changed", False)) else "kept_single_soma"
    return action, meta, swc_text


def clean_benchmark(input_dir: Path, output_dir: Path) -> dict:
    """Consolidate multi-node somas and drop soma-less files."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_summary = json.loads((input_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    summary: dict[str, dict] = {}

    for cell_type in CELL_TYPES:
        in_ct = input_dir / cell_type
        out_ct = output_dir / cell_type
        out_swc = out_ct / "swc"
        out_ct.mkdir(parents=True, exist_ok=True)
        out_swc.mkdir(parents=True, exist_ok=True)

        manifest_in = in_ct / "manifest.csv"
        with open(manifest_in, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        kept_rows: list[dict] = []
        source_counts: Counter = Counter()
        cleaned_type_sets: Counter = Counter()
        action_counts: Counter = Counter()
        dropped_examples: list[str] = []

        for row in rows:
            src = input_dir / cell_type / "swc" / row["copied_file"]
            action, meta, swc_text = _clean_process_file(src)
            action_counts[action] += 1
            if action == "drop_no_soma":
                if len(dropped_examples) < 20:
                    dropped_examples.append(row["copied_file"])
                continue

            (out_swc / row["copied_file"]).write_text(swc_text, encoding="utf-8")
            new_row = dict(row)
            new_row["type_set"] = " ".join(str(v) for v in meta["cleaned_type_set"])
            new_row["cleanup_action"] = action
            new_row["original_soma_count"] = str(meta["original_soma_count"])
            new_row["cleaned_soma_count"] = str(meta["cleaned_soma_count"])
            new_row["removed_soma_nodes"] = str(meta["removed_soma_nodes"])
            new_row["soma_group_count"] = str(meta["group_count"])
            kept_rows.append(new_row)
            source_counts[row["source"]] += 1
            cleaned_type_sets[tuple(meta["cleaned_type_set"])] += 1

        with open(out_ct / "manifest.csv", "w", newline="", encoding="utf-8") as f:
            if kept_rows:
                fieldnames: list[str] = []
                seen: set[str] = set()
                for row in kept_rows:
                    for key in row.keys():
                        if key not in seen:
                            seen.add(key)
                            fieldnames.append(key)
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(kept_rows)
            else:
                f.write("benchmark_id,cell_type,source,copied_file,original_path,type_set\n")

        requested = int(source_summary["summary"][cell_type]["requested"])
        selected = len(kept_rows)
        summary[cell_type] = {
            "requested": requested,
            "selected": selected,
            "shortfall": max(0, requested - selected),
            "source_counts": dict(source_counts),
            "type_sets": {str(list(k)): v for k, v in sorted(cleaned_type_sets.items())},
            "cleanup": {
                "dropped_no_soma": action_counts["drop_no_soma"],
                "kept_single_soma": action_counts["kept_single_soma"],
                "collapsed_multi_soma": action_counts["collapsed_multi_soma"],
                "dropped_examples": dropped_examples,
            },
        }

    out_summary = {
        "description": (
            "Benchmark copy with soma cleanup applied: files with no soma label were removed, "
            "and connected multi-node soma groups were consolidated to one soma anchor using "
            "swcstudio.core.validation_engine.consolidate_complex_somas_array()."
        ),
        "source_benchmark": str(input_dir),
        "cell_types": list(CELL_TYPES),
        "per_type_target": int(source_summary.get("per_type_target", 1000)),
        "summary": summary,
    }
    (output_dir / "benchmark_summary.json").write_text(json.dumps(out_summary, indent=2), encoding="utf-8")
    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write("# Soma-Clean Benchmark Copy\n\n")
        f.write("Files with no soma label were removed. Connected multi-node soma groups were collapsed to one anchor soma node.\n\n")
        for ct in CELL_TYPES:
            s = summary[ct]
            f.write(f"## {ct}\n")
            f.write(f"- selected: {s['selected']}\n")
            f.write(f"- shortfall: {s['shortfall']}\n")
            f.write(f"- source counts: {s['source_counts']}\n")
            f.write(f"- cleanup: {s['cleanup']}\n\n")
    return out_summary


# =============================================================================
# FILTER  (was qc_filter_benchmark.py)
# =============================================================================

_FILTER_MEDIAN_FEATURES = (
    "n_primary_subtrees",
    "z_span",
    "max_radial_distance",
    "mean_radial_distance",
    "max_radius",
    "mean_radius",
    "std_radius",
    "thickness_ratio",
)

_FILTER_LOW_MEDIAN_RULES = {
    "z_span": 0.35,
    "max_radial_distance": 0.35,
    "mean_radial_distance": 0.40,
    "max_radius": 0.35,
    "mean_radius": 0.45,
}

_FILTER_HIGH_MEDIAN_RULES = {
    "z_span": 3.5,
    "max_radial_distance": 2.5,
    "mean_radial_distance": 2.5,
    "max_radius": 3.0,
    "std_radius": 3.0,
}


def _filter_read_manifest_rows(cell_dir: Path):
    manifest_path = cell_dir / "manifest.csv"
    if not manifest_path.exists():
        return [], {}
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_name = {row["copied_file"]: row for row in rows if row.get("copied_file")}
    return rows, by_name


def _filter_reason(code, weight, value, ref):
    return {"code": code, "weight": weight, "value": value, "reference": ref}


def _filter_median_map(rows):
    out: dict[str, dict[str, float]] = {}
    for cell_type in CELL_TYPES:
        feats: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            if row["cell_type"] != cell_type:
                continue
            row_feats = row["features"]
            for key in _FILTER_MEDIAN_FEATURES:
                feats[key].append(float(row_feats.get(key, 0.0)))
        out[cell_type] = {
            key: float(np.median(vals)) if vals else 0.0
            for key, vals in feats.items()
        }
    return out


def _filter_collect_files(input_dir: Path):
    rows: list[dict] = []
    manifests: dict[str, list] = {}
    for cell_type in CELL_TYPES:
        cell_dir = input_dir / cell_type
        swc_dir = cell_dir / "swc" if (cell_dir / "swc").is_dir() else cell_dir
        manifest_rows, manifest_by_name = _filter_read_manifest_rows(cell_dir)
        manifests[cell_type] = manifest_rows
        for swc_path in sorted(swc_dir.glob("*.swc")):
            nodes = parse_swc(swc_path)
            if not nodes:
                rows.append({
                    "cell_type": cell_type, "src": swc_path,
                    "file_name": swc_path.name,
                    "manifest_row": manifest_by_name.get(swc_path.name),
                    "features": {}, "type_counts": {}, "type_set": [],
                    "parse_error": "empty_or_invalid",
                })
                continue
            type_counts = Counter(int(n.type) for n in nodes)
            features = extract_global_features(nodes)
            rows.append({
                "cell_type": cell_type, "src": swc_path,
                "file_name": swc_path.name,
                "manifest_row": manifest_by_name.get(swc_path.name),
                "features": features,
                "type_counts": dict(sorted(type_counts.items())),
                "type_set": sorted(type_counts),
                "parse_error": None,
            })
    return rows, manifests


def _filter_score_row(row, medians):
    cell_type = str(row["cell_type"])
    features = row["features"] or {}
    type_set = set(int(v) for v in row["type_set"])
    reasons: list[dict] = []

    if row["parse_error"] is not None:
        reasons.append(_filter_reason("parse_error", 100, row["parse_error"], "valid_swc"))
        return {"status": "drop", "qc_score": 100, "reasons": reasons}

    med = medians[cell_type]
    n_primary = float(features.get("n_primary_subtrees", 0.0))
    z_span = float(features.get("z_span", 0.0))
    max_rad = float(features.get("max_radial_distance", 0.0))
    mean_rad = float(features.get("mean_radial_distance", 0.0))
    max_radius = float(features.get("max_radius", 0.0))
    std_radius = float(features.get("std_radius", 0.0))

    if 1 not in type_set:
        reasons.append(_filter_reason("no_soma_label", 100, sorted(type_set), [1]))
    if 2 not in type_set:
        reasons.append(_filter_reason("no_axon_label", 2, sorted(type_set), "contains_label_2"))
    if n_primary <= 1.0:
        reasons.append(_filter_reason("single_primary_subtree", 2, round(n_primary, 4), ">1"))
    if cell_type == "pyramidal" and 4 not in type_set:
        reasons.append(_filter_reason("pyramidal_missing_apical_label", 1, sorted(type_set), "contains_label_4"))

    for key, frac in _FILTER_LOW_MEDIAN_RULES.items():
        value = float(features.get(key, 0.0))
        ref = float(med.get(key, 0.0))
        if ref > 1e-9 and value < frac * ref:
            reasons.append(_filter_reason(f"low_{key}", 1, round(value, 4), round(ref, 4)))

    for key, mult in _FILTER_HIGH_MEDIAN_RULES.items():
        value = float(features.get(key, 0.0))
        ref = float(med.get(key, 0.0))
        if ref > 1e-9 and value > mult * ref:
            reasons.append(_filter_reason(f"high_{key}", 1, round(value, 4), round(ref, 4)))

    low_z = med.get("z_span", 0.0) > 1e-9 and z_span < 0.35 * med["z_span"]
    low_extent = med.get("max_radial_distance", 0.0) > 1e-9 and max_rad < 0.35 * med["max_radial_distance"]
    low_radius = med.get("max_radius", 0.0) > 1e-9 and max_radius < 0.35 * med["max_radius"]
    huge_interneuron = (
        cell_type == "interneuron"
        and med.get("z_span", 0.0) > 1e-9
        and med.get("max_radial_distance", 0.0) > 1e-9
        and z_span > 3.5 * med["z_span"]
        and max_rad > 2.5 * med["max_radial_distance"]
    )
    compact_single_tree = n_primary <= 1.0 and low_z and low_extent

    if compact_single_tree and 2 not in type_set:
        reasons.append(_filter_reason("compact_single_tree_without_axon", 3, round(max_rad, 4), round(med.get("max_radial_distance", 0.0), 4)))
    if cell_type == "pyramidal" and compact_single_tree and 4 not in type_set:
        reasons.append(_filter_reason("compact_pyramidal_without_apical", 3, round(z_span, 4), round(med.get("z_span", 0.0), 4)))
    if cell_type == "pyramidal" and compact_single_tree and low_radius:
        reasons.append(_filter_reason("compact_low_radius_pyramidal", 2, round(max_radius, 4), round(med.get("max_radius", 0.0), 4)))
    if huge_interneuron:
        reasons.append(_filter_reason("huge_interneuron_outlier", 3, round(max_rad, 4), round(med.get("max_radial_distance", 0.0), 4)))
    if (cell_type == "interneuron" and mean_rad > 0
        and med.get("mean_radial_distance", 0.0) > 1e-9
        and mean_rad > 2.5 * med["mean_radial_distance"]
        and std_radius > 0 and med.get("std_radius", 0.0) > 1e-9
        and std_radius > 3.0 * med["std_radius"]):
        reasons.append(_filter_reason("broad_radius_interneuron_outlier", 2, round(std_radius, 4), round(med.get("std_radius", 0.0), 4)))

    qc_score = int(sum(int(r["weight"]) for r in reasons))
    hard_drop = any(int(r["weight"]) >= 100 for r in reasons)
    drop = hard_drop or any(
        r["code"] in {
            "compact_single_tree_without_axon",
            "compact_pyramidal_without_apical",
            "huge_interneuron_outlier",
        }
        for r in reasons
    ) or qc_score >= 6

    return {"status": "drop" if drop else "keep", "qc_score": qc_score, "reasons": reasons}


def _filter_write_manifest(path: Path, rows):
    if not rows:
        path.write_text("benchmark_id,cell_type,source,copied_file,original_path,type_set\n", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def filter_benchmark(input_dir: Path, output_dir: Path, audit_only: bool = False) -> dict:
    """Conservative QC filter that drops out-of-domain files."""
    rows, manifest_rows = _filter_collect_files(input_dir)
    medians = _filter_median_map(rows)

    reason_counts: Counter = Counter()
    kept_by_type: Counter = Counter()
    dropped_by_type: Counter = Counter()
    dropped_examples: dict[str, list[str]] = defaultdict(list)
    kept_manifest_rows: dict[str, list] = defaultdict(list)
    qc_rows: list[dict] = []

    if output_dir.exists() and not audit_only:
        shutil.rmtree(output_dir)
    if not audit_only:
        output_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        score = _filter_score_row(row, medians)
        cell_type = str(row["cell_type"])
        file_name = str(row["file_name"])
        src = row["src"]
        reasons = score["reasons"]
        status = score["status"]

        for reason in reasons:
            reason_counts[f"{cell_type}:{reason['code']}"] += 1

        qc_rows.append({
            "cell_type": cell_type, "file_name": file_name, "status": status,
            "qc_score": score["qc_score"],
            "type_set": " ".join(str(v) for v in row["type_set"]),
            "reason_codes": ";".join(str(r["code"]) for r in reasons),
            "reasons": reasons, "features": row["features"], "src": str(src),
        })

        if status == "drop":
            dropped_by_type[cell_type] += 1
            if len(dropped_examples[cell_type]) < 20:
                dropped_examples[cell_type].append(file_name)
            continue

        kept_by_type[cell_type] += 1
        if not audit_only:
            out_ct = output_dir / cell_type
            out_swc = out_ct / "swc"
            out_swc.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out_swc / file_name)

            manifest_row = row["manifest_row"]
            if manifest_row is not None:
                new_row = dict(manifest_row)
            else:
                new_row = {
                    "benchmark_id": "", "cell_type": cell_type, "source": "",
                    "copied_file": file_name, "original_path": str(src),
                    "type_set": " ".join(str(v) for v in row["type_set"]),
                }
            new_row["qc_status"] = status
            new_row["qc_score"] = str(score["qc_score"])
            new_row["qc_reason_codes"] = ";".join(str(r["code"]) for r in reasons)
            kept_manifest_rows[cell_type].append(new_row)

    summary = {
        "description": (
            "Conservative morphology-QC filter for the cortical benchmark. "
            "Files are dropped only when they match strong combinations of "
            "label-ontology and morphology outlier conditions."
        ),
        "source_benchmark": str(input_dir),
        "audit_only": audit_only,
        "cell_types": list(CELL_TYPES),
        "medians": {
            cell_type: {k: round(v, 4) for k, v in vals.items()}
            for cell_type, vals in medians.items()
        },
        "summary": {
            cell_type: {
                "kept": int(kept_by_type[cell_type]),
                "dropped": int(dropped_by_type[cell_type]),
                "dropped_examples": dropped_examples[cell_type],
            }
            for cell_type in CELL_TYPES
        },
        "reason_counts": dict(sorted(reason_counts.items())),
        "dropped_files": [
            {
                "cell_type": row["cell_type"], "file_name": row["file_name"],
                "src": row["src"], "qc_score": row["qc_score"],
                "type_set": row["type_set"],
                "reason_codes": row["reason_codes"].split(";") if row["reason_codes"] else [],
                "reasons": row["reasons"],
            }
            for row in qc_rows if row["status"] == "drop"
        ],
    }

    report_dir = output_dir if not audit_only else input_dir / "qc_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "qc_filter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(report_dir / "qc_filter_report.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cell_type", "file_name", "status", "qc_score", "type_set", "reason_codes", "src"])
        for row in qc_rows:
            writer.writerow([row["cell_type"], row["file_name"], row["status"],
                             row["qc_score"], row["type_set"], row["reason_codes"], row["src"]])

    if not audit_only:
        for cell_type in CELL_TYPES:
            out_ct = output_dir / cell_type
            out_ct.mkdir(parents=True, exist_ok=True)
            _filter_write_manifest(out_ct / "manifest.csv", kept_manifest_rows[cell_type])

        with open(output_dir / "README.md", "w", encoding="utf-8") as fh:
            fh.write("# Benchmark QC Filter\n\n")
            fh.write("This benchmark copy excludes files that look structurally out-of-domain for the current cortical pyramidal/interneuron task.\n\n")
            for cell_type in CELL_TYPES:
                info = summary["summary"][cell_type]
                fh.write(f"## {cell_type}\n")
                fh.write(f"- kept: {info['kept']}\n")
                fh.write(f"- dropped: {info['dropped']}\n")
                fh.write(f"- dropped examples: {info['dropped_examples']}\n\n")

    return summary


# =============================================================================
# PRUNE  (was prune_bad_gt.py)
# =============================================================================

def _prune_verdict_class(verdict: str) -> str:
    """'BAD GT — incomplete reconstruction' -> 'BAD GT'."""
    return verdict.split(" — ")[0].strip()


def _prune_load_score_rows(scores_csv: Path) -> list[dict]:
    with open(scores_csv) as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r.get("neurite_macro_f1_stage23", 1.0)))
    return rows


def _prune_scan_for_drops(score_rows, source_root, scan_top, drop_classes, max_f1_to_drop):
    drops: list[dict] = []
    kept_diagnosed: list[dict] = []
    for r in score_rows[:scan_top]:
        abs_path = (ROOT / r["path"]).resolve()
        if not abs_path.exists():
            print(f"  [warn] missing file: {r['path']}", file=sys.stderr)
            continue
        d = diagnose(abs_path, r["cell_type"])
        d["f1_stage23"] = float(r["neurite_macro_f1_stage23"])
        d["f1_stage2"] = float(r.get("neurite_macro_f1_stage2", 0.0))
        d["cell_type"] = r["cell_type"]
        d["verdict_class"] = _prune_verdict_class(d["verdict"])
        d["file_name"] = Path(r["path"]).name
        flagged = d["verdict_class"] in drop_classes
        low_f1 = d["f1_stage23"] <= max_f1_to_drop
        if flagged and low_f1:
            drops.append(d)
        else:
            if flagged and not low_f1:
                d["kept_note"] = f"kept: f1={d['f1_stage23']:.3f} above threshold"
            kept_diagnosed.append(d)
    return drops, kept_diagnosed


def _prune_build_drop_set(drops: list[dict]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for d in drops:
        out.setdefault(d["cell_type"], set()).add(d["file_name"])
    return out


def _prune_copy_pruned_tree(source_root: Path, out_root: Path, drop_set):
    if out_root.exists():
        raise RuntimeError(
            f"output directory already exists: {out_root}\n"
            f"delete it first or pick a different --out"
        )
    out_root.mkdir(parents=True)
    stats: dict[str, dict] = {}
    for cell_type_dir in sorted(source_root.iterdir()):
        if not cell_type_dir.is_dir():
            continue
        cell_type = cell_type_dir.name
        drops_for_type = drop_set.get(cell_type, set())
        swc_src = cell_type_dir / "swc"
        if not swc_src.is_dir():
            continue
        (out_root / cell_type / "swc").mkdir(parents=True)
        kept = 0
        dropped = 0
        for swc_file in sorted(swc_src.iterdir()):
            if swc_file.suffix != ".swc":
                continue
            if swc_file.name in drops_for_type:
                dropped += 1
                continue
            shutil.copy2(swc_file, out_root / cell_type / "swc" / swc_file.name)
            kept += 1
        manifest_src = cell_type_dir / "manifest.csv"
        kept_manifest_rows = 0
        if manifest_src.exists():
            with open(manifest_src) as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                rows = [r for r in reader if r.get("copied_file") not in drops_for_type]
            manifest_out = out_root / cell_type / "manifest.csv"
            with open(manifest_out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)
            kept_manifest_rows = len(rows)
        stats[cell_type] = {
            "kept_swc": kept, "dropped_swc": dropped,
            "kept_manifest_rows": kept_manifest_rows,
            "dropped_files": sorted(drops_for_type),
        }
        print(f"  {cell_type}: kept {kept}, dropped {dropped}, manifest rows kept {kept_manifest_rows}")
    return stats


def _prune_write_report(out_root: Path, drops, kept_diagnosed):
    report_csv = out_root / "diagnosis_prune_report.csv"
    report_json = out_root / "diagnosis_prune_report.json"
    fields = ["cell_type", "file_name", "f1_stage2", "f1_stage23",
              "verdict_class", "verdict", "drop", "recommendation", "flags"]
    with open(report_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for d in drops:
            w.writerow({
                "cell_type": d["cell_type"], "file_name": d["file_name"],
                "f1_stage2": f"{d['f1_stage2']:.4f}",
                "f1_stage23": f"{d['f1_stage23']:.4f}",
                "verdict_class": d["verdict_class"], "verdict": d["verdict"],
                "drop": 1, "recommendation": d.get("recommendation", ""),
                "flags": "; ".join(d.get("flags", [])),
            })
        for d in kept_diagnosed:
            w.writerow({
                "cell_type": d["cell_type"], "file_name": d["file_name"],
                "f1_stage2": f"{d['f1_stage2']:.4f}",
                "f1_stage23": f"{d['f1_stage23']:.4f}",
                "verdict_class": d["verdict_class"], "verdict": d["verdict"],
                "drop": 0, "recommendation": d.get("recommendation", ""),
                "flags": "; ".join(d.get("flags", [])),
            })
    with open(report_json, "w") as f:
        json.dump({"dropped": drops, "kept_diagnosed": kept_diagnosed},
                  f, indent=2, default=str)


def _prune_write_summary_and_readme(out_root, source_root, tree_stats, drops, drop_classes, scan_top):
    src_summary_path = source_root / "benchmark_summary.json"
    src_summary = {}
    if src_summary_path.exists():
        with open(src_summary_path) as f:
            src_summary = json.load(f)

    drops_by_type: dict[str, list] = {}
    for d in drops:
        drops_by_type.setdefault(d["cell_type"], []).append(d)

    per_type_summary: dict[str, dict] = {}
    for cell_type, stats in tree_stats.items():
        type_drops = drops_by_type.get(cell_type, [])
        reason_counts = Counter(d["verdict_class"] for d in type_drops)
        per_type_summary[cell_type] = {
            "selected": stats["kept_swc"],
            "dropped_this_round": stats["dropped_swc"],
            "manifest_rows": stats["kept_manifest_rows"],
            "diagnosis_pruning": {
                "scan_top": scan_top,
                "drop_classes": sorted(drop_classes),
                "dropped": stats["dropped_swc"],
                "dropped_examples": stats["dropped_files"][:30],
                "drop_reason_counts": dict(reason_counts),
            },
        }

    summary = {
        "description": (
            f"Diagnosis-based pruning. Dropped verdict classes: "
            f"{sorted(drop_classes)} among the lowest-F1 {scan_top} files from per_file_scores.csv."
        ),
        "source_benchmark": str(source_root),
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "cell_types": sorted(per_type_summary.keys()),
        "summary": per_type_summary,
        "prior_history": src_summary.get("summary", {}),
    }
    with open(out_root / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Diagnosis-Pruned Benchmark",
        "",
        f"Generated {summary['created_utc']} from `{source_root.name}`.",
        "",
        f"Dropped verdicts: {', '.join(sorted(drop_classes))} (scanned lowest-F1 {scan_top} files).",
        "",
    ]
    for ct, s in per_type_summary.items():
        lines += [
            f"## {ct}",
            f"- selected: {s['selected']}",
            f"- dropped this round: {s['dropped_this_round']}",
            f"- drop reasons: {s['diagnosis_pruning']['drop_reason_counts']}",
            "",
        ]
    with open(out_root / "README.md", "w") as f:
        f.write("\n".join(lines))


def prune_benchmark(source_root: Path, scores_csv: Path, out_root: Path | None,
                    scan_top: int = 50, drop_classes: set[str] | None = None,
                    max_f1_to_drop: float = 0.85, apply: bool = False) -> dict:
    """Public API: scan worst files and prune those matching drop_classes.

    With apply=False (default), only diagnoses and reports — no files copied.
    With apply=True, requires out_root and writes the new pruned directory.
    """
    drop_classes = drop_classes or {"BAD GT"}
    rows = _prune_load_score_rows(scores_csv)
    drops, kept = _prune_scan_for_drops(rows, source_root, scan_top, drop_classes, max_f1_to_drop)
    drop_set = _prune_build_drop_set(drops)

    result = {"drops": drops, "kept_diagnosed": kept, "drop_set": drop_set}

    if not apply:
        return result

    if out_root is None:
        raise ValueError("out_root is required with apply=True")
    tree_stats = _prune_copy_pruned_tree(source_root, out_root, drop_set)
    _prune_write_report(out_root, drops, kept)
    _prune_write_summary_and_readme(out_root, source_root, tree_stats, drops, drop_classes, scan_top)
    result["tree_stats"] = tree_stats
    return result


# =============================================================================
# CLI
# =============================================================================

def _cmd_diagnose(args) -> int:
    if args.from_csv:
        rows = list(csv.DictReader(open(args.from_csv)))
        rows.sort(key=lambda r: float(r["neurite_macro_f1_stage23"]))
        summary: Counter = Counter()
        for r in rows[: args.top]:
            d = diagnose(Path(r["path"]), r["cell_type"])
            d["f1"] = float(r["neurite_macro_f1_stage23"])
            _diagnose_print(d)
            v = d["verdict"].split(" — ")[0]
            summary[v] += 1
        print(f"\n=== Summary of {args.top} worst files ===")
        for v, c in summary.most_common():
            print(f"  {c:>3}  {v}")
        return 0

    if not args.file:
        print("error: provide a file path or --from-csv", file=sys.stderr)
        return 1

    d = diagnose(args.file, args.cell_type)
    _diagnose_print(d)
    return 0


def _cmd_clean_soma(args) -> int:
    result = clean_benchmark(args.input_dir, args.output_dir)
    print(f"Soma-clean benchmark written to {args.output_dir}")
    for ct, s in result["summary"].items():
        print(f"\n{ct}:")
        print(f"  selected: {s['selected']}")
        print(f"  shortfall: {s['shortfall']}")
        print(f"  source counts: {s['source_counts']}")
        print(f"  cleanup: {s['cleanup']}")
    return 0


def _cmd_filter(args) -> int:
    summary = filter_benchmark(args.input_dir, args.output_dir, audit_only=args.audit_only)
    report_dir = args.output_dir if not args.audit_only else args.input_dir / "qc_reports"
    print(f"QC report written to {report_dir}")
    if not args.audit_only:
        print(f"Filtered benchmark written to {args.output_dir}")
    for cell_type, info in summary["summary"].items():
        print(f"\n{cell_type}:")
        print(f"  kept: {info['kept']}")
        print(f"  dropped: {info['dropped']}")
        print(f"  dropped examples: {info['dropped_examples'][:5]}")
    return 0


def _cmd_prune(args) -> int:
    source_root: Path = args.source.resolve()
    if not source_root.is_dir():
        print(f"error: source not found: {source_root}", file=sys.stderr)
        return 1

    drop_classes = set(args.drop_classes)
    print(f"Source: {source_root}")
    print(f"Scores: {args.scores}")
    print(f"Scan top: {args.scan_top}")
    print(f"Drop classes: {sorted(drop_classes)}")
    print()

    rows = _prune_load_score_rows(args.scores)
    print(f"Loaded {len(rows)} per-file scores. Diagnosing worst {args.scan_top}...")
    drops, kept = _prune_scan_for_drops(rows, source_root, args.scan_top, drop_classes, args.max_f1_to_drop)
    drop_set = _prune_build_drop_set(drops)
    total_drop = sum(len(v) for v in drop_set.values())

    print()
    print("=" * 70)
    print(f"WOULD DROP {total_drop} files:")
    for ct, names in sorted(drop_set.items()):
        print(f"  {ct}: {len(names)} files")
        for n in sorted(names)[:10]:
            info = next((d for d in drops if d["file_name"] == n and d["cell_type"] == ct), None)
            if info:
                print(f"    - {n}  f1={info['f1_stage23']:.3f}  [{info['verdict_class']}]")
            else:
                print(f"    - {n}")
        if len(names) > 10:
            print(f"    ... ({len(names) - 10} more)")
    print()
    cls_counts = Counter(d["verdict_class"] for d in drops)
    print(f"Reason counts: {dict(cls_counts)}")
    print()
    kept_classes = Counter(k["verdict_class"] for k in kept)
    if kept_classes:
        print(f"Diagnosed but kept ({len(kept)}): {dict(kept_classes)}")
    print("=" * 70)

    if not args.apply:
        print("\n[DRY RUN] — pass --apply with --out to write the pruned benchmark.")
        return 0

    if args.out is None:
        print("error: --out is required with --apply", file=sys.stderr)
        return 1
    out_root: Path = args.out.resolve()
    print(f"\nCopying pruned tree to: {out_root}")
    tree_stats = _prune_copy_pruned_tree(source_root, out_root, drop_set)
    _prune_write_report(out_root, drops, kept)
    _prune_write_summary_and_readme(out_root, source_root, tree_stats, drops, drop_classes, args.scan_top)

    print(f"\nDone. Pruned benchmark at: {out_root}")
    print(f"  - benchmark_summary.json")
    print(f"  - README.md")
    print(f"  - diagnosis_prune_report.csv / .json")
    print(f"\nNext: retrain and re-evaluate against the new directory:")
    print(f"  python -m hybrid.train_stage1 --data-dir {out_root}")
    print(f"  python -m hybrid.train_stage2 --data-dir {out_root}")
    print(f"  python -m hybrid.evaluate     --data-dir {out_root}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="hybrid.qc",
        description="Dataset quality-control toolbox (diagnose / clean-soma / filter / prune).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="<command>")

    # --- diagnose ---
    pd_ = sub.add_parser("diagnose",
                         help="Inspect a single SWC file or the worst N from per_file_scores.csv.")
    pd_.add_argument("file", nargs="?", type=Path, help="SWC file to diagnose")
    pd_.add_argument("--cell-type", choices=("pyramidal", "interneuron"))
    pd_.add_argument("--from-csv", type=Path,
                     help="Read worst N files from per_file_scores.csv")
    pd_.add_argument("--top", type=int, default=10,
                     help="With --from-csv: how many worst files to diagnose")
    pd_.set_defaults(func=_cmd_diagnose)

    # --- clean-soma ---
    pc = sub.add_parser("clean-soma",
                        help="Consolidate multi-node somas; drop files with no soma.")
    pc.add_argument("--input-dir", type=Path,
                    default=ROOT / "data" / "benchmark_pyramidal_interneuron_v1")
    pc.add_argument("--output-dir", type=Path,
                    default=ROOT / "data" / "benchmark_pyramidal_interneuron_v1_soma_clean")
    pc.set_defaults(func=_cmd_clean_soma)

    # --- filter ---
    pf = sub.add_parser("filter",
                        help="Conservative QC filter that drops out-of-domain files.")
    pf.add_argument("--input-dir", type=Path,
                    default=ROOT / "data" / "benchmark_pyramidal_interneuron_v1_soma_clean")
    pf.add_argument("--output-dir", type=Path,
                    default=ROOT / "data" / "benchmark_pyramidal_interneuron_v1_qc_filtered")
    pf.add_argument("--audit-only", action="store_true",
                    help="Write reports only, do not copy filtered files")
    pf.set_defaults(func=_cmd_filter)

    # --- prune ---
    pp = sub.add_parser("prune",
                        help="Drop files matching diagnostic verdicts; copy survivors to a new dir.")
    pp.add_argument("--source", type=Path, required=True,
                    help="Existing benchmark directory to prune from")
    pp.add_argument("--scores", type=Path,
                    default=ROOT / "hybrid" / "models" / "per_file_scores.csv",
                    help="Per-file score CSV (worst-first)")
    pp.add_argument("--out", type=Path, default=None,
                    help="Output directory (required with --apply)")
    pp.add_argument("--scan-top", type=int, default=50,
                    help="Number of worst files to diagnose")
    pp.add_argument("--drop-classes", nargs="+", default=["BAD GT"],
                    help="Verdict classes to drop (e.g. 'BAD GT' 'SUSPICIOUS GT')")
    pp.add_argument("--max-f1-to-drop", type=float, default=0.85,
                    help="Only drop a flagged file if its stage23 F1 is <= this threshold")
    pp.add_argument("--apply", action="store_true",
                    help="Actually write the new pruned directory (otherwise dry run)")
    pp.set_defaults(func=_cmd_prune)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
