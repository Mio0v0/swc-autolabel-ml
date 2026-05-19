#!/usr/bin/env python3
"""One-off corpus QC scan for v12_uncurated.

Reads every SWC file in v12_uncurated/{pyramidal,interneuron}/swc/ and
runs structural checks:

  - parseable          (>= 7 columns, valid numeric fields)
  - non_empty          (>= 10 nodes after parsing)
  - has_soma           (>= 1 node with type == 1)
  - single_root        (exactly one node with parent == -1)
  - no_orphan          (every non-root node's parent exists in the file)
  - has_neurites       (>= 1 node with type in {2, 3, 4})
  - label_set_ok       (labels are a subset of {0, 1, 2, 3, 4, 5, 6, 7})
  - radius_positive    (radii are non-NaN and >= 0)
  - coords_finite      (no NaN/Inf in x,y,z)
  - reasonable_size    (10 <= n_nodes <= 200_000)

Output:
  paper/results/corpus_qc_v12_uncurated.csv   one row per file
  paper/results/corpus_qc_v12_uncurated.json  summary aggregates

A file PASSES if every check is True. The training script will read the
CSV and only train on PASSED files. The full list is preserved so we
know which files were dropped and why.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.features import parse_swc, SWCNode                                # noqa: E402
from hybrid.swc_normalize import count_custom_types, normalize_custom_types   # noqa: E402

CORPUS = ROOT / "data" / "v12_uncurated"
OUT_CSV = ROOT / "paper" / "results" / "corpus_qc_v12_uncurated.csv"
OUT_JSON = OUT_CSV.with_suffix(".json")

# Custom (non-standard) types are NO LONGER a rejection reason; parse_swc
# rewrites them to the branch's dominant standard type. We still surface
# the raw on-disk type values for transparency.
MIN_NODES = 10
MAX_NODES = 200_000


def _is_finite(v: float) -> bool:
    return not (math.isnan(v) or math.isinf(v))


def qc_one_file(path: Path) -> dict:
    """Return a dict of QC fields for one SWC file."""
    row = {
        "path": str(path),
        "rel_path": str(path.relative_to(CORPUS)).replace("\\", "/"),
        "cell_type": path.parents[1].name,
        "source": path.name.split("__", 1)[0] if "__" in path.name else "",
        "n_bytes": path.stat().st_size if path.is_file() else 0,
        # checks (all default False; set True on success)
        "parseable": False,
        "non_empty": False,
        "has_soma": False,
        "single_root": False,
        "no_orphan": False,
        "has_neurites": False,
        "radius_positive": False,
        "coords_finite": False,
        "reasonable_size": False,
        # derived counts (AFTER normalization)
        "n_nodes": 0,
        "n_soma": 0,
        "n_axon": 0,
        "n_basal": 0,
        "n_apical": 0,
        "n_other": 0,
        "n_roots": 0,
        "n_orphan": 0,
        # informational: raw on-disk custom types (NOT a rejection reason)
        "n_custom_type_nodes_raw": 0,
        "custom_type_values_raw": "",
        # PASS flag
        "qc_pass": False,
        "qc_reason": "",
    }

    # Parse RAW, then normalize ourselves so we can report raw custom-type
    # stats and the post-normalization view in a single pass.
    try:
        raw_nodes = parse_swc(path, normalize_types=False)
    except Exception as exc:
        row["qc_reason"] = f"parse_error: {exc}"
        return row
    raw_custom = count_custom_types(raw_nodes)
    row["n_custom_type_nodes_raw"] = int(sum(raw_custom.values()))
    if raw_custom:
        row["custom_type_values_raw"] = ",".join(
            f"{t}:{c}" for t, c in sorted(raw_custom.items())
        )

    try:
        nodes, _ = normalize_custom_types(raw_nodes)
    except Exception as exc:
        row["qc_reason"] = f"normalize_error: {exc}"
        return row
    row["parseable"] = True

    if not nodes:
        row["qc_reason"] = "empty_after_parse"
        return row

    row["n_nodes"] = len(nodes)
    row["non_empty"] = len(nodes) >= MIN_NODES
    row["reasonable_size"] = MIN_NODES <= len(nodes) <= MAX_NODES

    # Per-type counts (after normalization, types are in {0,1,2,3,4})
    type_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for n in nodes:
        if n.type in type_counts:
            type_counts[n.type] += 1
    row["n_soma"] = type_counts[1]
    row["n_axon"] = type_counts[2]
    row["n_basal"] = type_counts[3]
    row["n_apical"] = type_counts[4]
    row["n_other"] = sum(1 for n in nodes if n.type not in (1, 2, 3, 4))
    row["has_soma"] = row["n_soma"] >= 1
    row["has_neurites"] = (row["n_axon"] + row["n_basal"] + row["n_apical"]) >= 1

    # Roots / orphans
    by_id = {n.id: n for n in nodes}
    n_roots = sum(1 for n in nodes if n.parent == -1)
    n_orphan = sum(1 for n in nodes if n.parent != -1 and n.parent not in by_id)
    row["n_roots"] = n_roots
    row["n_orphan"] = n_orphan
    row["single_root"] = n_roots == 1
    row["no_orphan"] = n_orphan == 0

    # Coords / radii
    row["radius_positive"] = all(_is_finite(n.radius) and n.radius >= 0 for n in nodes)
    row["coords_finite"] = all(
        _is_finite(n.x) and _is_finite(n.y) and _is_finite(n.z) for n in nodes
    )

    # PASS = every structural check is True. label_set_ok is intentionally
    # NOT in this list — non-standard types are now absorbed by
    # normalize_custom_types instead of being a rejection reason.
    checks = [
        "parseable", "non_empty", "has_soma", "single_root", "no_orphan",
        "has_neurites", "radius_positive", "coords_finite", "reasonable_size",
    ]
    failed = [c for c in checks if not row[c]]
    if not failed:
        row["qc_pass"] = True
    else:
        row["qc_reason"] = "fail: " + ",".join(failed)
    return row


def main() -> int:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for ct in ("pyramidal", "interneuron"):
        d = CORPUS / ct / "swc"
        if not d.is_dir():
            print(f"WARN: missing {d}")
            continue
        files.extend(sorted(d.glob("*.swc")))
    print(f"Found {len(files)} SWCs total in {CORPUS}")

    t0 = time.perf_counter()
    rows: list[dict] = []
    by_status_pass = {"pyramidal": 0, "interneuron": 0}
    by_status_fail = {"pyramidal": 0, "interneuron": 0}
    fail_reasons: dict[str, int] = {}

    for i, fp in enumerate(files):
        row = qc_one_file(fp)
        rows.append(row)
        ct = row["cell_type"]
        if row["qc_pass"]:
            by_status_pass[ct] = by_status_pass.get(ct, 0) + 1
        else:
            by_status_fail[ct] = by_status_fail.get(ct, 0) + 1
            fail_reasons[row["qc_reason"]] = fail_reasons.get(row["qc_reason"], 0) + 1
        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{len(files)}  (elapsed {(time.perf_counter()-t0)/60:.1f} min)")

    # Write CSV
    if rows:
        keys = list(rows[0].keys())
        with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # Aggregate summary
    by_source = {}
    for r in rows:
        src = r["source"] or "unknown"
        by_source.setdefault(src, {"pass": 0, "fail": 0})
        by_source[src]["pass" if r["qc_pass"] else "fail"] += 1

    summary = {
        "corpus": str(CORPUS),
        "n_total": len(rows),
        "n_pass": sum(1 for r in rows if r["qc_pass"]),
        "n_fail": sum(1 for r in rows if not r["qc_pass"]),
        "pass_by_cell_type": by_status_pass,
        "fail_by_cell_type": by_status_fail,
        "by_source": by_source,
        "fail_reason_counts": dict(sorted(fail_reasons.items(), key=lambda kv: -kv[1])),
        "elapsed_min": (time.perf_counter() - t0) / 60.0,
        "csv": str(OUT_CSV),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"=== QC scan complete ({summary['elapsed_min']:.1f} min) ===")
    print(f"  Total       : {summary['n_total']}")
    print(f"  Passed      : {summary['n_pass']}  ({summary['n_pass'] / max(1, summary['n_total']) * 100:.1f}%)")
    print(f"  Failed      : {summary['n_fail']}")
    print(f"  Pass by cell-type    : {by_status_pass}")
    print(f"  Fail by cell-type    : {by_status_fail}")
    print(f"  By source            : {by_source}")
    print(f"  Top fail reasons:")
    for reason, n in list(summary["fail_reason_counts"].items())[:8]:
        print(f"    {n:>6}  {reason}")
    print()
    print(f"CSV  -> {OUT_CSV}")
    print(f"JSON -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
