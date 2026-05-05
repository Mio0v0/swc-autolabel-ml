#!/usr/bin/env python3
"""Census: pyramidal cells with no ground-truth apical.

Quantifies the false-apical rate on a known weak spot of the pipeline:
pyramidal-shaped cells whose ground-truth labeling contains no apical
dendrite (no nodes of type 4). Common cause: truncated reconstructions,
slice-prep cells, or cells that were partially traced.

For each pyramidal cell in the held-out v9 test split where GT has no
apical, the engine is run and we record whether it predicted apical
anyway. Paper-ready false-apical rate = (cells with predicted apical) /
(cells with no GT apical).

Usage::

    python -m paper.no_apical_pyramidal_census
    python -m paper.no_apical_pyramidal_census --limit 50  # quick smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = ROOT / "paper" / "results" / "snapshots"
DEFAULT_DATA_DIR = Path("D:/Desktop/SWC-Studio/data/v9_merged_dataset")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from hybrid.features import parse_swc                                          # noqa: E402
from hybrid.pipeline import run_pipeline_on_nodes                              # noqa: E402
from paper.gnn_inference import load_gnn                                       # noqa: E402


def _no_apical_pyramidal_files(
    data_dir: Path,
    eval_split_json: Path | None,
) -> list[Path]:
    """Find the held-out pyramidal test SWCs whose GT has no apical."""
    pyr_dir = data_dir / "pyramidal" / "swc"
    if not pyr_dir.is_dir():
        pyr_dir = data_dir / "pyramidal"

    if eval_split_json and eval_split_json.is_file():
        split = json.loads(eval_split_json.read_text(encoding="utf-8"))
        test_names = set(split.get("test_files", {}).get("pyramidal", []))
        candidates = [pyr_dir / name for name in sorted(test_names)]
        candidates = [c for c in candidates if c.is_file()]
    else:
        candidates = sorted(pyr_dir.glob("*.swc"))

    no_apical: list[Path] = []
    for f in candidates:
        try:
            nodes = parse_swc(f)
        except Exception:
            continue
        if not nodes:
            continue
        types = {int(n.type) for n in nodes}
        if 4 not in types:
            no_apical.append(f)
    return no_apical


def _run_engine_on(
    f: Path,
    stage1_path: Path | None,
    stage2_path: Path | None,
    gnn_state,
) -> tuple[int, set[int]]:
    """Return (n_nodes, set_of_predicted_types) for one SWC."""
    nodes = parse_swc(f)
    if not nodes:
        return 0, set()
    result = run_pipeline_on_nodes(
        nodes,
        file_path=str(f),
        stage1_model=stage1_path,
        stage2_model=stage2_path,
        gnn_state=gnn_state,
        use_subtree_stage2=True,
    )
    return len(nodes), set(int(t) for t in result.node_labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--eval-split", type=Path,
        default=SNAPSHOT_DIR / "v9_eval_split.json",
        help="Use only files in this v9 test split (default).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Cap number of cells (0 = all).")
    parser.add_argument(
        "--out-json", type=Path,
        default=SNAPSHOT_DIR / "no_apical_pyramidal_census.json",
    )
    parser.add_argument(
        "--out-text", type=Path,
        default=SNAPSHOT_DIR / "no_apical_pyramidal_census.txt",
    )
    parser.add_argument(
        "--out-csv", type=Path,
        default=SNAPSHOT_DIR / "no_apical_pyramidal_census.csv",
    )
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"ERROR: data dir not found: {args.data_dir}", file=sys.stderr)
        return 2

    print(f"Scanning {args.data_dir} for pyramidal cells with no GT apical...")
    files = _no_apical_pyramidal_files(args.data_dir, args.eval_split)
    if args.limit > 0:
        files = files[: args.limit]
    print(f"  found {len(files)} no-apical pyramidal cells")
    if not files:
        print("Nothing to do.")
        return 0

    # Models
    stage1_path = ROOT / "hybrid" / "models" / "cell_type_classifier.pkl"
    stage2_path = ROOT / "hybrid" / "models" / "branch_classifier.pkl"
    gnn_path = ROOT / "paper" / "models" / "gnn_apical_basal_v9.pt"
    if not stage1_path.is_file():
        # Fall back to bundled SWC-Studio models.
        bundled = Path("D:/Desktop/SWC-Studio/swcstudio/data/models")
        stage1_path = bundled / "cell_type_classifier.pkl"
        stage2_path = bundled / "branch_classifier.pkl"
        gnn_path = bundled / "gnn_apical_basal.pt"
        print(f"  using bundled models from {bundled}")
    print(f"  loading GNN from {gnn_path}")
    gnn_state = load_gnn(gnn_path) if gnn_path.is_file() else None
    if gnn_state is None:
        print("  warning: GNN checkpoint not loaded; engine runs Stages 1+2+3 only.")

    print(f"\nRunning engine on {len(files)} cells...")
    rows: list[dict] = []
    n_false = 0
    for idx, f in enumerate(files, 1):
        try:
            n_nodes, preds = _run_engine_on(f, stage1_path, stage2_path, gnn_state)
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "file": f.name, "n_nodes": 0, "predicted_apical": False,
                "predicted_types": "", "error": str(exc),
            })
            print(f"  [{idx:>3d}/{len(files)}] {f.name}: ERROR {exc}")
            continue

        has_apical_pred = 4 in preds
        if has_apical_pred:
            n_false += 1
        rows.append({
            "file": f.name,
            "n_nodes": n_nodes,
            "predicted_apical": has_apical_pred,
            "predicted_types": ",".join(str(t) for t in sorted(preds)),
            "error": "",
        })
        marker = "FALSE-APICAL" if has_apical_pred else "ok"
        if has_apical_pred or idx % 25 == 0 or idx == len(files):
            print(f"  [{idx:>3d}/{len(files)}] {f.name:<60s} n={n_nodes:>5d}  {marker}")

    n_total = len(files)
    rate = n_false / n_total if n_total else 0.0
    summary = {
        "n_total_no_apical_pyramidals": n_total,
        "n_false_apical_predictions": n_false,
        "false_apical_rate": rate,
        "false_apical_rate_pct": round(100.0 * rate, 2),
        "data_dir": str(args.data_dir),
        "eval_split": str(args.eval_split),
    }

    args.out_json.write_text(json.dumps(
        {"summary": summary, "rows": rows}, indent=2,
    ), encoding="utf-8")

    # Per-file CSV.
    with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # Plain-text paper-ready summary.
    lines = [
        "Pyramidal-with-no-apical census (v9 held-out test split)",
        "─" * 60,
        f"Population:  {n_total} pyramidal cells whose GT has no node of type 4",
        f"False-apical: {n_false} cells where the engine predicted apical anyway",
        f"Rate:        {100.0 * rate:.2f}%",
        "",
        f"Data dir:    {args.data_dir}",
        f"Eval split:  {args.eval_split}",
    ]
    if n_false > 0:
        lines.append("")
        lines.append("False-apical files:")
        for r in rows:
            if r.get("predicted_apical"):
                lines.append(
                    f"  {r['file']:<60s} n_nodes={r['n_nodes']:>5d}  "
                    f"predicted_types={{ {r['predicted_types']} }}"
                )
    text = "\n".join(lines)
    args.out_text.write_text(text, encoding="utf-8")
    # Re-open stdout with utf-8 so the unicode rule line doesn't crash
    # the final print on Windows cp1252 consoles. Files are already
    # written above so this never affects the snapshots.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print()
    print(text)
    print(f"\nWrote {args.out_json}")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
