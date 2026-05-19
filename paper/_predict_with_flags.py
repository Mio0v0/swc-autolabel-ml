#!/usr/bin/env python3
"""End-to-end three-stage auto-labeling CLI: QC + label + two-level flag.

Stages:
  1. Input QC  — reject unlabelable files (structural + OOD).
                 If REJECTED, no labels are produced.
  2. Auto-label — run the hybrid pipeline (Stage 1+2+3+refinement).
  3. Two-level flag — per-branch + per-cell confidence flags. Calibrated
                 thresholds loaded from a JSON config.

Outputs (per input file):
    <stem>_labeled.swc        — the labeled SWC (only if QC passed)
    <stem>_report.json        — full QC + label + flag report

Usage:
    # Single file
    python -m paper._predict_with_flags --input my_cell.swc

    # Batch on a folder
    python -m paper._predict_with_flags --input my_folder/ --batch

    # Use custom v12-trained models + calibrated flag config
    python -m paper._predict_with_flags \\
        --input cell.swc \\
        --stage1-model paper/models/v12/cell_type_classifier.pkl \\
        --stage2-model paper/models/v12/branch_classifier.pkl \\
        --gnn-model    paper/models/v12/gnn_apical_basal.pt \\
        --qc-gate      paper/models/v12/qc_gate.pkl \\
        --flag-config  paper/models/v12/flag_config.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.features import parse_swc                                       # noqa: E402
from hybrid.pipeline import run_pipeline_on_nodes                           # noqa: E402
from hybrid.qc_input import QCGate, QCResult                                # noqa: E402
from hybrid.confidence import (                                             # noqa: E402
    ConfidenceConfig, summarize_confidence, apply_two_level_flag,
    report_to_dict, LABEL_NAMES,
)


# =============================================================================
# IO
# =============================================================================
def _read_swc_lines(path: Path) -> list[str]:
    """Return all (non-comment) data lines from an SWC file in order."""
    out = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            s = line.rstrip("\n").rstrip("\r")
            if not s.strip() or s.strip().startswith("#"):
                out.append(s)
            else:
                out.append(s)
    return out


def write_labeled_swc(
    src_path: Path,
    dst_path: Path,
    new_labels: list[int],
) -> None:
    """Write a copy of `src_path` to `dst_path` with the type column (col 2)
    replaced by `new_labels` (in the order rows appear in the original file).

    Preserves all comments / blank lines verbatim. Adds a single provenance
    comment at the top noting the change.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    lines = _read_swc_lines(src_path)
    new_lines: list[str] = []
    new_lines.append(f"# Re-labeled by SWC-Studio auto-labeler (paper/_predict_with_flags.py)")
    new_lines.append(f"# Source: {src_path.name}")
    data_idx = 0
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            new_lines.append(line)
            continue
        parts = line.split()
        if len(parts) < 7:
            new_lines.append(line)
            continue
        if data_idx < len(new_labels):
            parts[1] = str(int(new_labels[data_idx]))
            # Reconstruct line preserving leading whitespace from the split
            new_lines.append(" ".join(parts))
            data_idx += 1
        else:
            new_lines.append(line)
    dst_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


# =============================================================================
# Process one file end-to-end
# =============================================================================
def process_one(
    input_path: Path,
    output_dir: Path,
    qc_gate: QCGate | None,
    flag_config: ConfidenceConfig,
    stage1_model: Path | None,
    stage2_model: Path | None,
    gnn_state: object | None = None,
    write_labeled: bool = True,
) -> dict:
    """Run QC → label → flag for one file. Returns the report dict."""
    t0 = time.perf_counter()
    stem = input_path.stem

    # STAGE 1: QC
    if qc_gate is not None:
        qc = qc_gate.evaluate(input_path)
    else:
        # If no gate provided, do structural-only via QCGate()
        qc = QCGate().evaluate(input_path)
    qc_pass = qc.passed

    report = {
        "input": str(input_path),
        "qc": {
            "passed":      qc.passed,
            "reasons":     qc.reasons,
            "n_nodes":     qc.n_nodes,
            "n_soma":      qc.n_soma,
            "n_roots":     qc.n_roots,
            "n_orphan":    qc.n_orphan,
            "ood_distance":   qc.ood_distance,
            "ood_threshold":  qc.ood_threshold,
        },
        "labeling": None,
        "flag":     None,
        "elapsed_sec": None,
    }

    if not qc_pass:
        report["elapsed_sec"] = round(time.perf_counter() - t0, 3)
        return report

    # STAGE 2: Auto-label
    nodes = parse_swc(input_path)
    pr = run_pipeline_on_nodes(
        nodes,
        file_path=str(input_path),
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        gnn_state=gnn_state,
        use_subtree_stage2=True,
    )
    report["labeling"] = {
        "cell_type":  pr.stage1.cell_type,
        "stage1_confidence": float(pr.stage1.confidence),
        "label_distribution": {
            LABEL_NAMES.get(lab, str(lab)): int(sum(1 for x in pr.node_labels if x == lab))
            for lab in sorted(set(pr.node_labels))
        },
    }

    # Write labeled SWC
    if write_labeled:
        out_swc = output_dir / f"{stem}_labeled.swc"
        write_labeled_swc(input_path, out_swc, pr.node_labels)
        report["labeling"]["output_swc"] = str(out_swc)

    # STAGE 3: Two-level confidence flag
    branches, cell = summarize_confidence(
        nodes,
        labels=pr.node_labels,
        confidences=pr.node_confidences,
        stage1_cell_type=pr.stage1.cell_type,
        stage1_confidence=float(pr.stage1.confidence),
        cfg=flag_config,
    )
    branches, cell = apply_two_level_flag(branches, cell, flag_config)
    report["flag"] = report_to_dict(branches, cell, flag_config)

    report["elapsed_sec"] = round(time.perf_counter() - t0, 3)
    return report


# =============================================================================
# Driver
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path,
                    help="SWC file, or folder if --batch")
    ap.add_argument("--batch", action="store_true",
                    help="Treat --input as a folder of *.swc files")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Where to write labeled SWCs + reports "
                         "(default: <input_dir>/swc_studio_predictions/)")
    ap.add_argument("--stage1-model", type=Path, default=None,
                    help="Path to Stage 1 pkl. Default: production model.")
    ap.add_argument("--stage2-model", type=Path, default=None,
                    help="Path to Stage 2 pkl. Default: production model.")
    ap.add_argument("--gnn-model", type=Path, default=None,
                    help="Path to GNN .pt. Default: production v11_final.")
    ap.add_argument("--qc-gate", type=Path, default=None,
                    help="Path to QCGate pkl. If unset, only structural QC runs (no OOD).")
    ap.add_argument("--flag-config", type=Path, default=None,
                    help="Path to ConfidenceConfig JSON. If unset, uses defaults.")
    ap.add_argument("--no-write-swc", action="store_true",
                    help="Don't write the labeled .swc, only the report.")
    args = ap.parse_args()

    # Resolve inputs
    if args.batch:
        if not args.input.is_dir():
            print(f"ERROR: --batch given but {args.input} is not a directory", file=sys.stderr)
            return 2
        inputs = sorted(args.input.glob("*.swc"))
    else:
        if not args.input.is_file():
            print(f"ERROR: {args.input} is not a file", file=sys.stderr)
            return 2
        inputs = [args.input]
    if not inputs:
        print("ERROR: no input files found", file=sys.stderr)
        return 2

    # Output dir
    if args.output_dir is None:
        args.output_dir = (
            args.input if args.input.is_dir()
            else args.input.parent
        ) / "swc_studio_predictions"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load QC gate
    qc_gate = None
    if args.qc_gate:
        if not args.qc_gate.is_file():
            print(f"ERROR: --qc-gate {args.qc_gate} not found", file=sys.stderr)
            return 2
        qc_gate = QCGate.load(args.qc_gate)
        print(f"QC gate loaded: structural + OOD (threshold={qc_gate.ood_detector.threshold:.3f})")
    else:
        print("QC gate: structural-only (no OOD detector loaded)")

    # Load flag config
    if args.flag_config:
        flag_config = ConfidenceConfig.load(args.flag_config)
        print(f"Flag config loaded: target precision = {flag_config.target_precision}, "
              f"calibration_id={flag_config.calibration_id}")
    else:
        flag_config = ConfidenceConfig.default()
        print("Flag config: defaults (uncalibrated)")

    # Load GNN once (saves time across the batch)
    gnn_state = None
    if args.gnn_model and args.gnn_model.is_file():
        from paper.gnn_inference import load_gnn
        gnn_state = load_gnn(args.gnn_model)
        print(f"GNN loaded from {args.gnn_model.name}")
    elif args.gnn_model:
        print(f"WARNING: --gnn-model path not found, GNN inference disabled")

    # Run
    print()
    print(f"Processing {len(inputs)} file(s) -> {args.output_dir}")
    overall_t0 = time.perf_counter()
    n_pass = n_reject = n_cell_flag = n_branch_flag = 0
    for i, inp in enumerate(inputs, 1):
        rep = process_one(
            inp,
            args.output_dir,
            qc_gate=qc_gate,
            flag_config=flag_config,
            stage1_model=args.stage1_model,
            stage2_model=args.stage2_model,
            gnn_state=gnn_state,
            write_labeled=not args.no_write_swc,
        )
        rpt = args.output_dir / f"{inp.stem}_report.json"
        rpt.write_text(json.dumps(rep, indent=2), encoding="utf-8")

        # Console summary
        if rep["qc"]["passed"]:
            n_pass += 1
            cf = rep["flag"]["cell"]["flag"]
            nbf = rep["flag"]["n_branches_flagged"]
            if cf:
                n_cell_flag += 1
            if nbf > 0:
                n_branch_flag += 1
            print(f"  [{i:>4}/{len(inputs)}] {inp.name:<60s} "
                  f"PASS  {rep['labeling']['cell_type']:<11s} "
                  f"cell_flag={cf}  branches_flagged={nbf}  "
                  f"({rep['elapsed_sec']:.1f}s)")
        else:
            n_reject += 1
            print(f"  [{i:>4}/{len(inputs)}] {inp.name:<60s} "
                  f"REJECT  reasons={rep['qc']['reasons']}  "
                  f"({rep['elapsed_sec']:.1f}s)")

    total = (time.perf_counter() - overall_t0)
    print()
    print(f"=== SUMMARY ({total:.1f} s for {len(inputs)} file(s)) ===")
    print(f"  QC passed       : {n_pass} ({n_pass/len(inputs)*100:.0f}%)")
    print(f"  QC rejected     : {n_reject} ({n_reject/len(inputs)*100:.0f}%)")
    if n_pass:
        print(f"  Cell flagged    : {n_cell_flag} ({n_cell_flag/n_pass*100:.0f}% of passed)")
        print(f"  Has branch flag : {n_branch_flag} ({n_branch_flag/n_pass*100:.0f}% of passed)")
    print()
    print(f"Per-file reports written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
