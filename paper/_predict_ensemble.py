#!/usr/bin/env python3
"""Ensemble inference + disagreement-based flagging.

For each input SWC, runs N independently-trained models through the full
hybrid pipeline (Stage 1+2+3+refinement), then:

  1. Aggregates per-node predictions via MAJORITY VOTE to get the
     ensemble's final label.
  2. Computes per-node DISAGREEMENT (fraction of models that disagree
     with the majority label) — this is the *post-refinement* confidence
     signal we couldn't get from single-model softmax.
  3. Per-branch aggregate: mean disagreement across the branch's nodes.
     Branches with high mean disagreement are flagged for review.
  4. Per-cell aggregate: fraction of nodes with non-zero disagreement.
     Cells with broadly high disagreement are flagged as a whole.

Output per file:
    <stem>_labeled.swc        — ensemble's majority-vote labels
    <stem>_ensemble_report.json — full report including disagreement +
                                  per-branch flags + per-cell flag

Usage:
    python -m paper._predict_ensemble \\
        --input my_cell.swc \\
        --model-dirs paper/models/v12_weighted_gentle,\\
                     paper/models/v12_gentle_seed123,\\
                     paper/models/v12_gentle_seed456

    # Batch mode on a folder
    python -m paper._predict_ensemble --input my_folder/ --batch --model-dirs ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.features import parse_swc, SWCNode                              # noqa: E402
from hybrid.pipeline import run_pipeline_on_nodes                           # noqa: E402
from hybrid.qc_input import QCGate                                          # noqa: E402
from hybrid.confidence import _segment_branches, LABEL_NAMES                # noqa: E402
from paper.gnn_inference import load_gnn                                    # noqa: E402


# ---------------------------------------------------------------------------
# Ensemble loader
# ---------------------------------------------------------------------------
def load_ensemble(model_dirs: list[Path]) -> list[dict]:
    """Load N model sets. Each entry: {s1, s2, gnn_state, tag, qc_gate?}.

    The first model's QC gate is used for input QC (they're effectively
    identical since the OOD detector is fit on overlapping training data;
    using the first is a defensible default).
    """
    models = []
    for d in model_dirs:
        s1 = d / "cell_type_classifier.pkl"
        s2 = d / "branch_classifier.pkl"
        gnn = d / "gnn_apical_basal.pt"
        qc_gate_path = d / "qc_gate.pkl"
        if not (s1.is_file() and s2.is_file() and gnn.is_file()):
            print(f"  WARN: missing model files in {d}, skipping")
            continue
        print(f"  Loading {d.name}: GNN -> ", end="", flush=True)
        gnn_state = load_gnn(gnn)
        print(f"{gnn_state.device}")
        qc_gate = QCGate.load(qc_gate_path) if qc_gate_path.is_file() else QCGate()
        models.append({
            "tag":       d.name,
            "s1":        s1,
            "s2":        s2,
            "gnn_state": gnn_state,
            "qc_gate":   qc_gate,
        })
    if not models:
        raise RuntimeError("No usable models found in --model-dirs.")
    print(f"  Loaded {len(models)} models for the ensemble.")
    return models


# ---------------------------------------------------------------------------
# Per-cell ensemble inference
# ---------------------------------------------------------------------------
def predict_with_ensemble(
    nodes: list[SWCNode], models: list[dict],
) -> dict:
    """Run each model's pipeline and aggregate. Returns:
        {
            "ensemble_labels": list[int]  # majority vote per node
            "per_node_disagreement": list[float]  # fraction of models != majority
            "per_model_labels": list[list[int]]    # one row per model
            "per_model_cell_type": list[str]
        }
    """
    n_models = len(models)
    n_nodes = len(nodes)
    per_model_labels: list[list[int]] = []
    per_model_cell_type: list[str] = []

    for m in models:
        pr = run_pipeline_on_nodes(
            nodes, file_path="",
            stage1_model=m["s1"], stage2_model=m["s2"],
            gnn_state=m["gnn_state"], use_subtree_stage2=True,
        )
        per_model_labels.append(list(pr.node_labels))
        per_model_cell_type.append(pr.stage1.cell_type)

    # Per-node majority vote + disagreement
    ensemble_labels: list[int] = []
    disagreement: list[float] = []
    for i in range(n_nodes):
        votes = [per_model_labels[k][i] for k in range(n_models)]
        most_common, count = Counter(votes).most_common(1)[0]
        ensemble_labels.append(int(most_common))
        disagreement.append(1.0 - count / n_models)
    return {
        "ensemble_labels":       ensemble_labels,
        "per_node_disagreement": disagreement,
        "per_model_labels":      per_model_labels,
        "per_model_cell_type":   per_model_cell_type,
    }


# ---------------------------------------------------------------------------
# Two-level flag using ensemble disagreement
# ---------------------------------------------------------------------------
def build_flag_report(
    nodes: list[SWCNode],
    labels: list[int],
    disagreement: list[float],
    per_model_cell_type: list[str],
    branch_disagreement_threshold: float = 0.25,
    cell_disagreement_fraction_threshold: float = 0.15,
) -> dict:
    """Use ensemble disagreement to flag branches and cells.

    Per-branch flag: branch flagged if its MEAN disagreement >= threshold.
    Per-cell flag: cell flagged if the FRACTION of nodes with
      disagreement > 0 (any disagreement) exceeds threshold, OR if the
      ensemble disagrees on the predicted cell type.
    """
    branches = _segment_branches(nodes, labels)
    n_total = len(nodes)
    n_models = len(per_model_cell_type)

    branch_records = []
    n_flagged_branches = 0
    for branch_idx, idxs in enumerate(branches):
        if not idxs:
            continue
        b_disag = [disagreement[i] for i in idxs]
        majority_label = Counter([labels[i] for i in idxs]).most_common(1)[0][0]
        mean_d = sum(b_disag) / len(b_disag)
        max_d  = max(b_disag)
        n_disagreeing = sum(1 for d in b_disag if d > 0)
        is_flagged = mean_d >= branch_disagreement_threshold
        if is_flagged:
            n_flagged_branches += 1
        branch_records.append({
            "branch_id":          branch_idx,
            "n_nodes":            len(idxs),
            "predicted_label":    int(majority_label),
            "predicted_name":     LABEL_NAMES.get(int(majority_label), str(majority_label)),
            "mean_disagreement":  round(mean_d, 4),
            "max_disagreement":   round(max_d, 4),
            "n_nodes_with_disagreement": n_disagreeing,
            "flag":               bool(is_flagged),
            "node_indices":       idxs,
        })

    # Cell-level
    n_nodes_disagreeing = sum(1 for d in disagreement if d > 0)
    frac_nodes_disagreeing = n_nodes_disagreeing / max(1, n_total)
    mean_node_disagreement = sum(disagreement) / max(1, n_total)
    cell_type_consensus = Counter(per_model_cell_type).most_common(1)
    cell_type_majority, ct_count = cell_type_consensus[0]
    cell_type_disagreement = 1.0 - ct_count / max(1, n_models)
    cell_reasons = []
    if frac_nodes_disagreeing > cell_disagreement_fraction_threshold:
        cell_reasons.append(f"frac_disagreeing>{cell_disagreement_fraction_threshold:.2f}")
    if cell_type_disagreement > 0:
        cell_reasons.append(f"cell_type_disagreement={cell_type_disagreement:.2f}")
    cell_flag = bool(cell_reasons)

    # Per-class breakdown
    per_class_disag: dict[str, float] = {}
    per_class_count: dict[str, int] = {}
    for c, name in LABEL_NAMES.items():
        idxs_c = [i for i, lab in enumerate(labels) if lab == c]
        per_class_count[name] = len(idxs_c)
        if idxs_c:
            per_class_disag[name] = float(np.mean([disagreement[i] for i in idxs_c]))

    return {
        "cell": {
            "n_nodes":                       n_total,
            "n_models_in_ensemble":          n_models,
            "mean_node_disagreement":        round(mean_node_disagreement, 4),
            "fraction_nodes_disagreeing":    round(frac_nodes_disagreeing, 4),
            "cell_type_majority":            cell_type_majority,
            "cell_type_disagreement":        round(cell_type_disagreement, 4),
            "per_class_mean_disagreement":   {k: round(v, 4) for k, v in per_class_disag.items()},
            "per_class_node_count":          per_class_count,
            "flag":                          cell_flag,
            "flag_reasons":                  cell_reasons,
        },
        "branches_flagged": [
            {k: v for k, v in b.items() if k != "node_indices"}
            for b in branch_records if b["flag"]
        ],
        "n_branches_total":   len(branch_records),
        "n_branches_flagged": n_flagged_branches,
        "thresholds": {
            "branch_disagreement_threshold":     branch_disagreement_threshold,
            "cell_disagreement_fraction_threshold": cell_disagreement_fraction_threshold,
        },
    }


# ---------------------------------------------------------------------------
# Write labeled SWC
# ---------------------------------------------------------------------------
def write_labeled_swc(src: Path, dst: Path, new_labels: list[int]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Re-labeled by SWC-Studio ensemble auto-labeler")
    lines.append(f"# Source: {src.name}  Ensemble size: --N--  (see report json)")
    data_idx = 0
    with src.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            raw = line.rstrip("\n").rstrip("\r")
            s = raw.strip()
            if not s or s.startswith("#"):
                lines.append(raw)
                continue
            parts = raw.split()
            if len(parts) < 7:
                lines.append(raw)
                continue
            if data_idx < len(new_labels):
                parts[1] = str(int(new_labels[data_idx]))
                lines.append(" ".join(parts))
                data_idx += 1
            else:
                lines.append(raw)
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def process_one(
    path: Path, out_dir: Path, models: list[dict],
    branch_thr: float, cell_thr: float,
    write_swc: bool = True,
) -> dict:
    t0 = time.perf_counter()

    # Input QC using first model's gate
    qc = models[0]["qc_gate"].evaluate(path)
    report = {
        "input": str(path),
        "qc": {"passed": qc.passed, "reasons": qc.reasons, "n_nodes": qc.n_nodes},
        "ensemble": None,
        "flag": None,
        "elapsed_sec": None,
    }
    if not qc.passed:
        report["elapsed_sec"] = round(time.perf_counter() - t0, 3)
        return report

    # Ensemble inference
    nodes = parse_swc(path)
    ens = predict_with_ensemble(nodes, models)
    flag_report = build_flag_report(
        nodes, ens["ensemble_labels"], ens["per_node_disagreement"],
        ens["per_model_cell_type"], branch_thr, cell_thr,
    )
    report["ensemble"] = {
        "n_models":          len(models),
        "model_tags":        [m["tag"] for m in models],
        "cell_type_votes":   Counter(ens["per_model_cell_type"]),
    }
    report["flag"] = flag_report

    if write_swc:
        out_swc = out_dir / f"{path.stem}_labeled_ensemble.swc"
        write_labeled_swc(path, out_swc, ens["ensemble_labels"])
        report["ensemble"]["output_swc"] = str(out_swc)

    report["elapsed_sec"] = round(time.perf_counter() - t0, 3)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path,
                    help="SWC file, or folder if --batch")
    ap.add_argument("--batch", action="store_true",
                    help="Treat --input as a folder of *.swc files")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Where to write labeled SWCs + reports "
                         "(default: <input>/swc_studio_ensemble_predictions/)")
    ap.add_argument("--model-dirs", required=True, type=str,
                    help="Comma-separated list of model dirs "
                         "(e.g. paper/models/v12_weighted_gentle,paper/models/v12_gentle_seed123,...)")
    ap.add_argument("--branch-flag-threshold", type=float, default=0.25,
                    help="Branch flagged if mean disagreement >= this (default 0.25)")
    ap.add_argument("--cell-flag-threshold", type=float, default=0.15,
                    help="Cell flagged if fraction-of-disagreeing-nodes > this (default 0.15)")
    ap.add_argument("--no-write-swc", action="store_true")
    args = ap.parse_args()

    model_dirs = [Path(p.strip()).resolve() for p in args.model_dirs.split(",") if p.strip()]
    print(f"Loading ensemble of {len(model_dirs)} models:")
    for d in model_dirs:
        print(f"  - {d}")
    print()
    models = load_ensemble(model_dirs)

    # Resolve inputs
    if args.batch:
        if not args.input.is_dir():
            print(f"ERROR: --batch but {args.input} is not a directory", file=sys.stderr)
            return 2
        inputs = sorted(args.input.glob("*.swc"))
    else:
        if not args.input.is_file():
            print(f"ERROR: {args.input} is not a file", file=sys.stderr)
            return 2
        inputs = [args.input]
    if not inputs:
        print("ERROR: no input files", file=sys.stderr)
        return 2

    out_dir = args.output_dir or (
        (args.input if args.input.is_dir() else args.input.parent)
        / "swc_studio_ensemble_predictions"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    n_pass = n_reject = n_cell_flag = n_branch_flag = 0
    t0 = time.perf_counter()
    print(f"\nProcessing {len(inputs)} file(s) with {len(models)}-model ensemble...")
    for i, inp in enumerate(inputs, 1):
        rep = process_one(
            inp, out_dir, models,
            args.branch_flag_threshold, args.cell_flag_threshold,
            write_swc=not args.no_write_swc,
        )
        rpath = out_dir / f"{inp.stem}_ensemble_report.json"
        rep_serial = json.loads(json.dumps(rep, default=lambda o: dict(o) if hasattr(o, '__dict__') else str(o)))
        rpath.write_text(json.dumps(rep_serial, indent=2, default=str), encoding="utf-8")
        if rep["qc"]["passed"]:
            n_pass += 1
            cf = rep["flag"]["cell"]["flag"]
            nbf = rep["flag"]["n_branches_flagged"]
            if cf: n_cell_flag += 1
            if nbf > 0: n_branch_flag += 1
            print(f"  [{i:>4}/{len(inputs)}] {inp.name:<60s} PASS  "
                  f"cell_flag={cf}  branches_flagged={nbf}  "
                  f"({rep['elapsed_sec']:.1f}s)")
        else:
            n_reject += 1
            print(f"  [{i:>4}/{len(inputs)}] {inp.name:<60s} REJECT  {rep['qc']['reasons']}")

    total = time.perf_counter() - t0
    print(f"\n=== Summary ({total:.1f}s for {len(inputs)} files) ===")
    print(f"  QC passed       : {n_pass} ({n_pass/len(inputs)*100:.0f}%)")
    print(f"  QC rejected     : {n_reject}")
    if n_pass:
        print(f"  Cell flagged    : {n_cell_flag} ({n_cell_flag/n_pass*100:.0f}% of passed)")
        print(f"  Has branch flag : {n_branch_flag} ({n_branch_flag/n_pass*100:.0f}% of passed)")
    print(f"\nReports + labeled SWCs -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
