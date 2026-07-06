#!/usr/bin/env python3
"""Score one model split into flag-model labels and deployment-safe features."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.evaluate import per_cell_neurite_f1  # noqa: E402
from hybrid.features import parse_swc  # noqa: E402
from hybrid.pipeline import run_pipeline_on_nodes  # noqa: E402
from paper._build_flag_confidence_features import (  # noqa: E402
    _branch_summary,
    _confidence_summary,
    _predicted_geometry,
    _round_row,
)
from paper._score_flag_dataset_branch3 import _branch3_disagreement_features  # noqa: E402
from paper._score_heldout_f1 import _per_class_f1, _structural_features  # noqa: E402
from paper.gnn_branch3_inference import load_branch3  # noqa: E402
from paper.gnn_inference import load_gnn  # noqa: E402

DATA_DIR = ROOT / "data" / "v12_uncurated"


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _load_model_dir(model_dir: Path) -> dict:
    s1 = model_dir / "cell_type_classifier.pkl"
    s2 = model_dir / "branch_classifier.pkl"
    gnn = model_dir / "gnn_apical_basal.pt"
    for p in (s1, s2, gnn):
        if not p.is_file():
            raise SystemExit(f"MISSING: {p}")
    branch3 = model_dir / "gnn_branch3_rescue.pt"
    return {
        "stage1_model": s1,
        "stage2_model": s2,
        "gnn_state": load_gnn(gnn),
        "branch3_state": load_branch3(branch3) if branch3.is_file() else None,
        "branch3_path": branch3 if branch3.is_file() else None,
    }


def _split_paths(model_dir: Path, split_name: str) -> list[tuple[str, Path]]:
    split_path = model_dir / "train_test_split.json"
    if not split_path.is_file():
        raise SystemExit(f"MISSING: {split_path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split_name not in {"train", "test", "all"}:
        raise SystemExit(f"Unknown split {split_name!r}")
    blocks = ["train", "test"] if split_name == "all" else [split_name]
    out: list[tuple[str, Path]] = []
    for block in blocks:
        for ct in ("pyramidal", "interneuron"):
            for name in split[block].get(ct, []):
                p = DATA_DIR / ct / "swc" / name
                if p.is_file():
                    out.append((ct, p))
    return sorted(out, key=lambda item: item[1].name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "test", "all"], default="test")
    ap.add_argument("--out-labels", type=Path, default=None)
    ap.add_argument("--out-features", type=Path, default=None)
    ap.add_argument("--out-summary", type=Path, default=None)
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--progress-every", type=int, default=100)
    args = ap.parse_args()

    model_dir = args.model_dir.resolve()
    model = _load_model_dir(model_dir)
    files = _split_paths(model_dir, args.split)
    if args.max_cells is not None:
        files = files[: args.max_cells]

    stem = f"{model_dir.name}_{args.split}_flag"
    out_labels = args.out_labels or (ROOT / "paper" / "results" / f"{stem}_labels.csv")
    out_features = args.out_features or (ROOT / "paper" / "results" / f"{stem}_features.csv")
    out_summary = args.out_summary or (ROOT / "paper" / "results" / f"{stem}_summary.json")

    print(f"=== Score {args.split} split for flag model ===")
    print(f"  model:   {_display_path(model_dir)}")
    print(f"  branch3: {_display_path(model['branch3_path']) if model['branch3_path'] else 'disabled'}")
    print(f"  cells:   {len(files)}")
    print(f"  labels:  {_display_path(out_labels)}")
    print(f"  feats:   {_display_path(out_features)}")

    label_rows: list[dict] = []
    feature_rows: list[dict] = []
    n_failed = 0
    t0 = time.perf_counter()

    for i, (ct_gt, path) in enumerate(files, 1):
        cell_t0 = time.perf_counter()
        try:
            nodes = parse_swc(path)
            if not nodes:
                n_failed += 1
                continue
            gt = [n.type for n in nodes]
            pr = run_pipeline_on_nodes(
                nodes,
                file_path="",
                stage1_model=model["stage1_model"],
                stage2_model=model["stage2_model"],
                gnn_state=model["gnn_state"],
                branch3_state=model["branch3_state"],
                use_subtree_stage2=True,
            )
            labels = list(pr.node_labels)
            confs = [float(c) for c in pr.node_confidences]

            base_labels: list[int] | None = None
            base_confs: list[float] | None = None
            if model["branch3_state"] is not None and pr.stage1.cell_type == "pyramidal":
                base_pr = run_pipeline_on_nodes(
                    nodes,
                    file_path="",
                    stage1_model=model["stage1_model"],
                    stage2_model=model["stage2_model"],
                    gnn_state=model["gnn_state"],
                    branch3_state=None,
                    use_subtree_stage2=True,
                )
                base_labels = list(base_pr.node_labels)
                base_confs = [float(c) for c in base_pr.node_confidences]

            pc_f1 = per_cell_neurite_f1(gt, labels, ct_gt)
            per_class = _per_class_f1(gt, labels, ct_gt)
            n_correct = sum(1 for g, q in zip(gt, labels) if g == q)
            pc_acc = n_correct / max(1, len(gt))
            structural = _structural_features(nodes)
            pred_counter = Counter(labels)
            elapsed_s = time.perf_counter() - cell_t0

            label_rows.append({
                "file": path.name,
                "cell_type_gt": ct_gt,
                "n_nodes": len(gt),
                "seed_used": model_dir.name,
                "held_out_F1": f"{pc_f1:.4f}",
                "acc": f"{pc_acc:.4f}",
                "axon_F1": f"{per_class['axon_f1']:.4f}" if not np.isnan(per_class["axon_f1"]) else "",
                "basal_F1": f"{per_class['basal_f1']:.4f}" if not np.isnan(per_class["basal_f1"]) else "",
                "apical_F1": f"{per_class['apical_f1']:.4f}" if not np.isnan(per_class["apical_f1"]) else "",
                "stage1_pred": pr.stage1.cell_type,
                "stage1_conf": f"{float(pr.stage1.confidence):.4f}",
                "stage1_correct": str(pr.stage1.cell_type == ct_gt),
                "n_components": int(structural.get("n_components", 1)),
                "soma_z": f"{structural.get('soma_z', 0):.2f}" if not np.isnan(structural.get("soma_z", float("nan"))) else "",
                "apical_mean_z": f"{structural.get('apical_mean_z', 0):.2f}" if not np.isnan(structural.get("apical_mean_z", float("nan"))) else "",
                "apical_above_soma": f"{structural.get('apical_above_soma', 0):.2f}" if not np.isnan(structural.get("apical_above_soma", float("nan"))) else "",
                "apical_z_extent": f"{structural.get('apical_z_extent', 0):.2f}",
                "axon_frac": f"{structural.get('axon_frac', 0):.3f}",
                "apical_frac": f"{structural.get('apical_frac', 0):.3f}",
                "basal_frac": f"{structural.get('basal_frac', 0):.3f}",
                "pred_axon": pred_counter.get(2, 0),
                "pred_basal": pred_counter.get(3, 0),
                "pred_apical": pred_counter.get(4, 0),
                "elapsed_s": f"{elapsed_s:.2f}",
                "source": path.name.split("__", 1)[0] if "__" in path.name else "",
            })

            feature = {
                "file": path.name,
                "n_nodes_conf": len(nodes),
                "stage1_pred_conf_run": pr.stage1.cell_type,
                "stage1_conf_conf_run": float(pr.stage1.confidence),
                "pred_axon_conf_run": pred_counter.get(2, 0),
                "pred_basal_conf_run": pred_counter.get(3, 0),
                "pred_apical_conf_run": pred_counter.get(4, 0),
                "node_label_disagree_frac": 0.0,
                "node_conf_abs_delta_mean": 0.0,
                **_branch3_disagreement_features(labels, confs, base_labels, base_confs),
                **_confidence_summary(labels, confs),
                **_branch_summary(nodes, labels, confs, pr.stage1.cell_type, float(pr.stage1.confidence)),
                **_predicted_geometry(nodes, labels),
            }
            feature_rows.append(_round_row(feature))
        except Exception as exc:
            print(f"  WARN {path.name}: {exc}", flush=True)
            n_failed += 1
            continue

        if i % args.progress_every == 0:
            elapsed = (time.perf_counter() - t0) / 60.0
            eta = elapsed * (len(files) - i) / max(1, i)
            print(f"  ... {i}/{len(files)} ({elapsed:.1f} min, ETA {eta:.1f} min)", flush=True)

    if not label_rows:
        raise SystemExit("No label rows produced")

    out_labels.parent.mkdir(parents=True, exist_ok=True)
    out_features.parent.mkdir(parents=True, exist_ok=True)
    with out_labels.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(label_rows[0].keys()))
        writer.writeheader()
        writer.writerows(label_rows)
    with out_features.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(feature_rows[0].keys()))
        writer.writeheader()
        writer.writerows(feature_rows)

    f1s = np.asarray([float(r["held_out_F1"]) for r in label_rows], dtype=float)
    cts = np.asarray([r["cell_type_gt"] for r in label_rows], dtype=object)
    summary = {
        "model_dir": _display_path(model_dir),
        "split": args.split,
        "branch3_enabled": model["branch3_state"] is not None,
        "branch3_path": _display_path(model["branch3_path"]) if model["branch3_path"] else None,
        "out_labels": _display_path(out_labels),
        "out_features": _display_path(out_features),
        "n_scored": len(label_rows),
        "n_failed": n_failed,
        "elapsed_min": (time.perf_counter() - t0) / 60.0,
        "f1_mean": float(f1s.mean()),
        "f1_p10": float(np.percentile(f1s, 10)),
        "f1_p25": float(np.percentile(f1s, 25)),
        "bad_rate_f050": float((f1s < 0.5).mean()),
        "bad_rate_f060": float((f1s < 0.6).mean()),
        "by_cell_type": {
            ct: {
                "n": int((cts == ct).sum()),
                "f1_mean": float(f1s[cts == ct].mean()) if np.any(cts == ct) else None,
                "f1_p10": float(np.percentile(f1s[cts == ct], 10)) if np.any(cts == ct) else None,
                "bad_rate_f060": float((f1s[cts == ct] < 0.6).mean()) if np.any(cts == ct) else None,
            }
            for ct in ("pyramidal", "interneuron")
        },
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDone: {len(label_rows)} scored, {n_failed} failed")
    print(f"Wrote {out_labels}")
    print(f"Wrote {out_features}")
    print(f"Wrote {out_summary}")
    print(
        f"F1 mean={summary['f1_mean']:.4f} P10={summary['f1_p10']:.4f} "
        f"bad<.5={summary['bad_rate_f050']:.3f} bad<.6={summary['bad_rate_f060']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
