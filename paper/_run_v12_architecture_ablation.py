#!/usr/bin/env python3
"""Run the paper architecture ladder for v12.

Rows:
  - Stage2 only: GT cell type + subtree Stage2, no GNN, no Stage3, no Branch3.
  - Stage2 + GNN: adds apical/basal GNN, still before Stage3.
  - Stage2 + GNN + Stage3: adds topology refinement.
  - Full Branch3: adds conservative Branch3 rescue.
  - Stage1 active full: same full model, but Stage1 chooses the cell type.

Outputs:
  paper/results/final_ablation_ladder_rows.csv
  paper/results/final_ablation_ladder_summary.csv
  paper/results/final_ablation_ladder_summary.json
  paper/results/final_ablation_ladder_summary.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.branch_features import extract_branches  # noqa: E402
from hybrid.cell_type_detector import CELL_TYPE_LABEL_SETS, CellTypeResult  # noqa: E402
from hybrid.comprehensive_metrics import build_full_report  # noqa: E402
from hybrid.evaluate import per_cell_neurite_f1, per_cell_per_class_f1  # noqa: E402
from hybrid.features import parse_swc  # noqa: E402
from hybrid.pipeline import (  # noqa: E402
    _apply_branch3_rescue,
    _apply_gnn_override,
    _best_apical_owner,
    _branch_feature_with_owner,
    _load_stage2_bundle,
    _predict_subtree_owner_map,
    _select_stage2_model,
    run_pipeline_on_nodes,
)
from hybrid.stage3_refine import refine  # noqa: E402
from paper.gnn_branch3_inference import load_branch3  # noqa: E402
from paper.gnn_inference import load_gnn  # noqa: E402


DATA_DIR = ROOT / "data" / "v12_uncurated"
RESULTS_DIR = ROOT / "paper" / "results"
MODE_ORDER = [
    "stage2_only_gt_celltype",
    "stage2_gnn_gt_celltype",
    "stage2_gnn_stage3_gt_celltype",
    "full_branch3_gt_celltype",
    "full_branch3_stage1_active",
]
MODE_LABEL = {
    "stage2_only_gt_celltype": "Stage2 only (GT cell type)",
    "stage2_gnn_gt_celltype": "Stage2 + GNN (GT cell type)",
    "stage2_gnn_stage3_gt_celltype": "Stage2 + GNN + Stage3 (GT cell type)",
    "full_branch3_gt_celltype": "Full Branch3 (GT cell type)",
    "full_branch3_stage1_active": "Full Branch3 (Stage1 active)",
}
METRIC_KEYS = [
    "accuracy",
    "neurite_macro_f1",
    "axon_f1",
    "basal_f1",
    "apical_f1",
    "per_cell_accuracy_mean",
    "per_cell_accuracy_p10",
    "per_cell_f1_mean",
    "per_cell_f1_p10",
    "per_cell_f1_p25",
    "pyramidal_neurite_f1",
    "pyramidal_apical_f1",
    "interneuron_neurite_f1",
]


def _test_files_for_seed(model_dir: Path) -> list[tuple[str, Path]]:
    split_json = model_dir / "train_test_split.json"
    if not split_json.is_file():
        raise SystemExit(f"MISSING: {split_json}")
    split = json.loads(split_json.read_text(encoding="utf-8"))
    out: list[tuple[str, Path]] = []
    for ct in ("pyramidal", "interneuron"):
        for fn in split["test"].get(ct, []):
            p = DATA_DIR / ct / "swc" / fn
            if p.is_file():
                out.append((ct, p))
    return out


def _raw_stage2_labels(
    nodes,
    file_path: str,
    cell_type: str,
    bundle: dict,
    use_subtree_stage2: bool = True,
) -> tuple[list[int], list[float], object, dict[int, dict[str, float | int]], int | None, bool]:
    """Mirror hybrid.pipeline Stage2, but stop before GNN/Stage3/Branch3."""
    n = len(nodes)
    label_set = set(CELL_TYPE_LABEL_SETS.get(cell_type, {1, 2, 3}))
    neurite_labels = sorted(label_set - {1})

    model, default_label = _select_stage2_model(bundle, cell_type)
    subtree_models_by_ct = bundle.get("subtree_owner_models_by_cell_type")
    if subtree_models_by_ct:
        subtree_owner_model = subtree_models_by_ct.get(cell_type)
    else:
        legacy = bundle.get("pyramidal_subtree_owner_model")
        subtree_owner_model = legacy if cell_type == "pyramidal" else None

    morph = extract_branches(nodes, cell_type, file_path)
    subtree_owner_map = _predict_subtree_owner_map(nodes, cell_type, subtree_owner_model)
    apical_owner_root = _best_apical_owner(subtree_owner_map) if cell_type == "pyramidal" else None

    proxy_soma = set(morph.soma_indices)
    node_labels = [1 if i in proxy_soma else 0 for i in range(n)]
    node_confidences = [1.0 if i in proxy_soma else 0.0 for i in range(n)]
    is_binary_b1 = bundle.get("kind") == "axon_dendrite_binary"

    if use_subtree_stage2:
        if not subtree_owner_map:
            fallback = 3 if 3 in neurite_labels else (neurite_labels[0] if neurite_labels else 3)
            for br in morph.branches:
                for node_idx in br.node_indices:
                    node_labels[node_idx] = fallback
                    node_confidences[node_idx] = 0.5
        else:
            for br in morph.branches:
                pr = br.primary_root_idx
                info = subtree_owner_map.get(pr) if pr is not None else None
                if info is None:
                    best_label = 3 if 3 in neurite_labels else (neurite_labels[0] if neurite_labels else 3)
                    best_conf = 0.5
                else:
                    pred = int(info.get("pred", 3))
                    if pred not in neurite_labels:
                        pred = 3 if 3 in neurite_labels else (neurite_labels[0] if neurite_labels else 3)
                    best_label = pred
                    best_conf = float(info.get("conf", 0.5))
                for node_idx in br.node_indices:
                    node_labels[node_idx] = best_label
                    node_confidences[node_idx] = best_conf
    elif model is not None:
        for br in morph.branches:
            x = _branch_feature_with_owner(br, subtree_owner_map).reshape(1, -1)
            probs = model.predict_proba(x)[0]
            classes = model.classes_

            if is_binary_b1:
                cls_list = list(classes)
                p_axon = float(probs[cls_list.index(1)]) if 1 in cls_list else 0.0
                p_dend = float(probs[cls_list.index(0)]) if 0 in cls_list else 1.0 - p_axon
                if p_axon > 0.5 and 2 in neurite_labels:
                    best_label = 2
                    best_conf = p_axon
                else:
                    dend_candidates = [lbl for lbl in neurite_labels if lbl != 2]
                    best_label = 3 if 3 in dend_candidates else (
                        dend_candidates[0] if dend_candidates else (neurite_labels[0] if neurite_labels else 3)
                    )
                    best_conf = p_dend
            else:
                valid_probs: dict[int, float] = {}
                for cls, prob in zip(classes, probs):
                    if int(cls) in neurite_labels:
                        valid_probs[int(cls)] = float(prob)
                if valid_probs:
                    total = sum(valid_probs.values())
                    if total > 0:
                        valid_probs = {k: v / total for k, v in valid_probs.items()}
                    best_label = max(valid_probs, key=lambda k: valid_probs[k])
                    best_conf = valid_probs[best_label]
                else:
                    best_label = neurite_labels[0] if neurite_labels else 3
                    best_conf = 0.5

            for node_idx in br.node_indices:
                node_labels[node_idx] = best_label
                node_confidences[node_idx] = best_conf
    else:
        fallback = default_label if default_label is not None else (neurite_labels[0] if neurite_labels else 3)
        for br in morph.branches:
            for node_idx in br.node_indices:
                node_labels[node_idx] = fallback
                node_confidences[node_idx] = 0.5

    for i in range(n):
        if node_labels[i] == 0:
            node_labels[i] = neurite_labels[0] if neurite_labels else 3
            node_confidences[i] = 0.3

    return node_labels, node_confidences, morph, subtree_owner_map, apical_owner_root, is_binary_b1


def _with_gnn(labels, confs, morph, gnn_state, cell_type: str, apical_owner_root: int | None, is_binary_b1: bool):
    labels = list(labels)
    confs = list(confs)
    if gnn_state is not None and cell_type == "pyramidal":
        _apply_gnn_override(
            labels,
            confs,
            morph,
            gnn_state,
            apical_evidence=((is_binary_b1 or True) and apical_owner_root is not None),
        )
    return labels, confs


def _synthetic_stage1(cell_type: str) -> CellTypeResult:
    return CellTypeResult(
        cell_type=cell_type,
        confidence=1.0,
        probabilities={cell_type: 1.0},
        label_set=set(CELL_TYPE_LABEL_SETS.get(cell_type, {1, 2, 3})),
        structure_flags={"override_cell_type": True},
        features={},
    )


def _stage3_labels(nodes, labels: list[int], confs: list[float], cell_type: str, apical_owner_root: int | None) -> list[int]:
    s3 = refine(
        nodes,
        labels,
        confs,
        _synthetic_stage1(cell_type),
        apical_owner_root=apical_owner_root,
    )
    return [rl.label for rl in s3.labels]


def _score_row(seed: int, mode: str, path: Path, cell_type_gt: str, gt: list[int], pred: list[int],
               stage1_pred: str = "", stage1_conf: float | None = None) -> dict:
    per_class = per_cell_per_class_f1(gt, pred, cell_type_gt)
    n_correct = sum(1 for g, q in zip(gt, pred) if g == q)
    return {
        "seed": seed,
        "ablation": mode,
        "ablation_label": MODE_LABEL[mode],
        "file": path.name,
        "cell_type_gt": cell_type_gt,
        "stage1_pred": stage1_pred,
        "stage1_conf": stage1_conf,
        "n_nodes": len(gt),
        "accuracy": n_correct / max(1, len(gt)),
        "neurite_macro_f1": per_cell_neurite_f1(gt, pred, cell_type_gt),
        "soma_f1": per_class.get("soma"),
        "axon_f1": per_class.get("axon"),
        "basal_f1": per_class.get("basal/dendrite"),
        "apical_f1": per_class.get("apical"),
        "pred_soma": sum(1 for q in pred if q == 1),
        "pred_axon": sum(1 for q in pred if q == 2),
        "pred_basal": sum(1 for q in pred if q == 3),
        "pred_apical": sum(1 for q in pred if q == 4),
    }


def _report_from_rows(mode: str, seed: int, eval_records: list[dict], inference_min: float,
                      stage1_records: list[dict] | None = None) -> dict:
    gt_pool: list[int] = []
    pred_pool: list[int] = []
    cell_records: list[dict] = []
    for rec in eval_records:
        gt_pool.extend(rec["gt"])
        pred_pool.extend(rec["pred"])
        cell_records.append({
            "cell_type": rec["cell_type"],
            "n_nodes": rec["n_nodes"],
            "gt": rec["gt"],
            "pred": rec["pred"],
        })
    return build_full_report(
        method=f"{mode}_seed{seed}",
        gt_pool=gt_pool,
        pred_pool=pred_pool,
        cell_records=cell_records,
        inference_min=inference_min,
        stage1_records=stage1_records,
    )


def _metric_row(seed: int, mode: str, report: dict) -> dict:
    corpus = report["corpus"]
    per_class = corpus["per_class"]
    pc = report["per_cell"]
    by_ct = report.get("by_cell_type", {})
    pyr = by_ct.get("pyramidal", {})
    intr = by_ct.get("interneuron", {})

    def ct_metric(block: dict, key: str) -> float:
        try:
            if key == "neurite_f1":
                return float(block["corpus"]["neurite_macro_f1"])
            return float(block["corpus"]["per_class"][key]["f1"])
        except Exception:
            return 0.0

    row = {
        "seed": seed,
        "ablation": mode,
        "ablation_label": MODE_LABEL[mode],
        "n_test_cells": report["n_test_cells"],
        "accuracy": corpus["accuracy"],
        "neurite_macro_f1": corpus["neurite_macro_f1"],
        "axon_f1": per_class["axon"]["f1"],
        "basal_f1": per_class["basal/dendrite"]["f1"],
        "apical_f1": per_class["apical"]["f1"],
        "per_cell_accuracy_mean": pc["accuracy"]["mean"],
        "per_cell_accuracy_p10": pc["accuracy"]["p10"],
        "per_cell_f1_mean": pc["neurite_macro_f1"]["mean"],
        "per_cell_f1_p10": pc["neurite_macro_f1"]["p10"],
        "per_cell_f1_p25": pc["neurite_macro_f1"]["p25"],
        "pyramidal_neurite_f1": ct_metric(pyr, "neurite_f1"),
        "pyramidal_apical_f1": ct_metric(pyr, "apical"),
        "interneuron_neurite_f1": ct_metric(intr, "neurite_f1"),
    }
    s1 = report.get("stage1")
    if s1 is not None:
        row["stage1_accuracy"] = s1.get("accuracy", 0.0)
        row["stage1_wrong"] = s1.get("n_wrong", 0)
    else:
        row["stage1_accuracy"] = ""
        row["stage1_wrong"] = ""
    return row


def _mean_sd(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row["ablation"]), []).append(row)
    out: list[dict] = []
    for mode in MODE_ORDER:
        items = groups.get(mode, [])
        if not items:
            continue
        rec: dict = {
            "ablation": mode,
            "ablation_label": MODE_LABEL[mode],
            "n_seeds": len(items),
        }
        for key in METRIC_KEYS:
            vals = [float(r[key]) for r in items]
            rec[f"{key}_mean"] = statistics.fmean(vals)
            rec[f"{key}_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
            rec[f"{key}_min"] = min(vals)
            rec[f"{key}_max"] = max(vals)
        s1_vals = [r.get("stage1_accuracy") for r in items if r.get("stage1_accuracy") != ""]
        if s1_vals:
            s1_float = [float(v) for v in s1_vals]
            rec["stage1_accuracy_mean"] = statistics.fmean(s1_float)
            rec["stage1_accuracy_sd"] = statistics.stdev(s1_float) if len(s1_float) > 1 else 0.0
        else:
            rec["stage1_accuracy_mean"] = ""
            rec["stage1_accuracy_sd"] = ""
        out.append(rec)
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = row.copy()
            for key, value in out.items():
                if value is None:
                    out[key] = ""
                elif isinstance(value, float):
                    out[key] = f"{value:.6f}"
            writer.writerow(out)


def _read_existing_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _completed_seeds(seed_rows: list[dict]) -> set[int]:
    by_seed: dict[int, set[str]] = {}
    for row in seed_rows:
        try:
            seed = int(row["seed"])
        except Exception:
            continue
        by_seed.setdefault(seed, set()).add(row.get("ablation", ""))
    return {seed for seed, modes in by_seed.items() if set(MODE_ORDER).issubset(modes)}


def _summary_text(summary_rows: list[dict]) -> str:
    lines = [
        "V12 architecture ablation ladder",
        "================================",
        "",
        "Rows are mean +/- SD over seeds. Stage2 rows bypass Stage1 with GT cell type.",
        "",
    ]
    show = [
        "accuracy",
        "neurite_macro_f1",
        "apical_f1",
        "per_cell_f1_mean",
        "per_cell_f1_p10",
        "pyramidal_neurite_f1",
        "pyramidal_apical_f1",
        "interneuron_neurite_f1",
    ]
    for row in summary_rows:
        lines.append(f"{row['ablation_label']}  (n={row['n_seeds']})")
        for key in show:
            lines.append(
                f"  {key:<28} {float(row[f'{key}_mean']):.4f} +/- {float(row[f'{key}_sd']):.4f} "
                f"[{float(row[f'{key}_min']):.4f}, {float(row[f'{key}_max']):.4f}]"
            )
        if row.get("stage1_accuracy_mean") != "":
            lines.append(
                f"  {'stage1_accuracy':<28} {float(row['stage1_accuracy_mean']):.4f} "
                f"+/- {float(row['stage1_accuracy_sd']):.4f}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def eval_seed(seed: int) -> tuple[list[dict], list[dict], dict[str, dict]]:
    model_dir = ROOT / "paper" / "models" / f"v12_gentle_seed{seed}"
    s1_path = model_dir / "cell_type_classifier.pkl"
    s2_path = model_dir / "branch_classifier.pkl"
    gnn_path = model_dir / "gnn_apical_basal.pt"
    branch3_path = model_dir / "gnn_branch3_rescue.pt"
    for p in (s1_path, s2_path, gnn_path, branch3_path):
        if not p.is_file():
            raise SystemExit(f"MISSING: {p}")

    test_files = _test_files_for_seed(model_dir)
    print(f"\n=== seed={seed} architecture ablation ===")
    print(f"  test cells: {len(test_files)}")
    print(f"  model dir: {model_dir.relative_to(ROOT)}")

    bundle = _load_stage2_bundle(s2_path)
    gnn_state = load_gnn(gnn_path)
    branch3_state = load_branch3(branch3_path)

    per_cell_rows: list[dict] = []
    eval_records_by_mode: dict[str, list[dict]] = {mode: [] for mode in MODE_ORDER}
    stage1_records: list[dict] = []

    t0 = time.perf_counter()
    for i, (ct_gt, path) in enumerate(test_files):
        try:
            nodes = parse_swc(path)
            if not nodes:
                continue
            gt = [n.type for n in nodes]
        except Exception as exc:
            print(f"  WARN {path.name}: {exc}")
            continue

        raw_labels, raw_confs, morph, owner_map, apical_owner_root, is_binary_b1 = _raw_stage2_labels(
            nodes,
            str(path),
            ct_gt,
            bundle,
            use_subtree_stage2=True,
        )
        gnn_labels, _gnn_confs = _with_gnn(
            raw_labels,
            raw_confs,
            morph,
            gnn_state,
            ct_gt,
            apical_owner_root,
            is_binary_b1,
        )
        no_b3_labels = _stage3_labels(
            nodes,
            list(gnn_labels),
            list(_gnn_confs),
            ct_gt,
            apical_owner_root,
        )
        branch3_labels = list(gnn_labels)
        branch3_confs = list(_gnn_confs)
        if ct_gt == "pyramidal":
            _apply_branch3_rescue(
                branch3_labels,
                branch3_confs,
                morph,
                branch3_state,
                owner_map,
                apical_owner_root,
            )
        full_gt_labels = _stage3_labels(
            nodes,
            branch3_labels,
            branch3_confs,
            ct_gt,
            apical_owner_root,
        )
        full_stage1 = run_pipeline_on_nodes(
            nodes,
            file_path=str(path),
            stage1_model=s1_path,
            stage2_model=s2_path,
            gnn_state=gnn_state,
            branch3_state=branch3_state,
            use_subtree_stage2=True,
        )

        preds = {
            "stage2_only_gt_celltype": raw_labels,
            "stage2_gnn_gt_celltype": gnn_labels,
            "stage2_gnn_stage3_gt_celltype": no_b3_labels,
            "full_branch3_gt_celltype": full_gt_labels,
            "full_branch3_stage1_active": list(full_stage1.node_labels),
        }
        stage1_records.append({
            "cell_type_gt": ct_gt,
            "stage1_pred": full_stage1.stage1.cell_type,
            "stage1_conf": float(full_stage1.stage1.confidence),
        })
        for mode, pred in preds.items():
            stage1_pred = full_stage1.stage1.cell_type if mode == "full_branch3_stage1_active" else ""
            stage1_conf = float(full_stage1.stage1.confidence) if mode == "full_branch3_stage1_active" else None
            per_cell_rows.append(_score_row(seed, mode, path, ct_gt, gt, pred, stage1_pred, stage1_conf))
            eval_records_by_mode[mode].append({
                "cell_type": ct_gt,
                "n_nodes": len(gt),
                "gt": gt,
                "pred": pred,
            })

        if (i + 1) % 200 == 0:
            print(f"    ... {i + 1}/{len(test_files)} ({(time.perf_counter() - t0) / 60.0:.1f} min)")

    elapsed = (time.perf_counter() - t0) / 60.0
    per_mode_min = elapsed / len(MODE_ORDER)
    print(f"  inference: {elapsed:.1f} min")

    reports: dict[str, dict] = {}
    metric_rows: list[dict] = []
    for mode in MODE_ORDER:
        report = _report_from_rows(
            mode,
            seed,
            eval_records_by_mode[mode],
            inference_min=per_mode_min,
            stage1_records=stage1_records if mode == "full_branch3_stage1_active" else None,
        )
        reports[mode] = report
        metric_rows.append(_metric_row(seed, mode, report))
        pc = report["per_cell"]["neurite_macro_f1"]
        print(
            f"  {mode:<32} acc={report['corpus']['accuracy']:.4f} "
            f"neurite_F1={report['corpus']['neurite_macro_f1']:.4f} "
            f"pc_F1_mean={pc['mean']:.4f} pc_F1_p10={pc['p10']:.4f}"
        )

    return per_cell_rows, metric_rows, reports


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="123,42,789")
    ap.add_argument("--resume-existing", action="store_true",
                    help="Reuse complete seeds already present in final_ablation_ladder_seed_rows.csv.")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    all_rows: list[dict] = []
    all_metric_rows: list[dict] = []
    all_reports: dict[str, dict[str, dict]] = {}
    done: set[int] = set()
    if args.resume_existing:
        all_rows = _read_existing_csv(RESULTS_DIR / "final_ablation_ladder_rows.csv")
        all_metric_rows = _read_existing_csv(RESULTS_DIR / "final_ablation_ladder_seed_rows.csv")
        done = _completed_seeds(all_metric_rows)
        if done:
            print(f"Resuming: completed seeds already on disk: {sorted(done)}")

    overall_t0 = time.perf_counter()
    for seed in seeds:
        if seed in done:
            print(f"\n=== seed={seed} already complete; skipping ===")
            continue
        rows, metric_rows, reports = eval_seed(seed)
        all_rows.extend(rows)
        all_metric_rows.extend(metric_rows)
        all_reports[str(seed)] = reports
        _write_csv(RESULTS_DIR / "final_ablation_ladder_rows.csv", all_rows)
        _write_csv(RESULTS_DIR / "final_ablation_ladder_seed_rows.csv", all_metric_rows)
        print(f"  incremental save: {RESULTS_DIR / 'final_ablation_ladder_rows.csv'}")

    summary_rows = _mean_sd(all_metric_rows)
    _write_csv(RESULTS_DIR / "final_ablation_ladder_summary.csv", summary_rows)
    (RESULTS_DIR / "final_ablation_ladder_summary.json").write_text(
        json.dumps(
            {
                "seeds": seeds,
                "mode_order": MODE_ORDER,
                "seed_rows": all_metric_rows,
                "summary": summary_rows,
                "reports": all_reports,
                "note": "Architecture ladder on each seed's own held-out split. Stage2 rows use GT cell type; the final row activates Stage1.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (RESULTS_DIR / "final_ablation_ladder_summary.txt").write_text(
        _summary_text(summary_rows),
        encoding="utf-8",
    )

    # Short aliases matching the paper checklist wording.
    _write_csv(RESULTS_DIR / "final_ablation_table.csv", summary_rows)
    (RESULTS_DIR / "final_ablation_table.json").write_text(
        json.dumps({"seeds": seeds, "summary": summary_rows}, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / "final_ablation_table.txt").write_text(
        _summary_text(summary_rows),
        encoding="utf-8",
    )

    print(f"\nTotal wall clock: {(time.perf_counter() - overall_t0) / 60.0:.1f} min")
    print(f"Wrote {RESULTS_DIR / 'final_ablation_ladder_summary.txt'}")
    print(f"Wrote {RESULTS_DIR / 'final_ablation_ladder_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
