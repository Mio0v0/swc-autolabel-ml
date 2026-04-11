#!/usr/bin/env python3
"""Evaluate CIC CA1 auto-label outputs against original labels.

Dataset layout:
- batch_XX/original          : ground-truth SWC files
- batch_XX/auto_label_output : predicted SWC files from the current algorithm

Outputs are written to:
- data/CIC_CA1_Dataset/accuracy_output/

Metrics:
- pooled node accuracy over supported labels {1,2,3,4}
- exact pooled per-class precision / recall / F1 from confusion counts
- pooled macro-F1
- per-file node accuracy and per-file macro-F1 with median summaries
- pooled confusion matrix
- subtree-level accuracy for primary soma-child subtrees

Ground-truth nodes with labels outside {1,2,3,4} are excluded from the evaluation
so custom labels in the original reconstructions do not distort the score.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from swcstudio.core.swc_io import parse_swc_text_preserve_tokens

LABELS = [1, 2, 3, 4]
NEURITE_LABELS = [2, 3, 4]
LABEL_NAMES = {
    0: "other",
    1: "soma",
    2: "axon",
    3: "basal",
    4: "apical",
}
DEFAULT_DATASET_ROOT = ROOT / "data" / "CIC_CA1_Dataset"
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_ROOT / "accuracy_output"


def _iter_batches(dataset_root: Path, requested: set[str]) -> list[Path]:
    batches = [p for p in sorted(dataset_root.iterdir()) if p.is_dir() and p.name.startswith("batch_")]
    if requested:
        batches = [p for p in batches if p.name in requested]
    return batches


def _merge_gt_pred(gt_path: Path, pred_path: Path) -> pd.DataFrame:
    gt = parse_swc_text_preserve_tokens(gt_path.read_text(encoding="utf-8", errors="ignore"))
    pred = parse_swc_text_preserve_tokens(pred_path.read_text(encoding="utf-8", errors="ignore"))

    gt_view = gt[["id", "parent", "type"]].rename(columns={"type": "gt_type"})
    pred_view = pred[["id", "parent", "type"]].rename(columns={"type": "pred_type", "parent": "pred_parent"})
    merged = gt_view.merge(pred_view, on="id", how="outer", indicator=True)

    if not (merged["_merge"] == "both").all():
        missing_gt = int((merged["_merge"] == "right_only").sum())
        missing_pred = int((merged["_merge"] == "left_only").sum())
        raise ValueError(f"node id mismatch (missing_in_gt={missing_gt}, missing_in_pred={missing_pred})")
    if not (merged["parent"] == merged["pred_parent"]).all():
        raise ValueError("parent structure mismatch between original and prediction")

    merged = merged.drop(columns=["_merge", "pred_parent"]).sort_values("id").reset_index(drop=True)
    merged["gt_type"] = merged["gt_type"].astype(int)
    merged["pred_type"] = merged["pred_type"].astype(int)
    merged["parent"] = merged["parent"].astype(int)
    return merged


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _median(values: Iterable[float]) -> float:
    vals = list(values)
    return float(statistics.median(vals)) if vals else 0.0


def _majority_label(values: list[int], preferred: int | None = None) -> int | None:
    if not values:
        return None
    counts = Counter(values)
    max_count = max(counts.values())
    winners = [label for label, count in counts.items() if count == max_count]
    if preferred in winners:
        return preferred
    return min(winners)


def _primary_subtree_records(merged: pd.DataFrame, batch_name: str, file_name: str) -> list[dict[str, object]]:
    children: dict[int, list[int]] = defaultdict(list)
    gt_type_by_id = {int(row.id): int(row.gt_type) for row in merged.itertuples(index=False)}
    pred_type_by_id = {int(row.id): int(row.pred_type) for row in merged.itertuples(index=False)}
    for row in merged.itertuples(index=False):
        children[int(row.parent)].append(int(row.id))

    soma_ids = {node_id for node_id, label in gt_type_by_id.items() if label == 1}
    if not soma_ids:
        return []

    primary_roots = sorted(
        child_id
        for soma_id in soma_ids
        for child_id in children.get(soma_id, [])
        if child_id not in soma_ids
    )

    records: list[dict[str, object]] = []
    for root_id in primary_roots:
        stack = [root_id]
        subtree_ids: list[int] = []
        while stack:
            node_id = stack.pop()
            subtree_ids.append(node_id)
            stack.extend(children.get(node_id, []))

        gt_values = [gt_type_by_id[n] for n in subtree_ids if gt_type_by_id[n] in NEURITE_LABELS]
        if not gt_values:
            continue
        pred_values = [pred_type_by_id[n] for n in subtree_ids if pred_type_by_id[n] in NEURITE_LABELS]

        gt_root_type = gt_type_by_id.get(root_id)
        pred_root_type = pred_type_by_id.get(root_id)
        gt_label = _majority_label(gt_values, gt_root_type if gt_root_type in NEURITE_LABELS else None)
        pred_label = _majority_label(pred_values, pred_root_type if pred_root_type in NEURITE_LABELS else None)
        gt_label = 0 if gt_label is None else int(gt_label)
        pred_label = 0 if pred_label is None else int(pred_label)

        records.append(
            {
                "batch": batch_name,
                "file_name": file_name,
                "subtree_root_id": int(root_id),
                "subtree_nodes": int(len(subtree_ids)),
                "subtree_supported_gt_nodes": int(len(gt_values)),
                "gt_subtree_label": gt_label,
                "gt_subtree_label_name": LABEL_NAMES.get(gt_label, str(gt_label)),
                "pred_subtree_label": pred_label,
                "pred_subtree_label_name": LABEL_NAMES.get(pred_label, str(pred_label)),
                "root_gt_label": int(gt_root_type),
                "root_pred_label": int(pred_root_type),
                "correct": int(pred_label == gt_label),
            }
        )
    return records


def _file_confusion_counts(y_true: list[int], y_pred: list[int]) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for gt_label, pred_label in zip(y_true, y_pred):
        pred_bucket = pred_label if pred_label in LABELS else 0
        counts[(gt_label, pred_bucket)] += 1
    return counts


def _evaluate_file(
    batch_name: str,
    gt_path: Path,
    pred_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]], Counter[tuple[int, int]]]:
    merged = _merge_gt_pred(gt_path, pred_path)
    eval_df = merged.loc[merged["gt_type"].isin(LABELS), ["id", "parent", "gt_type", "pred_type"]].copy()
    y_true = eval_df["gt_type"].astype(int).tolist()
    y_pred = [label if label in LABELS else 0 for label in eval_df["pred_type"].astype(int).tolist()]

    total = len(y_true)
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    node_accuracy = _safe_div(correct, total)
    confusion = _file_confusion_counts(y_true, y_pred)

    per_class: dict[int, dict[str, float]] = {}
    f1_values_present: list[float] = []
    for label in LABELS:
        tp = confusion.get((label, label), 0)
        support = sum(confusion.get((label, pred_label), 0) for pred_label in LABELS + [0])
        fp = sum(confusion.get((gt_label, label), 0) for gt_label in LABELS if gt_label != label)
        fn = support - tp
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        if support > 0:
            f1_values_present.append(f1)

    macro_f1 = _safe_div(sum(f1_values_present), len(f1_values_present)) if f1_values_present else 0.0
    subtree_records = _primary_subtree_records(merged, batch_name=batch_name, file_name=gt_path.name)
    subtree_total = len(subtree_records)
    subtree_correct = sum(int(r["correct"]) for r in subtree_records)
    subtree_accuracy = _safe_div(subtree_correct, subtree_total)

    row = {
        "batch": batch_name,
        "file_name": gt_path.name,
        "original_path": str(gt_path),
        "predicted_path": str(pred_path),
        "nodes_total_raw": int(len(merged)),
        "nodes_evaluated": int(total),
        "nodes_excluded_gt_outside_1_4": int(len(merged) - total),
        "correct_nodes": int(correct),
        "node_accuracy": node_accuracy,
        "macro_f1": macro_f1,
        "subtree_total": int(subtree_total),
        "subtree_correct": int(subtree_correct),
        "subtree_accuracy": subtree_accuracy,
    }
    for label in LABELS:
        prefix = LABEL_NAMES[label]
        row[f"{prefix}_precision"] = per_class[label]["precision"]
        row[f"{prefix}_recall"] = per_class[label]["recall"]
        row[f"{prefix}_f1"] = per_class[label]["f1"]
        row[f"{prefix}_support"] = int(per_class[label]["support"])
    return row, subtree_records, confusion


def _per_class_from_confusion(counts: Counter[tuple[int, int]]) -> pd.DataFrame:
    rows = []
    for label in LABELS:
        tp = counts.get((label, label), 0)
        support = sum(counts.get((label, pred_label), 0) for pred_label in LABELS + [0])
        pred_total = sum(counts.get((gt_label, label), 0) for gt_label in LABELS)
        fp = pred_total - tp
        fn = support - tp
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "label": label,
                "label_name": LABEL_NAMES[label],
                "support": int(support),
                "predicted": int(pred_total),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def _write_confusion_csv(path: Path, counts: Counter[tuple[int, int]]) -> None:
    labels_with_other = LABELS + [0]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gt\\pred", *[LABEL_NAMES[label] for label in labels_with_other]])
        for gt_label in LABELS:
            writer.writerow([
                LABEL_NAMES[gt_label],
                *[counts.get((gt_label, pred_label), 0) for pred_label in labels_with_other],
            ])


def _batch_summary(
    per_file_df: pd.DataFrame,
    subtree_df: pd.DataFrame,
    batch_confusions: dict[str, Counter[tuple[int, int]]],
) -> pd.DataFrame:
    rows = []
    for batch_name, group in per_file_df.groupby("batch", sort=True):
        subtree_group = subtree_df.loc[subtree_df["batch"] == batch_name]
        total_nodes = int(group["nodes_evaluated"].sum())
        correct_nodes = int(group["correct_nodes"].sum())
        class_df = _per_class_from_confusion(batch_confusions.get(batch_name, Counter()))
        class_present = class_df.loc[class_df["support"] > 0, "f1"].tolist()
        rows.append(
            {
                "batch": batch_name,
                "files": int(len(group)),
                "nodes_evaluated": total_nodes,
                "node_accuracy_pooled": _safe_div(correct_nodes, total_nodes),
                "node_accuracy_median_per_file": _median(group["node_accuracy"].tolist()),
                "macro_f1_pooled": _safe_div(sum(class_present), len(class_present)) if class_present else 0.0,
                "macro_f1_median_per_file": _median(group["macro_f1"].tolist()),
                "subtree_total": int(len(subtree_group)),
                "subtree_accuracy": _safe_div(int(subtree_group["correct"].sum()), int(len(subtree_group))),
            }
        )
    return pd.DataFrame(rows)


def _format_pct(value: float) -> str:
    return f"{100.0 * float(value):.4f}%"


def _write_report(
    path: Path,
    *,
    dataset_root: Path,
    per_file_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    subtree_df: pd.DataFrame,
    confusion_counts: Counter[tuple[int, int]],
    confusion_total: int,
) -> None:
    total_nodes = int(per_file_df["nodes_evaluated"].sum())
    correct_nodes = int(per_file_df["correct_nodes"].sum())
    pooled_node_accuracy = _safe_div(correct_nodes, total_nodes)
    class_present = per_class_df.loc[per_class_df["support"] > 0, "f1"].tolist()
    pooled_macro_f1 = _safe_div(sum(class_present), len(class_present)) if class_present else 0.0
    median_file_accuracy = _median(per_file_df["node_accuracy"].tolist())
    median_file_macro_f1 = _median(per_file_df["macro_f1"].tolist())
    subtree_total = int(len(subtree_df))
    subtree_correct = int(subtree_df["correct"].sum()) if subtree_total else 0
    subtree_accuracy = _safe_div(subtree_correct, subtree_total)
    excluded_nodes = int(per_file_df["nodes_excluded_gt_outside_1_4"].sum())

    lines = []
    lines.append("# CIC CA1 Auto-Label Evaluation")
    lines.append("")
    lines.append(f"Dataset root: `{dataset_root}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Files evaluated: {len(per_file_df)}")
    lines.append(f"- Nodes evaluated: {total_nodes}")
    lines.append(f"- Ground-truth nodes excluded because labels were outside 1..4: {excluded_nodes}")
    lines.append(f"- Pooled node accuracy: {_format_pct(pooled_node_accuracy)}")
    lines.append(f"- Median per-file node accuracy: {_format_pct(median_file_accuracy)}")
    lines.append(f"- Pooled macro-F1: {_format_pct(pooled_macro_f1)}")
    lines.append(f"- Median per-file macro-F1: {_format_pct(median_file_macro_f1)}")
    lines.append(f"- Primary-subtree accuracy: {_format_pct(subtree_accuracy)} ({subtree_correct}/{subtree_total})")
    lines.append("")
    lines.append("## Per-Class Metrics")
    lines.append("")
    lines.append("| Class | Support | Precision | Recall | F1 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in per_class_df.itertuples(index=False):
        lines.append(
            f"| {row.label_name} | {int(row.support)} | {_format_pct(row.precision)} | {_format_pct(row.recall)} | {_format_pct(row.f1)} |"
        )
    lines.append("")
    lines.append("## Per-Batch Summary")
    lines.append("")
    lines.append("| Batch | Files | Pooled Node Accuracy | Median File Accuracy | Pooled Macro-F1 | Median File Macro-F1 | Subtree Accuracy |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in batch_df.itertuples(index=False):
        lines.append(
            f"| {row.batch} | {int(row.files)} | {_format_pct(row.node_accuracy_pooled)} | {_format_pct(row.node_accuracy_median_per_file)} | "
            f"{_format_pct(row.macro_f1_pooled)} | {_format_pct(row.macro_f1_median_per_file)} | {_format_pct(row.subtree_accuracy)} |"
        )
    lines.append("")
    lines.append("## Confusion Matrix")
    lines.append("")
    lines.append(f"Evaluated nodes in confusion matrix: {confusion_total}")
    lines.append("")
    header_labels = LABELS + [0]
    lines.append("| Ground Truth \\ Prediction | " + " | ".join(LABEL_NAMES[label] for label in header_labels) + " |")
    lines.append("| --- | " + " | ".join(["---:"] * len(header_labels)) + " |")
    for gt_label in LABELS:
        lines.append(
            "| " + LABEL_NAMES[gt_label] + " | " + " | ".join(
                str(confusion_counts.get((gt_label, pred_label), 0)) for pred_label in header_labels
            ) + " |"
        )
    lines.append("")
    worst = per_file_df.sort_values(["node_accuracy", "macro_f1", "subtree_accuracy"], ascending=[True, True, True]).head(10)
    lines.append("## Lowest-Accuracy Files")
    lines.append("")
    lines.append("| Batch | File | Node Accuracy | Macro-F1 | Subtree Accuracy |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for row in worst.itertuples(index=False):
        lines.append(
            f"| {row.batch} | {row.file_name} | {_format_pct(row.node_accuracy)} | {_format_pct(row.macro_f1)} | {_format_pct(row.subtree_accuracy)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate CIC CA1 auto-label outputs against original labels.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch", action="append", default=[], help="Optional batch_XX filter; can be passed multiple times")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    batches = _iter_batches(dataset_root, set(args.batch or []))
    if not batches:
        raise RuntimeError("no matching batch folders found")

    file_rows: list[dict[str, object]] = []
    subtree_rows: list[dict[str, object]] = []
    total_confusion: Counter[tuple[int, int]] = Counter()
    batch_confusions: dict[str, Counter[tuple[int, int]]] = defaultdict(Counter)

    print(f"dataset_root: {dataset_root}")
    print(f"output_dir: {output_dir}")
    for batch_dir in batches:
        original_dir = batch_dir / "original"
        predicted_dir = batch_dir / "auto_label_output"
        if not original_dir.exists() or not predicted_dir.exists():
            raise FileNotFoundError(f"missing original or auto_label_output in {batch_dir}")
        gt_files = sorted(p for p in original_dir.glob("*.swc") if p.is_file())
        print(f"evaluating {batch_dir.name}: files={len(gt_files)}")
        for gt_path in gt_files:
            pred_path = predicted_dir / gt_path.name
            if not pred_path.exists():
                raise FileNotFoundError(f"missing prediction for {gt_path.name} in {predicted_dir}")
            file_row, subtree_records, confusion = _evaluate_file(batch_dir.name, gt_path, pred_path)
            file_rows.append(file_row)
            subtree_rows.extend(subtree_records)
            total_confusion.update(confusion)
            batch_confusions[batch_dir.name].update(confusion)

    per_file_df = pd.DataFrame(file_rows).sort_values(["batch", "file_name"]).reset_index(drop=True)
    subtree_df = pd.DataFrame(subtree_rows)
    if subtree_df.empty:
        subtree_df = pd.DataFrame(columns=[
            "batch", "file_name", "subtree_root_id", "subtree_nodes", "subtree_supported_gt_nodes",
            "gt_subtree_label", "gt_subtree_label_name", "pred_subtree_label", "pred_subtree_label_name",
            "root_gt_label", "root_pred_label", "correct",
        ])
    else:
        subtree_df = subtree_df.sort_values(["batch", "file_name", "subtree_root_id"]).reset_index(drop=True)

    per_class_df = _per_class_from_confusion(total_confusion)
    batch_df = _batch_summary(per_file_df, subtree_df, batch_confusions)
    confusion_total = int(sum(total_confusion.values()))

    per_file_path = output_dir / "per_file_metrics.csv"
    per_class_path = output_dir / "per_class_metrics.csv"
    batch_path = output_dir / "batch_summary.csv"
    subtree_path = output_dir / "subtree_metrics.csv"
    confusion_path = output_dir / "confusion_matrix.csv"
    report_path = output_dir / "evaluation_report.md"

    per_file_df.to_csv(per_file_path, index=False)
    per_class_df.to_csv(per_class_path, index=False)
    batch_df.to_csv(batch_path, index=False)
    subtree_df.to_csv(subtree_path, index=False)
    _write_confusion_csv(confusion_path, total_confusion)
    _write_report(
        report_path,
        dataset_root=dataset_root,
        per_file_df=per_file_df,
        per_class_df=per_class_df,
        batch_df=batch_df,
        subtree_df=subtree_df,
        confusion_counts=total_confusion,
        confusion_total=confusion_total,
    )

    total_nodes = int(per_file_df["nodes_evaluated"].sum())
    correct_nodes = int(per_file_df["correct_nodes"].sum())
    pooled_node_accuracy = _safe_div(correct_nodes, total_nodes)
    class_present = per_class_df.loc[per_class_df["support"] > 0, "f1"].tolist()
    pooled_macro_f1 = _safe_div(sum(class_present), len(class_present)) if class_present else 0.0
    subtree_total = int(len(subtree_df))
    subtree_correct = int(subtree_df["correct"].sum()) if subtree_total else 0
    subtree_accuracy = _safe_div(subtree_correct, subtree_total)

    print(f"files_evaluated: {len(per_file_df)}")
    print(f"nodes_evaluated: {total_nodes}")
    print(f"pooled_node_accuracy: {pooled_node_accuracy:.6f}")
    print(f"pooled_macro_f1: {pooled_macro_f1:.6f}")
    print(f"median_file_accuracy: {_median(per_file_df['node_accuracy'].tolist()):.6f}")
    print(f"median_file_macro_f1: {_median(per_file_df['macro_f1'].tolist()):.6f}")
    print(f"subtree_accuracy: {subtree_accuracy:.6f} ({subtree_correct}/{subtree_total})")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
