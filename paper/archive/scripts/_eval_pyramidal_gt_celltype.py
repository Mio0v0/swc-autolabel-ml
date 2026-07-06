#!/usr/bin/env python3
"""Comprehensive pyramidal-only evaluation with GT cell-type override.

Loads a trained pyramidal-only model dir, runs the full v12 pipeline on
the model's held-out pyramidal test set with `override_cell_type="pyramidal"`
so Stage 1 is BYPASSED -- every cell goes straight to the pyramidal
Stage 2 + Stage 3 (GNN + topology refinement) regardless of what Stage 1
would have predicted. This isolates Stage 2+3 performance from any
Stage 1 propagation errors.

Reports the canonical baseline metric pack:
    Per-class precision / recall / F1 / support
    4x4 confusion matrix
    Per-cell distributions: mean, P10, P25, P50 (median), P75, P90, std
        for per-cell accuracy AND per-cell neurite_macro_F1
    Top-N worst cells by F1 (with their reasons)
    Per-class F1 distributions (apical F1 mean / P10 etc.)

Usage:
    python -m paper._eval_pyramidal_gt_celltype
        --model-dir paper/models/v12_pyramidal_only_seed2024_clean
        [--top-worst 30]                  # how many worst-F1 cells to list
        [--out paper/results/pyramidal_baseline_report]

Output:
    {--out}.json    full metric pack
    {--out}.csv     per-cell results (file, gt, n_nodes, F1, per-class F1, etc.)
    {--out}.txt     pretty-printed report
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.features import parse_swc                                   # noqa: E402
from hybrid.pipeline import run_pipeline_on_nodes                       # noqa: E402
from hybrid.evaluate import per_cell_neurite_f1                         # noqa: E402
from paper.gnn_inference import load_gnn                                # noqa: E402
from paper.gnn_branch3_inference import load_branch3                    # noqa: E402

DATA_DIR = ROOT / "data" / "v12_uncurated"
LABEL_NAMES = {1: "soma", 2: "axon", 3: "basal", 4: "apical"}


def _per_class_metrics(gt: list[int], pred: list[int]) -> dict[str, dict]:
    """Return per-class precision/recall/F1/support, scored 1-vs-rest.

    Only classes present in GT get reported (matches v11 convention).
    """
    out: dict[str, dict] = {}
    y_t = np.asarray(gt, dtype=int)
    y_p = np.asarray(pred, dtype=int)
    for cls in (1, 2, 3, 4):
        sup = int((y_t == cls).sum())
        if sup == 0: continue
        yt = (y_t == cls).astype(int)
        yp = (y_p == cls).astype(int)
        out[LABEL_NAMES[cls]] = {
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall":    float(recall_score(yt, yp, zero_division=0)),
            "f1":        float(f1_score(yt, yp, zero_division=0)),
            "support":   sup,
        }
    return out


def _aggregate_per_class(rows: list[dict]) -> dict[str, dict]:
    """Pool all GT and predictions across cells for corpus-level per-class metrics."""
    all_gt: list[int] = []
    all_pred: list[int] = []
    for r in rows:
        all_gt.extend(r["_gt"])
        all_pred.extend(r["_pred"])
    return _per_class_metrics(all_gt, all_pred)


def _confusion_matrix(gt: list[int], pred: list[int]) -> dict[str, dict[str, int]]:
    """Pooled confusion matrix as {true: {pred: count}}."""
    classes = [1, 2, 3, 4]
    M: dict[str, dict[str, int]] = {
        LABEL_NAMES[c]: {LABEL_NAMES[c2]: 0 for c2 in classes} for c in classes
    }
    for g, p in zip(gt, pred):
        if g in LABEL_NAMES and p in LABEL_NAMES:
            M[LABEL_NAMES[g]][LABEL_NAMES[p]] += 1
    return M


def _distributions(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {k: 0.0 for k in ("mean","std","min","P10","P25","P50","P75","P90","max")}
    return {
        "mean":   float(a.mean()),
        "std":    float(a.std()),
        "min":    float(a.min()),
        "P10":    float(np.percentile(a, 10)),
        "P25":    float(np.percentile(a, 25)),
        "P50":    float(np.percentile(a, 50)),
        "P75":    float(np.percentile(a, 75)),
        "P90":    float(np.percentile(a, 90)),
        "max":    float(a.max()),
    }


def _load_split(model_dir: Path) -> list[Path]:
    """Return the model's held-out pyramidal test cells as full paths."""
    sp = json.loads((model_dir / "train_test_split.json").read_text(encoding="utf-8"))
    test = sp["test"].get("pyramidal", [])
    paths: list[Path] = []
    for fn in test:
        p = DATA_DIR / "pyramidal" / "swc" / fn
        if p.is_file():
            paths.append(p)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--top-worst", type=int, default=30,
                    help="Number of worst-F1 cells to list explicitly (default 30)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output stem; defaults to paper/results/<model_dir_name>_eval")
    ap.add_argument("--branch3-ckpt", type=Path, default=None,
                    help="Optional 3-class pyramidal branch rescue checkpoint. "
                         "Defaults to <model-dir>/gnn_branch3_rescue.pt if present.")
    ap.add_argument("--branch3-gate", type=Path, default=None,
                    help="Optional learned Branch3 accept/abstain gate joblib.")
    ap.add_argument("--no-branch3", action="store_true",
                    help="Disable automatic Branch3 rescue loading.")
    args = ap.parse_args()

    md = args.model_dir.resolve()  # absolute, so .relative_to(ROOT) works
    if not md.is_dir():
        raise SystemExit(f"MISSING: {md}")
    s1_path = md / "cell_type_classifier.pkl"
    s2_path = md / "branch_classifier.pkl"
    gnn_path = md / "gnn_apical_basal.pt"
    for p in (s1_path, s2_path, gnn_path):
        if not p.is_file():
            raise SystemExit(f"MISSING: {p}")

    out_stem = (args.out or (ROOT / "paper" / "results" / f"{md.name}_eval")).resolve()
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    # Test files
    test_files = _load_split(md)
    print(f"=== Pyramidal-only comprehensive eval (Stage 1 BYPASSED) ===")
    print(f"  model:      {md.relative_to(ROOT)}")
    print(f"  test cells: {len(test_files)}")
    print(f"  out stem:   {out_stem.relative_to(ROOT)}")
    print()

    gnn_state = load_gnn(gnn_path)
    branch3_state = None
    branch3_path = args.branch3_ckpt
    if branch3_path is None and not args.no_branch3:
        auto_branch3 = md / "gnn_branch3_rescue.pt"
        if auto_branch3.is_file():
            branch3_path = auto_branch3
    if branch3_path is not None:
        branch3_path = branch3_path.resolve()
        if not branch3_path.is_file():
            raise SystemExit(f"MISSING: {branch3_path}")
        branch3_state = load_branch3(branch3_path, gate_path=args.branch3_gate)
        print(f"  branch3:    {branch3_path.relative_to(ROOT)}")
        if args.branch3_gate is not None:
            print(f"  b3 gate:    {args.branch3_gate.resolve().relative_to(ROOT)}")

    rows: list[dict] = []
    n_failed = 0
    t0 = time.perf_counter()
    for i, p in enumerate(test_files):
        try:
            nodes = parse_swc(p)
            if not nodes:
                n_failed += 1; continue
            gt = [n.type for n in nodes]
        except Exception as exc:
            print(f"  WARN parse {p.name}: {exc}")
            n_failed += 1; continue
        try:
            pr = run_pipeline_on_nodes(
                nodes, file_path="",
                stage1_model=s1_path, stage2_model=s2_path,
                gnn_state=gnn_state, branch3_state=branch3_state,
                use_subtree_stage2=True,
                override_cell_type="pyramidal",  # BYPASS Stage 1
            )
        except Exception as exc:
            print(f"  WARN pipeline {p.name}: {exc}")
            n_failed += 1; continue

        pred = list(pr.node_labels)
        n_correct = sum(1 for g, q in zip(gt, pred) if g == q)
        acc = n_correct / max(1, len(gt))
        f1 = per_cell_neurite_f1(gt, pred, "pyramidal")
        per_cls = _per_class_metrics(gt, pred)
        rows.append({
            "file":        p.name,
            "n_nodes":     len(gt),
            "n_apical_gt": int(sum(1 for g in gt if g == 4)),
            "n_basal_gt":  int(sum(1 for g in gt if g == 3)),
            "n_axon_gt":   int(sum(1 for g in gt if g == 2)),
            "n_apical_pred": int(sum(1 for q in pred if q == 4)),
            "n_basal_pred":  int(sum(1 for q in pred if q == 3)),
            "n_axon_pred":   int(sum(1 for q in pred if q == 2)),
            "accuracy":    acc,
            "F1_neurite":  f1,
            "axon_F1":     per_cls.get("axon", {}).get("f1"),
            "basal_F1":    per_cls.get("basal", {}).get("f1"),
            "apical_F1":   per_cls.get("apical", {}).get("f1"),
            "soma_F1":     per_cls.get("soma", {}).get("f1"),
            "_gt":         gt,
            "_pred":       pred,
        })
        if (i + 1) % 200 == 0:
            elapsed = (time.perf_counter() - t0) / 60.0
            eta = elapsed * (len(test_files) - i - 1) / max(1, i + 1)
            print(f"  ... {i+1}/{len(test_files)}  ({elapsed:.1f} min, ETA {eta:.0f} min)", flush=True)

    elapsed_min = (time.perf_counter() - t0) / 60.0
    print(f"\nInference done: {len(rows)} scored, {n_failed} failed  ({elapsed_min:.1f} min)")

    # ---------------------------------------------------------------------
    # Aggregate metrics
    # ---------------------------------------------------------------------
    corpus_per_cls = _aggregate_per_class(rows)
    # Pooled accuracy + neurite_macro_f1
    all_gt   = [g for r in rows for g in r["_gt"]]
    all_pred = [p for r in rows for p in r["_pred"]]
    corpus_acc = sum(1 for g, p in zip(all_gt, all_pred) if g == p) / max(1, len(all_gt))
    confusion = _confusion_matrix(all_gt, all_pred)
    # Macro / neurite macro F1 corpus-level
    f1s_present = [v["f1"] for v in corpus_per_cls.values()]
    macro_f1_corpus = float(np.mean(f1s_present)) if f1s_present else 0.0
    f1s_neurite_present = [
        v["f1"] for k, v in corpus_per_cls.items() if k != "soma"
    ]
    neurite_macro_f1_corpus = float(np.mean(f1s_neurite_present)) if f1s_neurite_present else 0.0

    # Per-cell distributions
    acc_dist   = _distributions([r["accuracy"]    for r in rows])
    f1_dist    = _distributions([r["F1_neurite"]  for r in rows])
    # Per-class F1 distributions (only cells where class is in GT)
    perclass_dist: dict[str, dict] = {}
    for cls in ("axon_F1", "basal_F1", "apical_F1"):
        vals = [r[cls] for r in rows if r[cls] is not None]
        perclass_dist[cls] = _distributions(vals)

    # Worst-N cells by F1
    rows_sorted = sorted(rows, key=lambda r: r["F1_neurite"])
    worst = rows_sorted[: args.top_worst]

    # ---------------------------------------------------------------------
    # Build report
    # ---------------------------------------------------------------------
    report = {
        "model_dir":         str(md.relative_to(ROOT)),
        "n_test":            len(rows),
        "n_failed":          n_failed,
        "inference_min":     elapsed_min,
        "convention":        "v11 (macro F1 over GT-present classes only, soma excluded in neurite metric)",
        "note":              (
            "Stage 1 BYPASSED via override_cell_type='pyramidal'. "
            "Branch3 rescue enabled." if branch3_state is not None
            else "Stage 1 BYPASSED via override_cell_type='pyramidal'. Tests Stage 2+3 quality only."
        ),
        "corpus": {
            "accuracy":           corpus_acc,
            "macro_f1":           macro_f1_corpus,
            "neurite_macro_f1":   neurite_macro_f1_corpus,
            "per_class":          corpus_per_cls,
            "confusion":          confusion,
        },
        "per_cell": {
            "accuracy":       acc_dist,
            "F1_neurite":     f1_dist,
            "per_class_F1":   perclass_dist,
        },
        "top_worst_by_F1":   [
            {k: v for k, v in r.items() if not k.startswith("_")} for r in worst
        ],
    }
    (out_stem.with_suffix(".json")).write_text(json.dumps(report, indent=2), encoding="utf-8")

    # CSV: per-cell flat
    csv_path = out_stem.with_suffix(".csv")
    fieldnames = ["file","n_nodes","n_apical_gt","n_basal_gt","n_axon_gt",
                  "n_apical_pred","n_basal_pred","n_axon_pred",
                  "accuracy","F1_neurite","axon_F1","basal_F1","apical_F1","soma_F1"]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_sorted:
            row = {k: r[k] for k in fieldnames}
            for cls in ("axon_F1","basal_F1","apical_F1","soma_F1"):
                if row[cls] is None: row[cls] = ""
                elif isinstance(row[cls], float): row[cls] = f"{row[cls]:.4f}"
            row["accuracy"]   = f"{row['accuracy']:.4f}"
            row["F1_neurite"] = f"{row['F1_neurite']:.4f}"
            w.writerow(row)

    # Pretty TXT
    txt_path = out_stem.with_suffix(".txt")
    L: list[str] = []
    P = L.append
    P("=" * 80)
    title = (
        "PYRAMIDAL-ONLY + BRANCH3 RESCUE (Stage 1 bypassed, GT cell-type override)"
        if branch3_state is not None
        else "PYRAMIDAL-ONLY BASELINE (Stage 1 bypassed, GT cell-type override)"
    )
    P(f"  {title}")
    P(f"  Model:    {md.name}")
    P(f"  Cells:    {len(rows)} test pyramidals  (corpus: cleaned 11,862 -> 8,488 pyramidals, 50/50 split)")
    P("=" * 80)
    P("")
    P("== CORPUS-LEVEL ==")
    P(f"  Per-node accuracy            = {corpus_acc:.4f}")
    P(f"  Macro F1 (4 classes)         = {macro_f1_corpus:.4f}")
    P(f"  Neurite macro F1 (3 classes) = {neurite_macro_f1_corpus:.4f}")
    P("")
    P("  Per-class P/R/F1:")
    P(f"    {'class':<12} {'P':>8} {'R':>8} {'F1':>8} {'support':>14}")
    for cls in ("soma","axon","basal","apical"):
        if cls in corpus_per_cls:
            d = corpus_per_cls[cls]
            P(f"    {cls:<12} {d['precision']:>8.4f} {d['recall']:>8.4f} {d['f1']:>8.4f} {d['support']:>14,}")
    P("")
    P("== CONFUSION MATRIX (rows = GT, cols = PRED) ==")
    cls_order = ["soma","axon","basal","apical"]
    P(f"    {'':>10}" + "".join(f"{c:>12}" for c in cls_order))
    for gt_c in cls_order:
        row = confusion.get(gt_c, {})
        tot = sum(row.values())
        cells = "".join(f"{row.get(p,0):>12,}" for p in cls_order)
        pct = f"  ({tot:,} GT)" if tot else ""
        P(f"    {gt_c:>10}" + cells + pct)
    P("")
    P("== PER-CELL DISTRIBUTIONS ==")
    P(f"  {'metric':<24} {'mean':>8} {'std':>8} {'min':>8} {'P10':>8} {'P25':>8} {'P50':>8} {'P75':>8} {'P90':>8}")
    for name, d in [
        ("accuracy",                 acc_dist),
        ("F1_neurite (overall)",     f1_dist),
        ("axon F1",                  perclass_dist["axon_F1"]),
        ("basal F1",                 perclass_dist["basal_F1"]),
        ("apical F1",                perclass_dist["apical_F1"]),
    ]:
        P(f"  {name:<24} "
          f"{d['mean']:>8.4f} {d['std']:>8.4f} {d['min']:>8.4f} "
          f"{d['P10']:>8.4f} {d['P25']:>8.4f} {d['P50']:>8.4f} "
          f"{d['P75']:>8.4f} {d['P90']:>8.4f}")
    P("")
    P(f"== TOP {len(worst)} WORST CELLS BY F1 ==")
    P(f"  {'file':<58} {'n':>6} {'F1':>6} {'acc':>6} {'axF1':>6} {'baF1':>6} {'apF1':>6} {'GT a/b/p':>14} {'PRED a/b/p':>14}")
    for r in worst:
        gt_abp  = f"{r['n_axon_gt']}/{r['n_basal_gt']}/{r['n_apical_gt']}"
        pr_abp  = f"{r['n_axon_pred']}/{r['n_basal_pred']}/{r['n_apical_pred']}"
        def _fmt(x): return f"{x:.3f}" if isinstance(x, (int, float)) else "-"
        P(f"  {r['file'][:58]:<58} {r['n_nodes']:>6} "
          f"{_fmt(r['F1_neurite']):>6} {_fmt(r['accuracy']):>6} "
          f"{_fmt(r['axon_F1']):>6} {_fmt(r['basal_F1']):>6} {_fmt(r['apical_F1']):>6} "
          f"{gt_abp:>14} {pr_abp:>14}")
    P("")

    body = "\n".join(L)
    txt_path.write_text(body, encoding="utf-8")
    print()
    print(body)
    print()
    print(f"Wrote:")
    print(f"  {out_stem.with_suffix('.json')}")
    print(f"  {out_stem.with_suffix('.csv')}")
    print(f"  {out_stem.with_suffix('.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
