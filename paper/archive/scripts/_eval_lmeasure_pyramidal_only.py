#!/usr/bin/env python3
"""L-Measure-RF baseline trained + evaluated on pyramidal-only split.

Reuses the train_test_split.json from a pyramidal-only model directory
(so we evaluate the L-Measure baseline on the EXACT same 4225 train /
4263 test pyramidals that v12 Stage 2+3 sees). Produces the same
comprehensive metric pack as `_eval_pyramidal_gt_celltype.py`:
    Per-class P / R / F1 / support
    4x4 confusion matrix
    Per-cell distributions (mean / P10 / P25 / P50 / P75 / P90) for
        accuracy and neurite-macro-F1 and per-class F1
    Top-N worst cells by F1

This is the apples-to-apples L-Measure vs v12 (Stage 2+3) comparison
on the cleaned 11862-cell corpus.

Usage:
    python -m paper._eval_lmeasure_pyramidal_only
        --split-from-model paper/models/v12_pyramidal_only_seed2024_clean
        [--top-worst 30]
        [--out paper/results/lmeasure_pyramidal_only_eval]

Wall time: ~25-40 min (CPU only -- feature extraction + sklearn RF).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper.external_baselines import (                                  # noqa: E402
    _train_lmeasure_clf, _make_lmeasure_predictor,
)
from hybrid.features import parse_swc                                   # noqa: E402
from hybrid.evaluate import per_cell_neurite_f1                         # noqa: E402

DATA_DIR = ROOT / "data" / "v12_uncurated"
LABEL_NAMES = {1: "soma", 2: "axon", 3: "basal", 4: "apical"}


def _per_class_metrics(gt: list[int], pred: list[int]) -> dict[str, dict]:
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


def _confusion(gt: list[int], pred: list[int]) -> dict[str, dict[str, int]]:
    classes = [1, 2, 3, 4]
    M = {LABEL_NAMES[c]: {LABEL_NAMES[c2]: 0 for c2 in classes} for c in classes}
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-from-model", type=Path, required=True,
                    help="A model dir containing train_test_split.json. We use its "
                         "pyramidal train/test lists to ensure apples-to-apples.")
    ap.add_argument("--top-worst", type=int, default=30)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42,
                    help="RandomForest training seed (default 42)")
    args = ap.parse_args()

    # Resolve to absolute path so relative_to(ROOT) works whatever the CLI passed.
    args.split_from_model = args.split_from_model.resolve()
    if not (args.split_from_model / "train_test_split.json").is_file():
        raise SystemExit(f"Missing split JSON in {args.split_from_model}")
    sp = json.loads((args.split_from_model / "train_test_split.json").read_text(encoding="utf-8"))

    train_pyr = [DATA_DIR / "pyramidal" / "swc" / fn for fn in sp["train"].get("pyramidal", [])]
    test_pyr  = [DATA_DIR / "pyramidal" / "swc" / fn for fn in sp["test"].get("pyramidal", [])]
    train_pyr = [p for p in train_pyr if p.is_file()]
    test_pyr  = [p for p in test_pyr  if p.is_file()]
    print(f"=== L-Measure-RF on pyramidal-only split ===")
    print(f"  Split from: {args.split_from_model.relative_to(ROOT)}")
    print(f"  Train: {len(train_pyr)} pyramidals")
    print(f"  Test : {len(test_pyr)} pyramidals")
    print(f"  Seed:  {args.seed}")
    print()

    # ---- Train L-Measure RF on pyramidals only ----
    train_files_dict = {"pyramidal": train_pyr, "interneuron": []}
    print("[Train] Fitting L-Measure-RF...")
    t0 = time.perf_counter()
    clf = _train_lmeasure_clf(train_files_dict, seed=args.seed)
    predict_fn = _make_lmeasure_predictor(clf)
    print(f"  trained in {(time.perf_counter()-t0)/60:.1f} min\n")

    # ---- Predict on test pyramidals ----
    print(f"[Predict] Running on {len(test_pyr)} test pyramidals...")
    rows: list[dict] = []
    n_failed = 0
    t1 = time.perf_counter()
    for i, p in enumerate(test_pyr):
        try:
            nodes = parse_swc(p)
            if not nodes:
                n_failed += 1; continue
            gt = [n.type for n in nodes]
        except Exception as exc:
            print(f"  WARN parse {p.name}: {exc}")
            n_failed += 1; continue
        try:
            pred = predict_fn(nodes, "pyramidal")  # baseline takes GT cell-type
        except Exception as exc:
            print(f"  WARN predict {p.name}: {exc}")
            n_failed += 1; continue
        if isinstance(pred, np.ndarray):
            pred = pred.tolist()
        n_correct = sum(1 for g, q in zip(gt, pred) if g == q)
        acc = n_correct / max(1, len(gt))
        f1 = per_cell_neurite_f1(gt, pred, "pyramidal")
        per_cls = _per_class_metrics(gt, pred)
        rows.append({
            "file":          p.name,
            "n_nodes":       len(gt),
            "n_apical_gt":   int(sum(1 for g in gt if g == 4)),
            "n_basal_gt":    int(sum(1 for g in gt if g == 3)),
            "n_axon_gt":     int(sum(1 for g in gt if g == 2)),
            "n_apical_pred": int(sum(1 for q in pred if q == 4)),
            "n_basal_pred":  int(sum(1 for q in pred if q == 3)),
            "n_axon_pred":   int(sum(1 for q in pred if q == 2)),
            "accuracy":      acc,
            "F1_neurite":    f1,
            "axon_F1":       per_cls.get("axon", {}).get("f1"),
            "basal_F1":      per_cls.get("basal", {}).get("f1"),
            "apical_F1":     per_cls.get("apical", {}).get("f1"),
            "soma_F1":       per_cls.get("soma", {}).get("f1"),
            "_gt":           gt,
            "_pred":         pred,
        })
        if (i + 1) % 500 == 0:
            elapsed = (time.perf_counter() - t1) / 60.0
            eta = elapsed * (len(test_pyr) - i - 1) / max(1, i + 1)
            print(f"  ... {i+1}/{len(test_pyr)}  ({elapsed:.1f} min, ETA {eta:.0f} min)", flush=True)

    elapsed_min = (time.perf_counter() - t1) / 60.0
    print(f"\nPredict done: {len(rows)} scored, {n_failed} failed  ({elapsed_min:.1f} min)")

    # ---- Aggregate ----
    all_gt   = [g for r in rows for g in r["_gt"]]
    all_pred = [p for r in rows for p in r["_pred"]]
    corpus_per_cls = _per_class_metrics(all_gt, all_pred)
    corpus_acc = sum(1 for g, p in zip(all_gt, all_pred) if g == p) / max(1, len(all_gt))
    confusion = _confusion(all_gt, all_pred)
    f1s_present = [v["f1"] for v in corpus_per_cls.values()]
    macro_f1_corpus = float(np.mean(f1s_present)) if f1s_present else 0.0
    neurite_macro_f1_corpus = float(np.mean([
        v["f1"] for k, v in corpus_per_cls.items() if k != "soma"
    ])) if len(corpus_per_cls) > 1 else 0.0
    acc_dist = _distributions([r["accuracy"]   for r in rows])
    f1_dist  = _distributions([r["F1_neurite"] for r in rows])
    perclass_dist: dict[str, dict] = {}
    for cls in ("axon_F1","basal_F1","apical_F1"):
        vals = [r[cls] for r in rows if r[cls] is not None]
        perclass_dist[cls] = _distributions(vals)

    rows_sorted = sorted(rows, key=lambda r: r["F1_neurite"])
    worst = rows_sorted[: args.top_worst]

    out_stem = args.out or (ROOT / "paper" / "results" / "lmeasure_pyramidal_only_eval")
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "model":              "lmeasure_rf (pyramidal-only training, GT cell-type override)",
        "split_from_model":   str(args.split_from_model.relative_to(ROOT)),
        "n_train":            len(train_pyr),
        "n_test":             len(rows),
        "n_failed":           n_failed,
        "predict_min":        elapsed_min,
        "corpus": {
            "accuracy":          corpus_acc,
            "macro_f1":          macro_f1_corpus,
            "neurite_macro_f1":  neurite_macro_f1_corpus,
            "per_class":         corpus_per_cls,
            "confusion":         confusion,
        },
        "per_cell": {
            "accuracy":      acc_dist,
            "F1_neurite":    f1_dist,
            "per_class_F1":  perclass_dist,
        },
        "top_worst_by_F1": [{k: v for k, v in r.items() if not k.startswith("_")} for r in worst],
    }
    out_stem.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fieldnames = ["file","n_nodes","n_apical_gt","n_basal_gt","n_axon_gt",
                  "n_apical_pred","n_basal_pred","n_axon_pred",
                  "accuracy","F1_neurite","axon_F1","basal_F1","apical_F1","soma_F1"]
    with out_stem.with_suffix(".csv").open("w", encoding="utf-8", newline="") as fh:
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
    L: list[str] = []
    P = L.append
    P("=" * 80)
    P(f"  L-MEASURE-RF BASELINE (pyramidal-only training + eval)")
    P(f"  Split from: {args.split_from_model.name}")
    P(f"  Train: {len(train_pyr)} pyramidals    Test: {len(rows)} pyramidals")
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
        gt_abp = f"{r['n_axon_gt']}/{r['n_basal_gt']}/{r['n_apical_gt']}"
        pr_abp = f"{r['n_axon_pred']}/{r['n_basal_pred']}/{r['n_apical_pred']}"
        def _f(x): return f"{x:.3f}" if isinstance(x, (int, float)) else "-"
        P(f"  {r['file'][:58]:<58} {r['n_nodes']:>6} "
          f"{_f(r['F1_neurite']):>6} {_f(r['accuracy']):>6} "
          f"{_f(r['axon_F1']):>6} {_f(r['basal_F1']):>6} {_f(r['apical_F1']):>6} "
          f"{gt_abp:>14} {pr_abp:>14}")
    P("")

    body = "\n".join(L)
    out_stem.with_suffix(".txt").write_text(body, encoding="utf-8")
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
