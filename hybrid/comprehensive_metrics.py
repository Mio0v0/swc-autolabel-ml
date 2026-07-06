"""Single source of truth for the full evaluation metric pack.

Used by every evaluation script (`paper/_eval_per_seed_own_test.py`,
`paper/_eval_baselines_on_v12.py` via `paper/baselines.py`, and the
v12 ensemble eval) so all comparisons emit the IDENTICAL JSON shape
and use the IDENTICAL metric definitions.

The output dict layout:

    {
      "method": str,
      "n_test_cells": int,
      "n_test_nodes": int,
      "inference_min": float | None,
      "convention": "v11 (macro F1 over GT-present classes only)",

      "corpus": {                                # pooled across all test nodes
        "accuracy": float,
        "macro_f1": float,                       # macro over all 4 classes
        "neurite_macro_f1": float,               # macro over {axon, basal, apical}
        "per_class": {                           # for each of {soma, axon, basal/dendrite, apical}
          "<class>": {"precision","recall","f1","support"}
        },
        "confusion": {                           # gt_name -> pred_name -> count
          "<gt>": {"<pred>": int}
        },
      },

      "per_cell": {                              # one F1 per cell, summarized
        "accuracy":         {"n","mean","median","p10","p25","p75","p90","min"},
        "neurite_macro_f1": {...},
        "per_class_f1": {                        # one distribution per class,
          "<class>": {...},                      # over cells with that class in GT
        },
      },

      "by_cell_type": {                          # same shape recursed by cell type
        "pyramidal":   { "n_cells","n_nodes","corpus","per_cell" },
        "interneuron": { "n_cells","n_nodes","corpus","per_cell" },
      },
    }
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .evaluate import _compute_metrics, per_cell_per_class_f1, per_cell_neurite_f1
from .cell_type_detector import CELL_TYPE_LABEL_SETS

CLASS_NAMES = ("soma", "axon", "basal/dendrite", "apical")
CLASS_IDS = (1, 2, 3, 4)
LABEL_NAME = {1: "soma", 2: "axon", 3: "basal/dendrite", 4: "apical"}


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _dist(values: Iterable[float | None]) -> dict:
    """Distribution summary; Nones are dropped (signal: class absent in cell)."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0,
                "p10": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0, "min": 0.0}
    return {
        "n":      int(arr.size),
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10":    float(np.percentile(arr, 10)),
        "p25":    float(np.percentile(arr, 25)),
        "p75":    float(np.percentile(arr, 75)),
        "p90":    float(np.percentile(arr, 90)),
        "min":    float(np.min(arr)),
    }


def _per_class_dict_from_metrics(metrics: dict) -> dict:
    """Pull per-class precision/recall/F1/support from `_compute_metrics`'s output,
    rounded floats for compact JSON."""
    out = {}
    for name in CLASS_NAMES:
        info = metrics.get("per_label", {}).get(name, {})
        out[name] = {
            "precision": float(info.get("precision", 0.0)),
            "recall":    float(info.get("recall", 0.0)),
            "f1":        float(info.get("f1", 0.0)),
            "support":   int(info.get("support", 0)),
        }
    return out


def _confusion_dict_from_metrics(metrics: dict) -> dict:
    """Normalize confusion-matrix dict to {gt_name: {pred_name: count}} with all
    four classes present (zeros filled in)."""
    raw = metrics.get("confusion", {})
    out = {g: {p: 0 for p in CLASS_NAMES} for g in CLASS_NAMES}
    for g_name, row in raw.items():
        for p_name, n in row.items():
            if g_name in out and p_name in out[g_name]:
                out[g_name][p_name] = int(n)
    return out


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def build_full_report(
    method: str,
    gt_pool: list[int],
    pred_pool: list[int],
    cell_records: list[dict],
    inference_min: float | None = None,
    stage1_records: list[dict] | None = None,
) -> dict:
    """Build the canonical comprehensive metric report.

    Parameters
    ----------
    method : str
        Name of the method (eg "v12_seed42", "lmeasure_rf").
    gt_pool, pred_pool : list[int]
        ALL test nodes concatenated. Used for corpus-level metrics.
    cell_records : list of dicts, one per test cell, each with keys:
        - "cell_type":   "pyramidal" | "interneuron"
        - "n_nodes":     int
        - "gt":          list[int]   per-node GT labels
        - "pred":        list[int]   per-node predicted labels
    inference_min : float, optional
        Wall-clock for inference (None for baselines that include training time).
    """
    n_cells = len(cell_records)
    n_nodes = len(gt_pool)

    # ---- Corpus-level (over all classes 1-4) ----
    corpus_metrics = _compute_metrics(gt_pool, pred_pool, set(CLASS_IDS))
    corpus_block = {
        "accuracy":         float(corpus_metrics.get("accuracy", 0.0)),
        "macro_f1":         float(corpus_metrics.get("macro_f1", 0.0)),
        "neurite_macro_f1": float(corpus_metrics.get("neurite_macro_f1", 0.0)),
        "per_class":        _per_class_dict_from_metrics(corpus_metrics),
        "confusion":        _confusion_dict_from_metrics(corpus_metrics),
    }

    # ---- Per-cell distributions ----
    pc_acc_vals    = []
    pc_neuritef1_vals = []
    pc_per_class_vals = {n: [] for n in CLASS_NAMES}
    for rec in cell_records:
        # Per-cell accuracy = fraction of nodes correctly labeled
        if rec["n_nodes"] > 0:
            n_correct = sum(1 for g, p in zip(rec["gt"], rec["pred"]) if g == p)
            pc_acc_vals.append(n_correct / rec["n_nodes"])
        else:
            pc_acc_vals.append(0.0)
        # Per-cell neurite F1 (v11 convention)
        pc_neuritef1_vals.append(
            per_cell_neurite_f1(rec["gt"], rec["pred"], rec["cell_type"])
        )
        # Per-class F1 per cell (None for classes absent in this cell's GT)
        per_class = per_cell_per_class_f1(rec["gt"], rec["pred"], rec["cell_type"])
        for cname in CLASS_NAMES:
            pc_per_class_vals[cname].append(per_class.get(cname))

    per_cell_block = {
        "accuracy":         _dist(pc_acc_vals),
        "neurite_macro_f1": _dist(pc_neuritef1_vals),
        "per_class_f1":     {n: _dist(pc_per_class_vals[n]) for n in CLASS_NAMES},
    }

    # ---- Per-cell-type breakdown ----
    by_ct: dict[str, dict] = {}
    for ct in ("pyramidal", "interneuron"):
        ct_records = [r for r in cell_records if r["cell_type"] == ct]
        if not ct_records:
            continue
        # Pool nodes for this CT
        ct_gt_pool: list[int] = []
        ct_pred_pool: list[int] = []
        for r in ct_records:
            ct_gt_pool.extend(r["gt"])
            ct_pred_pool.extend(r["pred"])
        valid = set(CELL_TYPE_LABEL_SETS.get(ct, {1, 2, 3}))
        ct_corpus = _compute_metrics(ct_gt_pool, ct_pred_pool, valid)

        ct_pc_acc, ct_pc_neuritef1 = [], []
        ct_per_class_vals = {n: [] for n in CLASS_NAMES}
        for rec in ct_records:
            if rec["n_nodes"] > 0:
                ct_pc_acc.append(
                    sum(1 for g, p in zip(rec["gt"], rec["pred"]) if g == p) / rec["n_nodes"]
                )
            else:
                ct_pc_acc.append(0.0)
            ct_pc_neuritef1.append(
                per_cell_neurite_f1(rec["gt"], rec["pred"], rec["cell_type"])
            )
            per_class = per_cell_per_class_f1(rec["gt"], rec["pred"], rec["cell_type"])
            for cname in CLASS_NAMES:
                ct_per_class_vals[cname].append(per_class.get(cname))

        by_ct[ct] = {
            "n_cells":  len(ct_records),
            "n_nodes":  len(ct_gt_pool),
            "corpus": {
                "accuracy":         float(ct_corpus.get("accuracy", 0.0)),
                "macro_f1":         float(ct_corpus.get("macro_f1", 0.0)),
                "neurite_macro_f1": float(ct_corpus.get("neurite_macro_f1", 0.0)),
                "per_class":        _per_class_dict_from_metrics(ct_corpus),
                "confusion":        _confusion_dict_from_metrics(ct_corpus),
            },
            "per_cell": {
                "accuracy":         _dist(ct_pc_acc),
                "neurite_macro_f1": _dist(ct_pc_neuritef1),
                "per_class_f1":     {n: _dist(ct_per_class_vals[n]) for n in CLASS_NAMES},
            },
        }

    # ---- Stage 1 accuracy + confusion (optional; v12 only) ----
    stage1_block = None
    if stage1_records:
        n_s1 = len(stage1_records)
        n_correct = sum(1 for r in stage1_records if r["stage1_pred"] == r["cell_type_gt"])
        s1_cm: dict = {ct: {ct2: 0 for ct2 in ("pyramidal", "interneuron")}
                        for ct in ("pyramidal", "interneuron")}
        for r in stage1_records:
            gt = r["cell_type_gt"]; pr = r["stage1_pred"]
            if gt in s1_cm and pr in s1_cm[gt]:
                s1_cm[gt][pr] += 1
        # Per-cell-type Stage 1 recall (= accuracy on cells of that GT type)
        per_ct_acc = {}
        for ct in ("pyramidal", "interneuron"):
            denom = sum(s1_cm[ct].values())
            per_ct_acc[ct] = {
                "n_gt_cells":      denom,
                "n_correct":       s1_cm[ct].get(ct, 0),
                "accuracy":        (s1_cm[ct].get(ct, 0) / denom) if denom > 0 else 0.0,
            }
        # Per-cell-type Stage 1 confidence summary
        per_ct_conf = {}
        for ct in ("pyramidal", "interneuron"):
            confs = [r["stage1_conf"] for r in stage1_records if r["cell_type_gt"] == ct]
            per_ct_conf[ct] = _dist(confs)
        stage1_block = {
            "n_cells":             n_s1,
            "accuracy":            n_correct / n_s1 if n_s1 > 0 else 0.0,
            "n_correct":           n_correct,
            "n_wrong":             n_s1 - n_correct,
            "confusion":           s1_cm,         # gt_ct -> pred_ct -> count
            "per_cell_type_recall": per_ct_acc,
            "confidence_distribution_by_gt_type": per_ct_conf,
        }

    out = {
        "method":         method,
        "n_test_cells":   n_cells,
        "n_test_nodes":   n_nodes,
        "inference_min":  inference_min,
        "convention":     "v11 (macro F1 over GT-present classes only)",
        "corpus":         corpus_block,
        "per_cell":       per_cell_block,
        "by_cell_type":   by_ct,
    }
    if stage1_block is not None:
        out["stage1"] = stage1_block
    return out


# -----------------------------------------------------------------------------
# Pretty-print helpers (for console summary tables)
# -----------------------------------------------------------------------------

def print_summary_table(reports: list[dict]) -> None:
    """One-line-per-method side-by-side summary."""
    cols = [
        ("method",                          24, "{:<24}"),
        ("n_test",                           8, "{:>8}"),
        ("acc",                              8, "{:>8.4f}"),
        ("neurite_F1",                      11, "{:>11.4f}"),
        ("axon_F1",                          9, "{:>9.4f}"),
        ("basal_F1",                        10, "{:>10.4f}"),
        ("apical_F1",                       11, "{:>11.4f}"),
        ("pc_acc_mean",                     12, "{:>12.4f}"),
        ("pc_acc_p10",                      12, "{:>12.4f}"),
        ("pc_F1_mean",                      11, "{:>11.4f}"),
        ("pc_F1_p10",                       10, "{:>10.4f}"),
        ("pc_F1_p25",                       10, "{:>10.4f}"),
    ]
    hdr = "  " + " ".join(f"{n:>{w}}" if i else f"{n:<{w}}"
                            for i, (n, w, _) in enumerate(cols))
    print(hdr)
    for r in reports:
        row = [
            r["method"], r["n_test_cells"],
            r["corpus"]["accuracy"], r["corpus"]["neurite_macro_f1"],
            r["corpus"]["per_class"]["axon"]["f1"],
            r["corpus"]["per_class"]["basal/dendrite"]["f1"],
            r["corpus"]["per_class"]["apical"]["f1"],
            r["per_cell"]["accuracy"]["mean"],
            r["per_cell"]["accuracy"]["p10"],
            r["per_cell"]["neurite_macro_f1"]["mean"],
            r["per_cell"]["neurite_macro_f1"]["p10"],
            r["per_cell"]["neurite_macro_f1"]["p25"],
        ]
        line = "  " + " ".join(c[2].format(v) for c, v in zip(cols, row))
        print(line)


def print_per_class_per_cell(reports: list[dict]) -> None:
    """Per-class per-cell distributions for each method."""
    for r in reports:
        print(f"\n  {r['method']}:")
        print(f"    {'class':<16} {'n_cells':>8} {'mean':>8} {'median':>8} "
              f"{'p10':>8} {'p25':>8} {'p90':>8}")
        for cname in CLASS_NAMES:
            d = r["per_cell"]["per_class_f1"][cname]
            print(f"    {cname:<16} {d['n']:>8} {d['mean']:>8.4f} {d['median']:>8.4f} "
                  f"{d['p10']:>8.4f} {d['p25']:>8.4f} {d['p90']:>8.4f}")


def print_per_cell_type(reports: list[dict]) -> None:
    """Per-cell-type breakdown for each method."""
    for r in reports:
        print(f"\n  ### {r['method']} ###")
        for ct, info in r.get("by_cell_type", {}).items():
            c = info["corpus"]
            pc = info["per_cell"]
            print(f"  {ct}  (n_cells={info['n_cells']}, n_nodes={info['n_nodes']})")
            print(f"    corpus  acc={c['accuracy']:.4f}  neurite_F1={c['neurite_macro_f1']:.4f}  "
                  f"axon={c['per_class']['axon']['f1']:.4f}  "
                  f"basal={c['per_class']['basal/dendrite']['f1']:.4f}  "
                  f"apical={c['per_class']['apical']['f1']:.4f}")
            print(f"    per-cell  acc mean={pc['accuracy']['mean']:.4f} p10={pc['accuracy']['p10']:.4f}  "
                  f"F1 mean={pc['neurite_macro_f1']['mean']:.4f} p10={pc['neurite_macro_f1']['p10']:.4f}")
            cm = c["confusion"]
            print(f"    confusion (GT\\pred):")
            print(f"      {'':<14} {'soma':>10} {'axon':>10} {'basal':>10} {'apical':>10}")
            for g_name in CLASS_NAMES:
                row = cm.get(g_name, {})
                cells = [f"{row.get(p_name, 0):>10d}" for p_name in CLASS_NAMES]
                print(f"      {g_name:<14} {' '.join(cells)}")
