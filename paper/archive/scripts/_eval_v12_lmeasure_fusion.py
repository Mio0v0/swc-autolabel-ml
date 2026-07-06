#!/usr/bin/env python3
"""Evaluate conservative v12 + baseline fusion post-processors.

Goal: recover the tiny corpus-level gap to L-Measure without giving up v12's
much better per-cell tail. This does not train; it runs v12 and cached external
baselines on the seed split, then sweeps low-confidence override rules.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.evaluate import per_cell_neurite_f1, per_cell_per_class_f1  # noqa: E402
from hybrid.features import parse_swc  # noqa: E402
from hybrid.pipeline import run_pipeline_on_nodes  # noqa: E402
from paper._eval_baselines_on_v12 import CACHE_DIR, _load_split_from_qc  # noqa: E402
from paper.external_baselines import predict_with_cache  # noqa: E402
from paper.gnn_branch3_inference import load_branch3  # noqa: E402
from paper.gnn_inference import load_gnn  # noqa: E402

CLASS_IDS = (1, 2, 3, 4)
CLASS_NAMES = {1: "soma", 2: "axon", 3: "basal/dendrite", 4: "apical"}


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _dist(values: list[float | None]) -> dict:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "p10": 0.0, "p25": 0.0, "median": 0.0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
    }


class Accumulator:
    def __init__(self, name: str):
        self.name = name
        self.conf = np.zeros((5, 5), dtype=np.int64)
        self.by_ct = {
            "pyramidal": np.zeros((5, 5), dtype=np.int64),
            "interneuron": np.zeros((5, 5), dtype=np.int64),
        }
        self.n_cells = 0
        self.n_nodes = 0
        self.n_cells_by_ct = {"pyramidal": 0, "interneuron": 0}
        self.n_nodes_by_ct = {"pyramidal": 0, "interneuron": 0}
        self.pc_acc: list[float] = []
        self.pc_f1: list[float] = []
        self.pc_acc_by_ct = {"pyramidal": [], "interneuron": []}
        self.pc_f1_by_ct = {"pyramidal": [], "interneuron": []}
        self.pc_class = {name: [] for name in CLASS_NAMES.values()}

    def add(self, ct: str, gt: list[int], pred: list[int]) -> None:
        self.n_cells += 1
        self.n_nodes += len(gt)
        self.n_cells_by_ct[ct] += 1
        self.n_nodes_by_ct[ct] += len(gt)
        for g, p in zip(gt, pred):
            if g in CLASS_IDS and p in CLASS_IDS:
                self.conf[g, p] += 1
                self.by_ct[ct][g, p] += 1
        acc = sum(1 for g, p in zip(gt, pred) if g == p) / max(1, len(gt))
        f1 = per_cell_neurite_f1(gt, pred, ct)
        self.pc_acc.append(acc)
        self.pc_f1.append(f1)
        self.pc_acc_by_ct[ct].append(acc)
        self.pc_f1_by_ct[ct].append(f1)
        pc = per_cell_per_class_f1(gt, pred, ct)
        for name in CLASS_NAMES.values():
            self.pc_class[name].append(pc.get(name))

    @staticmethod
    def _metrics_from_conf(conf: np.ndarray, valid_labels: tuple[int, ...] = CLASS_IDS) -> dict:
        total = int(conf.sum())
        acc = float(sum(int(conf[i, i]) for i in valid_labels) / max(1, total))
        per_class = {}
        f1s = []
        neurite_f1s = []
        for cls in valid_labels:
            tp = int(conf[cls, cls])
            fp = int(conf[:, cls].sum() - tp)
            fn = int(conf[cls, :].sum() - tp)
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-12, precision + recall)
            name = CLASS_NAMES[cls]
            per_class[name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(conf[cls, :].sum()),
            }
            f1s.append(f1)
            if cls in (2, 3, 4):
                neurite_f1s.append(f1)
        return {
            "accuracy": acc,
            "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
            "neurite_macro_f1": float(np.mean(neurite_f1s)) if neurite_f1s else 0.0,
            "per_class": per_class,
        }

    def report(self) -> dict:
        corpus = self._metrics_from_conf(self.conf)
        by_ct = {}
        for ct in ("pyramidal", "interneuron"):
            labels = (1, 2, 3, 4) if ct == "pyramidal" else (1, 2, 3)
            by_ct[ct] = {
                "n_cells": self.n_cells_by_ct[ct],
                "n_nodes": self.n_nodes_by_ct[ct],
                "corpus": self._metrics_from_conf(self.by_ct[ct], labels),
                "per_cell": {
                    "accuracy": _dist(self.pc_acc_by_ct[ct]),
                    "neurite_macro_f1": _dist(self.pc_f1_by_ct[ct]),
                },
            }
        return {
            "method": self.name,
            "n_test_cells": self.n_cells,
            "n_test_nodes": self.n_nodes,
            "corpus": corpus,
            "per_cell": {
                "accuracy": _dist(self.pc_acc),
                "neurite_macro_f1": _dist(self.pc_f1),
                "per_class_f1": {k: _dist(v) for k, v in self.pc_class.items()},
            },
            "by_cell_type": by_ct,
        }


def _baseline_consensus(labels_by_method: dict[str, list[int]], i: int) -> tuple[int | None, int]:
    votes = [pred[i] for pred in labels_by_method.values()]
    label, n = Counter(votes).most_common(1)[0]
    if n >= 2:
        return int(label), int(n)
    return None, int(n)


def _apply_variant(
    name: str,
    v12: list[int],
    conf: list[float],
    lm: list[int],
    labels_by_method: dict[str, list[int]],
    ct: str,
    threshold: float,
) -> list[int]:
    out = list(v12)
    for i, label in enumerate(v12):
        c = float(conf[i])
        if c > threshold:
            continue
        consensus, n_cons = _baseline_consensus(labels_by_method, i)
        if name == "lmeasure_lowconf":
            if lm[i] != label:
                out[i] = int(lm[i])
        elif name == "lmeasure_axon_cleanup":
            if label == 2 and lm[i] != 2:
                out[i] = int(lm[i])
        elif name == "consensus_lowconf":
            if consensus is not None and consensus != label:
                out[i] = consensus
        elif name == "consensus_axon_cleanup":
            if label == 2 and consensus is not None and consensus != 2:
                out[i] = consensus
        elif name == "apical_consensus":
            if ct == "pyramidal" and label != 4 and sum(1 for p in labels_by_method.values() if p[i] == 4) >= 2:
                out[i] = 4
        elif name == "combo_axon_apical":
            if label == 2 and consensus is not None and consensus != 2:
                out[i] = consensus
            elif ct == "pyramidal" and label != 4 and sum(1 for p in labels_by_method.values() if p[i] == 4) >= 2:
                out[i] = 4
    return out


def _row_from_report(r: dict) -> dict:
    pc = r["per_cell"]["neurite_macro_f1"]
    corpus = r["corpus"]
    return {
        "method": r["method"],
        "n_test_cells": r["n_test_cells"],
        "accuracy": corpus["accuracy"],
        "neurite_macro_f1": corpus["neurite_macro_f1"],
        "axon_f1": corpus["per_class"]["axon"]["f1"],
        "basal_f1": corpus["per_class"]["basal/dendrite"]["f1"],
        "apical_f1": corpus["per_class"].get("apical", {}).get("f1", 0.0),
        "per_cell_f1_mean": pc["mean"],
        "per_cell_f1_p10": pc["p10"],
        "per_cell_f1_p25": pc["p25"],
        "pyramidal_acc": r["by_cell_type"]["pyramidal"]["corpus"]["accuracy"],
        "pyramidal_neurite_f1": r["by_cell_type"]["pyramidal"]["corpus"]["neurite_macro_f1"],
        "pyramidal_apical_f1": r["by_cell_type"]["pyramidal"]["corpus"]["per_class"]["apical"]["f1"],
        "interneuron_neurite_f1": r["by_cell_type"]["interneuron"]["corpus"]["neurite_macro_f1"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument(
        "--methods",
        default="lmeasure_rf,sholl_rf,neurom_rf",
        help="Comma-separated cached baselines to run. Use lmeasure_rf for the fast focused sweep.",
    )
    ap.add_argument("--out-json", type=Path, default=ROOT / "paper" / "results" / "v12_lmeasure_fusion_sweep_seed123.json")
    ap.add_argument("--out-csv", type=Path, default=ROOT / "paper" / "results" / "v12_lmeasure_fusion_sweep_seed123.csv")
    ap.add_argument("--out-txt", type=Path, default=ROOT / "paper" / "results" / "v12_lmeasure_fusion_sweep_seed123.txt")
    args = ap.parse_args()

    model_dir = ROOT / "paper" / "models" / f"v12_gentle_seed{args.seed}"
    s1 = model_dir / "cell_type_classifier.pkl"
    s2 = model_dir / "branch_classifier.pkl"
    gnn_path = model_dir / "gnn_apical_basal.pt"
    branch3_path = model_dir / "gnn_branch3_rescue.pt"
    for p in (s1, s2, gnn_path, branch3_path):
        if not p.is_file():
            raise SystemExit(f"MISSING: {p}")
    gnn_state = load_gnn(gnn_path)
    branch3_state = load_branch3(branch3_path)

    train, test = _load_split_from_qc(args.seed)
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    if "lmeasure_rf" not in methods:
        raise SystemExit("--methods must include lmeasure_rf because L-Measure rules need it")
    predictors = {
        method: predict_with_cache(
            method,
            train,
            seed=args.seed,
            cache_path=CACHE_DIR / f"{method}.pkl",
            force_retrain=False,
        )
        for method in methods
    }

    files: list[tuple[str, Path]] = []
    for ct in ("pyramidal", "interneuron"):
        files.extend((ct, p) for p in test[ct])
    files = sorted(files, key=lambda x: x[1].name)
    if args.max_cells is not None:
        files = files[: args.max_cells]

    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 1.01]
    rule_names = [
        "lmeasure_lowconf",
        "lmeasure_axon_cleanup",
        "consensus_lowconf",
        "consensus_axon_cleanup",
        "apical_consensus",
        "combo_axon_apical",
    ]
    variant_names = ["v12", "lmeasure_rf", "sholl_rf", "neurom_rf"]
    variant_names.extend(f"{rule}@{thr:g}" for rule in rule_names for thr in thresholds)
    accs = {name: Accumulator(name) for name in variant_names}

    t0 = time.perf_counter()
    print(f"Evaluating {len(files)} cells with v12 + baseline fusion rules")
    for idx, (ct, path) in enumerate(files, 1):
        nodes = parse_swc(path)
        gt = [n.type for n in nodes]
        pr = run_pipeline_on_nodes(
            nodes,
            file_path="",
            stage1_model=s1,
            stage2_model=s2,
            gnn_state=gnn_state,
            branch3_state=branch3_state,
            use_subtree_stage2=True,
            override_cell_type=ct,
        )
        v12 = list(pr.node_labels)
        conf = [float(c) for c in pr.node_confidences]
        labels_by_method = {
            method: list(predict_fn(nodes, ct))
            for method, predict_fn in predictors.items()
        }
        lm = labels_by_method["lmeasure_rf"]

        accs["v12"].add(ct, gt, v12)
        for method in methods:
            accs[method].add(ct, gt, labels_by_method[method])
        for rule in rule_names:
            for thr in thresholds:
                pred = _apply_variant(rule, v12, conf, lm, labels_by_method, ct, thr)
                accs[f"{rule}@{thr:g}"].add(ct, gt, pred)
        if idx % args.progress_every == 0:
            elapsed = (time.perf_counter() - t0) / 60.0
            eta = elapsed * (len(files) - idx) / max(1, idx)
            print(f"  ... {idx}/{len(files)} ({elapsed:.1f} min, ETA {eta:.1f} min)", flush=True)

    reports = [acc.report() for acc in accs.values()]
    rows = [_row_from_report(r) for r in reports]
    rows = sorted(
        rows,
        key=lambda r: (
            r["neurite_macro_f1"],
            r["accuracy"],
            r["apical_f1"],
            r["per_cell_f1_p10"],
        ),
        reverse=True,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "n_cells": len(files),
                "elapsed_min": (time.perf_counter() - t0) / 60.0,
                "reports": reports,
                "summary_rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with args.out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "v12 + L-Measure/baseline fusion sweep",
        "=" * 120,
        "Sorted by corpus neurite macro F1. Higher is better; pc_F1_p10 guards tail behavior.",
        "",
        f"{'method':<32} {'acc':>7} {'neurF1':>7} {'axon':>7} {'basal':>7} {'apical':>7} {'pcMean':>7} {'pcP10':>7} {'pyrF1':>7} {'intF1':>7}",
    ]
    for r in rows[:25]:
        lines.append(
            f"{r['method']:<32} {r['accuracy']:7.4f} {r['neurite_macro_f1']:7.4f} "
            f"{r['axon_f1']:7.4f} {r['basal_f1']:7.4f} {r['apical_f1']:7.4f} "
            f"{r['per_cell_f1_mean']:7.4f} {r['per_cell_f1_p10']:7.4f} "
            f"{r['pyramidal_neurite_f1']:7.4f} {r['interneuron_neurite_f1']:7.4f}"
        )
    args.out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {_display(args.out_json)}")
    print(f"Wrote {_display(args.out_csv)}")
    print(f"Wrote {_display(args.out_txt)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
