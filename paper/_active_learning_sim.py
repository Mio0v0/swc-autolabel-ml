#!/usr/bin/env python3
"""Simulated-curator active-learning experiment for the flag-guided loop.

Demonstrates the value of the human-in-the-loop feedback structure
(§4.7 / §5.7): when a curator corrects the cells the flag step surfaces,
the labeler improves *faster* than when the same number of random cells
are corrected. We simulate the curator by revealing ground-truth labels
for queried cells.

Design
------
- Working set: QC-pass cells, deterministically subsampled to N_MAX and
  split by hash into SEED (initial train) / POOL (curator-queryable, labels
  hidden until queried) / TEST (fixed, never queried).
- Acquisition = the flag/uncertainty signal: per-cell node-weighted mean
  (1 - max posterior) of the current Stage-2 branch classifier. High =
  the model is unsure = the flag step surfaces it.
- Two arms share the round-0 model then diverge:
    flag   — query the top-K most-uncertain unqueried POOL cells
    random — query K random unqueried POOL cells
  Each round: reveal queried cells' labels, add to train, refit, eval TEST.
- Metric = per-cell neurite macro-F1 (mean + P10) and node-pooled per-class
  F1 on the fixed TEST set.

To keep multi-round retraining tractable we cache per-cell branch features
once and retrain only the Stage-2 branch classifier each round (the core
learnable per-branch component). The acquisition comparison is unaffected —
both arms use the identical training procedure, isolating the value of
flag-guided selection.

Usage
-----
    python -m paper._active_learning_sim --n-max 5000 --rounds 8 --k 200 --seed 123

Outputs
-------
    paper/results/active_learning_curve.json
    paper/results/active_learning_curve.csv
    paper/results/active_learning_curve.txt
    paper/models/al_feature_cache_<n>.pkl   (feature cache, reused across runs)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SWCAL_CLASS_BALANCE_POWER", "1.25")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.features import parse_swc                               # noqa: E402
from hybrid.branch_features import extract_branches                 # noqa: E402
from hybrid.evaluate import per_cell_neurite_f1, _compute_metrics   # noqa: E402
from hybrid.train_stage2 import _node_balanced_weights             # noqa: E402
from hybrid.cell_type_detector import CELL_TYPE_LABEL_SETS          # noqa: E402

QC_CSV = ROOT / "paper" / "results" / "corpus_qc_v12_uncurated.csv"
CACHE_DIR = ROOT / "paper" / "models"
RESULTS = ROOT / "paper" / "results"

LABEL_NAMES = {1: "soma", 2: "axon", 3: "basal/dendrite", 4: "apical"}


# ---------------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------------

def _hash_bucket(name: str, salt: str) -> float:
    import hashlib
    h = hashlib.md5(f"{salt}:{name}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _load_qc_cells(n_max: int) -> list[tuple[str, str, Path]]:
    """Return [(source, cell_type, path)] stratified-subsampled to n_max."""
    rows: list[tuple[str, str, Path]] = []
    with QC_CSV.open("r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("qc_pass", "").strip().lower() not in ("true", "1"):
                continue
            p = Path(r["path"])
            if not p.is_file():
                continue
            rows.append((r.get("source", "unknown"), r["cell_type"], p))
    # Deterministic stratified subsample: sort by hash within each stratum,
    # take a proportional slice.
    if len(rows) <= n_max:
        return rows
    by_stratum: dict[tuple[str, str], list] = {}
    for src, ct, p in rows:
        by_stratum.setdefault((src, ct), []).append((src, ct, p))
    keep: list = []
    total = len(rows)
    for stratum, items in by_stratum.items():
        items.sort(key=lambda t: _hash_bucket(t[2].name, "subsample"))
        quota = max(1, round(n_max * len(items) / total))
        keep.extend(items[:quota])
    return keep


def _build_cache(n_max: int, cache_path: Path) -> list[dict]:
    cells = _load_qc_cells(n_max)
    print(f"  building feature cache for {len(cells)} cells...")
    cache: list[dict] = []
    t0 = time.perf_counter()
    for i, (src, ct, p) in enumerate(cells):
        try:
            nodes = parse_swc(p)
            if not nodes:
                continue
            morph = extract_branches(nodes, ct, str(p))
        except Exception:
            continue
        gt_nodes = [nd.type for nd in nodes]
        branches = []
        for br in morph.branches:
            branches.append({
                "feat": np.asarray(br.features, dtype=np.float32),
                "node_indices": list(br.node_indices),
                "gt": int(br.gt_label),
                "n_nodes": int(br.n_nodes),
            })
        if not branches:
            continue
        cache.append({
            "file": p.name,
            "source": src,
            "cell_type": ct,
            "n_nodes": len(nodes),
            "gt_nodes": np.asarray(gt_nodes, dtype=np.int8),
            "soma_indices": list(morph.soma_indices),
            "branches": branches,
        })
        if (i + 1) % 500 == 0:
            print(f"    ... {i+1}/{len(cells)} ({(time.perf_counter()-t0)/60:.1f} min)")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as fh:
        pickle.dump(cache, fh)
    print(f"  cached {len(cache)} cells -> {cache_path.name} "
          f"({(time.perf_counter()-t0)/60:.1f} min)")
    return cache


# ---------------------------------------------------------------------------
# Stage-2 classifier (core per-branch model; mirrors the real Stage 2 fit)
# ---------------------------------------------------------------------------

def _fit_stage2(train_cells: list[dict], seed: int):
    from hybrid._xgb_classifiers import XGBRandomForestClassifier
    X, y, w = [], [], []
    for c in train_cells:
        valid = set(CELL_TYPE_LABEL_SETS.get(c["cell_type"], {1, 2, 3})) - {1}
        for br in c["branches"]:
            if br["gt"] in valid:
                X.append(br["feat"])
                y.append(br["gt"])
                w.append(float(br["n_nodes"]))
    X = np.stack(X).astype(np.float32)
    y = np.asarray(y)
    w = np.asarray(w, dtype=np.float64)
    sw = _node_balanced_weights(y, w, power=1.25)
    clf = XGBRandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    clf.fit(X, y, sample_weight=sw)
    return clf


def _predict_cell(clf, cell: dict):
    """Return (pred_nodes, per_cell_uncertainty)."""
    feats = np.stack([br["feat"] for br in cell["branches"]]).astype(np.float32)
    proba = clf.predict_proba(feats)
    classes = clf.classes_
    valid = set(CELL_TYPE_LABEL_SETS.get(cell["cell_type"], {1, 2, 3})) - {1}
    # Restrict to valid neurite classes for this cell type.
    col_mask = np.array([int(c) in valid for c in classes])
    pred_nodes = np.zeros(cell["n_nodes"], dtype=np.int64)
    for idx in cell["soma_indices"]:
        if 0 <= idx < len(pred_nodes):
            pred_nodes[idx] = 1
    unc_num = 0.0
    unc_den = 0.0
    for br, row in zip(cell["branches"], proba):
        if col_mask.any():
            vp = row.copy()
            vp[~col_mask] = -1.0
            best_j = int(np.argmax(vp))
            pmax = float(row[col_mask].max())
        else:
            best_j = int(np.argmax(row))
            pmax = float(row.max())
        label = int(classes[best_j])
        for idx in br["node_indices"]:
            if 0 <= idx < len(pred_nodes):
                pred_nodes[idx] = label
        unc_num += (1.0 - pmax) * br["n_nodes"]
        unc_den += br["n_nodes"]
    uncertainty = unc_num / unc_den if unc_den > 0 else 0.0
    return pred_nodes, uncertainty


def _eval_test(clf, test_cells: list[dict]) -> dict:
    per_cell_f1 = []
    gt_pool, pred_pool, ct_pool = [], [], []
    for c in test_cells:
        pred_nodes, _ = _predict_cell(clf, c)
        gt = c["gt_nodes"].tolist()
        pred = pred_nodes.tolist()
        per_cell_f1.append(per_cell_neurite_f1(gt, pred, c["cell_type"]))
        gt_pool.extend(gt); pred_pool.extend(pred)
    arr = np.array(per_cell_f1)
    # Node-pooled per-class F1 over the union label set.
    m = _compute_metrics(gt_pool, pred_pool, {1, 2, 3, 4})
    per_label = m.get("per_label", {})
    return {
        "per_cell_f1_mean": round(float(arr.mean()), 4),
        "per_cell_f1_p10":  round(float(np.quantile(arr, 0.10)), 4),
        "per_cell_f1_median": round(float(np.median(arr)), 4),
        "node_accuracy":    round(float(m.get("accuracy", 0.0)), 4),
        "neurite_macro_f1": round(float(m.get("neurite_macro_f1", 0.0)), 4),
        "axon_f1":   round(float(per_label.get("axon", {}).get("f1", 0.0)), 4),
        "basal_f1":  round(float(per_label.get("basal/dendrite", {}).get("f1", 0.0)), 4),
        "apical_f1": round(float(per_label.get("apical", {}).get("f1", 0.0)), 4),
    }


# ---------------------------------------------------------------------------
# Active-learning loop
# ---------------------------------------------------------------------------

def _run_arm(arm: str, seed_cells, pool_cells, test_cells, rounds, k, seed):
    rng = np.random.RandomState(seed)
    train = list(seed_cells)
    remaining = list(pool_cells)
    curve = []

    clf = _fit_stage2(train, seed)
    m0 = _eval_test(clf, test_cells)
    m0.update({"round": 0, "arm": arm, "n_labeled": len(train),
               "n_curator_corrections": 0})
    curve.append(m0)
    print(f"  [{arm}] round 0  n_train={len(train)}  "
          f"macroF1={m0['neurite_macro_f1']}  apical={m0['apical_f1']}  "
          f"pcF1={m0['per_cell_f1_mean']}  P10={m0['per_cell_f1_p10']}")

    for rnd in range(1, rounds + 1):
        if not remaining:
            break
        if arm == "flag":
            # Score remaining pool by current-model uncertainty; pick top-K.
            scored = []
            for c in remaining:
                _, unc = _predict_cell(clf, c)
                scored.append((unc, c))
            scored.sort(key=lambda t: t[0], reverse=True)
            queried = [c for _, c in scored[:k]]
        else:  # random
            idx = rng.permutation(len(remaining))[:k]
            queried = [remaining[i] for i in idx]
        queried_names = {c["file"] for c in queried}
        remaining = [c for c in remaining if c["file"] not in queried_names]
        train.extend(queried)  # reveal GT labels (curator correction)

        clf = _fit_stage2(train, seed)
        m = _eval_test(clf, test_cells)
        m.update({"round": rnd, "arm": arm, "n_labeled": len(train),
                  "n_curator_corrections": len(train) - len(seed_cells)})
        curve.append(m)
        print(f"  [{arm}] round {rnd}  n_train={len(train)}  "
              f"macroF1={m['neurite_macro_f1']}  apical={m['apical_f1']}  "
              f"pcF1={m['per_cell_f1_mean']}  P10={m['per_cell_f1_p10']}")
    return curve


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-max", type=int, default=5000)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--seed-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()

    cache_path = CACHE_DIR / f"al_feature_cache_{args.n_max}.pkl"
    if cache_path.is_file() and not args.rebuild_cache:
        print(f"  loading feature cache {cache_path.name}...")
        with cache_path.open("rb") as fh:
            cache = pickle.load(fh)
        print(f"  loaded {len(cache)} cells")
    else:
        cache = _build_cache(args.n_max, cache_path)

    # Deterministic seed / pool / test split by hash bucket.
    seed_cells, pool_cells, test_cells = [], [], []
    for c in cache:
        b = _hash_bucket(c["file"], f"al_split_{args.seed}")
        if b < args.test_frac:
            test_cells.append(c)
        elif b < args.test_frac + args.seed_frac:
            seed_cells.append(c)
        else:
            pool_cells.append(c)
    print(f"\n  split: seed={len(seed_cells)}  pool={len(pool_cells)}  test={len(test_cells)}")
    print(f"  rounds={args.rounds}  k={args.k}  "
          f"(max queried={args.rounds*args.k} of {len(pool_cells)} pool)\n")

    t0 = time.perf_counter()
    print("=== ARM: flag-guided (uncertainty acquisition) ===")
    curve_flag = _run_arm("flag", seed_cells, pool_cells, test_cells,
                          args.rounds, args.k, args.seed)
    print("\n=== ARM: random (control) ===")
    curve_rand = _run_arm("random", seed_cells, pool_cells, test_cells,
                          args.rounds, args.k, args.seed)
    elapsed = (time.perf_counter() - t0) / 60.0
    print(f"\n  both arms done in {elapsed:.1f} min")

    all_rows = curve_flag + curve_rand
    tag = f"_seed{args.seed}"
    out_json = RESULTS / f"active_learning_curve{tag}.json"
    out_json.write_text(json.dumps({
        "n_max": args.n_max, "seed": args.seed,
        "rounds": args.rounds, "k": args.k,
        "n_seed": len(seed_cells), "n_pool": len(pool_cells), "n_test": len(test_cells),
        "acquisition": "per-cell node-weighted mean (1 - max posterior) of Stage-2 classifier",
        "note": "simulated curator reveals GT labels for queried cells; Stage-2 branch "
                "classifier retrained each round on seed + revealed cells.",
        "flag_arm": curve_flag,
        "random_arm": curve_rand,
    }, indent=2), encoding="utf-8")

    out_csv = RESULTS / f"active_learning_curve{tag}.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        cols = ["arm", "round", "n_labeled", "n_curator_corrections",
                "neurite_macro_f1", "apical_f1", "basal_f1", "axon_f1",
                "per_cell_f1_mean", "per_cell_f1_p10", "node_accuracy"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in cols})

    # Pretty summary: paired flag vs random per round.
    lines = ["Active-learning curve: flag-guided vs random curator selection",
             "=" * 68, "",
             f"n_max={args.n_max} seed={args.seed}  "
             f"seed_train={len(seed_cells)} pool={len(pool_cells)} test={len(test_cells)}",
             f"rounds={args.rounds} k={args.k}  acquisition=uncertainty (flag proxy)",
             "",
             f"{'round':>5} {'corrections':>12} "
             f"{'macroF1 flag':>13} {'macroF1 rand':>13} {'d':>7}  "
             f"{'apical flag':>12} {'apical rand':>12} {'d':>7}"]
    fmap = {r["round"]: r for r in curve_flag}
    rmap = {r["round"]: r for r in curve_rand}
    for rnd in sorted(fmap):
        f = fmap[rnd]; r = rmap.get(rnd, {})
        dmac = f["neurite_macro_f1"] - r.get("neurite_macro_f1", f["neurite_macro_f1"])
        dap = f["apical_f1"] - r.get("apical_f1", f["apical_f1"])
        lines.append(f"{rnd:>5} {f['n_curator_corrections']:>12} "
                     f"{f['neurite_macro_f1']:>13.4f} {r.get('neurite_macro_f1',0):>13.4f} {dmac:>+7.4f}  "
                     f"{f['apical_f1']:>12.4f} {r.get('apical_f1',0):>12.4f} {dap:>+7.4f}")
    out_txt = RESULTS / f"active_learning_curve{tag}.txt"
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("\n".join(lines))
    print(f"\n  wrote {out_json.name}, {out_csv.name}, {out_txt.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
