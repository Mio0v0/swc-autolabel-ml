#!/usr/bin/env python3
"""T2: cross-seed Stage 1 cell-type disagreement scan.

For every QC-passed cell in `paper/results/corpus_qc_v12_uncurated.csv`,
runs both Stage 1 classifiers (v12_gentle_seed42, v12_gentle_seed789)
and records:
    - Each seed's predicted cell type + confidence
    - Whether each seed disagrees with GT
    - For each seed: was the cell in its train or test split (honesty flag)

A cell is flagged as `STRONG_CANDIDATE` if BOTH seeds high-confidence
disagree with GT. Such cells are very likely whole-cell GT errors:
both Stage 1 models, trained on different splits using only
morphology summary features (not node labels), independently say
the cell is a different type from its directory label.

Usage:
    python -m paper._scan_stage1_disagreement
        [--max-cells N]   # debug: limit to first N cells

Output:
    paper/results/stage1_disagreement.csv   ranked by suspicion
    paper/results/stage1_disagreement_summary.json

Wall time: ~15 min for 12,484 cells on CPU.
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

from hybrid.cell_type_detector import CellTypeClassifier               # noqa: E402
from hybrid.features import (                                          # noqa: E402
    parse_swc, FEATURE_NAMES, extract_global_features,
)

QC_CSV  = ROOT / "paper" / "results" / "corpus_qc_v12_uncurated.csv"
OUT_CSV = ROOT / "paper" / "results" / "stage1_disagreement.csv"
OUT_JSON = ROOT / "paper" / "results" / "stage1_disagreement_summary.json"

SEEDS = [42, 789]


def _load_split(seed: int) -> tuple[set[str], set[str]]:
    p = ROOT / "paper" / "models" / f"v12_gentle_seed{seed}" / "train_test_split.json"
    sp = json.loads(p.read_text(encoding="utf-8"))
    tr: set[str] = set()
    te: set[str] = set()
    for ct in ("pyramidal", "interneuron"):
        tr.update(sp["train"].get(ct, []))
        te.update(sp["test"].get(ct, []))
    return tr, te


def _classify(model: CellTypeClassifier, feature_vec: np.ndarray) -> tuple[str, float]:
    pred, probs = model.predict(feature_vec)
    return pred, float(probs.get(pred, 0.0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-cells", type=int, default=None,
                    help="Process only the first N cells (debug).")
    args = ap.parse_args()

    if not QC_CSV.is_file():
        raise SystemExit(f"MISSING: {QC_CSV}")

    # Load both seed Stage 1 classifiers
    models = {}
    splits = {}
    for s in SEEDS:
        d = ROOT / "paper" / "models" / f"v12_gentle_seed{s}"
        mp = d / "cell_type_classifier.pkl"
        if not mp.is_file():
            raise SystemExit(f"MISSING: {mp}")
        models[s] = CellTypeClassifier.load(mp)
        train, test = _load_split(s)
        splits[s] = {"train": train, "test": test}
        print(f"  seed={s}: model loaded; train={len(train)} test={len(test)}")

    # Load QC-passed cells
    cells: list[tuple[str, Path]] = []
    with QC_CSV.open("r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("qc_pass", "").strip().lower() not in ("true", "1"):
                continue
            p = Path(r["path"])
            if p.is_file():
                cells.append((r["cell_type"], p))
    if args.max_cells:
        cells = cells[: args.max_cells]
    print(f"\nScanning {len(cells)} QC-passed cells with Stage 1 from both seeds...")

    rows: list[dict] = []
    t0 = time.perf_counter()
    n_failed = 0
    for i, (ct_gt, p) in enumerate(cells):
        try:
            nodes = parse_swc(p)
            if not nodes:
                n_failed += 1
                continue
            features = extract_global_features(nodes)
            fv = np.array([features[n] for n in FEATURE_NAMES], dtype=np.float64)
        except Exception:
            n_failed += 1
            continue

        # Predict with each seed's Stage 1
        per_seed: dict[int, dict] = {}
        for s in SEEDS:
            pred, conf = _classify(models[s], fv)
            in_test  = p.name in splits[s]["test"]
            in_train = p.name in splits[s]["train"]
            per_seed[s] = {
                "pred":    pred,
                "conf":    conf,
                "in_test": in_test,
                "in_train": in_train,
                "disagrees": (pred != ct_gt),
            }

        # Composite suspicion score
        # +5 per seed that disagrees high-confidence (conf>0.85)
        # +3 per seed that disagrees mid-confidence (0.65-0.85)
        # +1 per seed that disagrees low-confidence (<0.65)
        # +1 if the disagreeing seed is HONEST (cell was in its test split, not train)
        score = 0
        n_seeds_disagree = 0
        n_disagree_honest = 0   # disagreed AND that seed didn't train on this cell
        for s in SEEDS:
            d = per_seed[s]
            if d["disagrees"]:
                n_seeds_disagree += 1
                if d["conf"] > 0.85: score += 5
                elif d["conf"] > 0.65: score += 3
                else: score += 1
                if d["in_test"]:
                    score += 1
                    n_disagree_honest += 1

        # Classification:
        if n_seeds_disagree == 2 and all(per_seed[s]["conf"] > 0.85 for s in SEEDS):
            cls = "STRONG_CANDIDATE"
        elif n_seeds_disagree == 2:
            cls = "LIKELY_GT_ERROR"
        elif n_seeds_disagree == 1 and any(per_seed[s]["conf"] > 0.85 and per_seed[s]["disagrees"] for s in SEEDS):
            cls = "MAYBE_GT_ERROR"
        else:
            cls = "OK"

        rows.append({
            "file":              p.name,
            "cell_type_gt":      ct_gt,
            "seed42_pred":       per_seed[42]["pred"],
            "seed42_conf":       f"{per_seed[42]['conf']:.4f}",
            "seed42_held_out":   str(per_seed[42]["in_test"]),
            "seed789_pred":      per_seed[789]["pred"],
            "seed789_conf":      f"{per_seed[789]['conf']:.4f}",
            "seed789_held_out":  str(per_seed[789]["in_test"]),
            "n_seeds_disagree":  n_seeds_disagree,
            "n_honest_disagree": n_disagree_honest,
            "suspect_score":     score,
            "classification":    cls,
            "source":            p.name.split("__", 1)[0] if "__" in p.name else "",
        })

        if (i + 1) % 1000 == 0:
            print(f"  ... {i+1}/{len(cells)}  ({(time.perf_counter()-t0)/60:.1f} min)", flush=True)

    elapsed = (time.perf_counter() - t0) / 60.0
    print(f"\nDone scanning: {len(rows)} cells scored, {n_failed} skipped ({elapsed:.1f} min)")

    # Sort by suspect_score desc
    rows.sort(key=lambda r: (-r["suspect_score"], r["file"]))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"Wrote {OUT_CSV}")

    # Summary
    cls_counter = Counter(r["classification"] for r in rows)
    by_celltype: dict[str, Counter] = {"pyramidal": Counter(), "interneuron": Counter()}
    for r in rows:
        by_celltype[r["cell_type_gt"]][r["classification"]] += 1

    summary = {
        "total_scanned":          len(rows),
        "n_failed":               n_failed,
        "classification_counts":  dict(cls_counter),
        "by_celltype":            {ct: dict(c) for ct, c in by_celltype.items()},
        "elapsed_min":            elapsed,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=== Classification breakdown ===")
    for cls in ("STRONG_CANDIDATE", "LIKELY_GT_ERROR", "MAYBE_GT_ERROR", "OK"):
        c = cls_counter.get(cls, 0)
        print(f"  {cls:<20}  {c:>5}  ({100*c/len(rows):.1f}%)")
    print()
    print("=== By cell type ===")
    for ct, c in by_celltype.items():
        total = sum(c.values())
        bad = c.get("STRONG_CANDIDATE", 0) + c.get("LIKELY_GT_ERROR", 0)
        pct = (100 * bad / total) if total else 0.0
        print(f"  {ct:<12}  total={total}  STRONG+LIKELY={bad}  ({pct:.1f}%)")
    print()
    print("=== Top 25 most suspect (STRONG_CANDIDATE preview) ===")
    print(f"  {'file':<60} {'gt':<12} {'s42 pred/conf':<22} {'s789 pred/conf':<22} {'score':>5}")
    for r in rows[:25]:
        print(f"  {r['file'][:60]:<60} {r['cell_type_gt']:<12} "
              f"{r['seed42_pred']+' '+r['seed42_conf']:<22} "
              f"{r['seed789_pred']+' '+r['seed789_conf']:<22} "
              f"{r['suspect_score']:>5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
