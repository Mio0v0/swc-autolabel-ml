#!/usr/bin/env python3
"""§5.8 — does curator feedback improve the FLAG model (not just the labeler)?

Simulates the flag half of the feedback loop: the flag model surfaces cells,
a curator reveals each flagged cell's true bad/good status (per-cell F1 < 0.60,
standing in for a correction), that verdict is added to the flag model's
training set, and the flag model is retrained. We measure flag precision /
recall / AP on a FIXED held-out set of cells whose verdicts were never used.

Three arms:
  factory  — flag model trained on base only, never updated (flat reference)
  feedback — each round, review the top-K FLAGGED pool cells, add verdicts, retrain
  random   — each round, review K RANDOM pool cells, add verdicts, retrain

Leakage control: split is by FILE (all seed-rows of a file share a partition),
and the test set is disjoint from base_train and pool.

Usage:
    python -m paper._flag_feedback_sim --rounds 8 --k 150 --reject 0.10

Outputs:
    paper/results/flag_feedback_curve.json
    paper/results/flag_feedback_curve.csv
    paper/results/flag_feedback_curve.txt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "paper" / "results"
FEATURES = RESULTS / "final_flag_multiseed_features.csv"
LABELS = RESULTS / "final_flag_multiseed_labels.csv"
BAD_F1 = 0.60

ID_COLS = {"row_id", "model_seed", "file"}


def _hash_bucket(name: str, salt: str) -> float:
    h = hashlib.md5(f"{salt}:{name}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _load() -> tuple[pd.DataFrame, list[str]]:
    feats = pd.read_csv(FEATURES)
    labs = pd.read_csv(LABELS)[["row_id", "file", "held_out_F1", "cell_type_gt"]]
    df = feats.merge(labs, on="row_id", how="inner", suffixes=("", "_lab"))
    df["file"] = df["file"].fillna(df.get("file_lab"))
    df = df.dropna(subset=["held_out_F1"])
    df["bad"] = (df["held_out_F1"] < BAD_F1).astype(int)
    feat_cols = [c for c in feats.columns
                 if c not in ID_COLS and df[c].dtype != object]
    # Numeric-only, drop all-NaN columns, fill remaining NaNs with 0.
    feat_cols = [c for c in feat_cols if df[c].notna().any()]
    df[feat_cols] = df[feat_cols].fillna(0.0)
    return df, feat_cols


def _fit_flag(X, y, seed):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=seed,
    )
    clf.fit(X, y)
    return clf


def _score(clf, X):
    # Probability of class "bad" (=1).
    classes = list(clf.classes_)
    j = classes.index(1) if 1 in classes else 1
    return clf.predict_proba(X)[:, j]


def _metrics_at_reject(scores, y_true, reject_rate):
    from sklearn.metrics import average_precision_score
    n = len(scores)
    k = max(1, int(round(n * reject_rate)))
    order = np.argsort(scores)[::-1]
    flagged = np.zeros(n, dtype=bool)
    flagged[order[:k]] = True
    tp = int(((flagged) & (y_true == 1)).sum())
    fp = int(((flagged) & (y_true == 0)).sum())
    n_bad = int((y_true == 1).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / n_bad if n_bad > 0 else 0.0
    ap = float(average_precision_score(y_true, scores)) if n_bad > 0 and n_bad < n else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "ap": round(ap, 4), "n_flagged": k, "n_bad": n_bad}


def _run(df, feat_cols, split_seed, rounds, k, reject):
    # Split by FILE.
    files = df["file"].unique()
    part = {}
    for f in files:
        b = _hash_bucket(str(f), f"flagfb_{split_seed}")
        part[f] = "test" if b < 0.20 else ("base" if b < 0.35 else "pool")
    df = df.assign(_part=df["file"].map(part))
    test = df[df["_part"] == "test"]
    base = df[df["_part"] == "base"]
    pool = df[df["_part"] == "pool"].copy()

    Xte, yte = test[feat_cols].values, test["bad"].values
    curves = {"factory": [], "feedback": [], "random": []}

    # Factory model (base only) — flat reference across rounds.
    clf0 = _fit_flag(base[feat_cols].values, base["bad"].values, split_seed)
    m0 = _metrics_at_reject(_score(clf0, Xte), yte, reject)
    for arm in curves:
        curves[arm].append({"round": 0, "n_verdicts": 0, **m0})

    rng = np.random.RandomState(split_seed)
    for arm in ("feedback", "random"):
        train = base.copy()
        remaining = pool.copy()
        clf = clf0
        for rnd in range(1, rounds + 1):
            if len(remaining) == 0:
                break
            if arm == "feedback":
                s = _score(clf, remaining[feat_cols].values)
                take = np.argsort(s)[::-1][:k]
            else:
                take = rng.permutation(len(remaining))[:k]
            reviewed = remaining.iloc[take]
            remaining = remaining.drop(reviewed.index)
            train = pd.concat([train, reviewed], axis=0)
            clf = _fit_flag(train[feat_cols].values, train["bad"].values, split_seed)
            m = _metrics_at_reject(_score(clf, Xte), yte, reject)
            n_verdicts = len(train) - len(base)
            curves[arm].append({"round": rnd, "n_verdicts": n_verdicts, **m})
    return curves, {"n_test": len(test), "n_base": len(base), "n_pool": len(pool)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--k", type=int, default=150)
    ap.add_argument("--reject", type=float, default=0.10)
    ap.add_argument("--split-seeds", default="42,123,789")
    args = ap.parse_args()
    split_seeds = [int(s) for s in args.split_seeds.split(",")]

    print("  loading flag dataset...")
    df, feat_cols = _load()
    print(f"  rows={len(df)}  unique files={df['file'].nunique()}  "
          f"features={len(feat_cols)}  bad_rate={df['bad'].mean():.3f}")

    # Run over split seeds, then average per-round.
    per_seed = []
    meta = None
    for s in split_seeds:
        curves, meta = _run(df, feat_cols, s, args.rounds, args.k, args.reject)
        per_seed.append(curves)
        f_last = curves["feedback"][-1]; r_last = curves["random"][-1]; fac = curves["factory"][0]
        print(f"  split {s}: factory P={fac['precision']:.3f} -> "
              f"feedback P={f_last['precision']:.3f}  random P={r_last['precision']:.3f}  "
              f"(recall {fac['recall']:.3f}->{f_last['recall']:.3f})")

    def _agg(arm):
        n = min(len(ps[arm]) for ps in per_seed)
        out = []
        for i in range(n):
            row = {"round": per_seed[0][arm][i]["round"],
                   "n_verdicts": per_seed[0][arm][i]["n_verdicts"]}
            for m in ("precision", "recall", "ap"):
                vals = [ps[arm][i][m] for ps in per_seed]
                row[m] = round(float(np.mean(vals)), 4)
                row[m + "_sd"] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)
            out.append(row)
        return out

    agg = {arm: _agg(arm) for arm in ("factory", "feedback", "random")}
    out = {"split_seeds": split_seeds, "rounds": args.rounds, "k": args.k,
           "reject_rate": args.reject, "bad_f1_threshold": BAD_F1,
           "n_features": len(feat_cols), **meta, **agg,
           "note": "Flag model retrained on curator-derived bad/good verdicts of "
                   "REVIEWED cells; precision/recall on disjoint held-out test."}
    (RESULTS / "flag_feedback_curve.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    with (RESULTS / "flag_feedback_curve.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "round", "n_verdicts", "precision", "precision_sd",
                    "recall", "recall_sd", "ap", "ap_sd"])
        for arm in ("factory", "feedback", "random"):
            for r in agg[arm]:
                w.writerow([arm, r["round"], r["n_verdicts"], r["precision"], r["precision_sd"],
                            r["recall"], r["recall_sd"], r["ap"], r["ap_sd"]])

    lines = ["Flag-model feedback: precision/recall vs curator verdicts",
             "=" * 66, "",
             f"Mean +/- SD over split seeds {split_seeds}. Reject rate {args.reject:.0%}.",
             f"test={meta['n_test']} base={meta['n_base']} pool={meta['n_pool']}  bad target: per-cell F1<{BAD_F1}",
             "",
             f"{'verdicts':>9} {'precision feedback':>20} {'precision random':>18} {'precision factory':>19}"]
    fac_p = agg["factory"][0]["precision"]
    for ff, rr in zip(agg["feedback"], agg["random"]):
        lines.append(f"{ff['n_verdicts']:>9} "
                     f"{ff['precision']:>10.4f}+/-{ff['precision_sd']:<6.4f} "
                     f"{rr['precision']:>8.4f}+/-{rr['precision_sd']:<6.4f} "
                     f"{fac_p:>17.4f}")
    lines += ["", "Recall (feedback) and AP:",
              f"{'verdicts':>9} {'recall':>16} {'AP':>16}"]
    for ff in agg["feedback"]:
        lines.append(f"{ff['n_verdicts']:>9} {ff['recall']:>8.4f}+/-{ff['recall_sd']:<6.4f} "
                     f"{ff['ap']:>8.4f}+/-{ff['ap_sd']:<6.4f}")
    (RESULTS / "flag_feedback_curve.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print()
    print("\n".join(lines))
    print("\n  wrote flag_feedback_curve.{json,csv,txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
