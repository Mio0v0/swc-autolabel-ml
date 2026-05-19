#!/usr/bin/env python3
"""Train Stage 1 + Stage 2 on v12_uncurated using the gpu-xgboost branch.

Pipeline of this script:
  1.  Read paper/results/corpus_qc_v12_uncurated.csv (from _scan_corpus_qc.py).
  2.  Keep only files where qc_pass == True.
  3.  Hash-bucket 80/20 split into train / test (seed=42 by default).
  4.  Train Stage 1 + Stage 2 on the train split using the GPU XGBoost
      wrappers in hybrid/_xgb_classifiers.py (already wired into
      hybrid/train_stage1.py and hybrid/train_stage2.py).
  5.  Save models to paper/models/v12/{cell_type_classifier.pkl,
      branch_classifier.pkl}.
  6.  Fit an OOD detector on the Stage 1 feature distribution of training
      cells. Save as paper/models/v12/qc_gate.pkl.
  7.  Save the train/test split JSON so the eval + calibration scripts
      can reuse the exact partition.

Wall time on a 4080 GPU: roughly ~30-60 min for ~17k cells (rate-limited
by Stage 2 RF training, which is the bottleneck even on GPU).

Output dir:
    paper/models/v12/
        cell_type_classifier.pkl
        branch_classifier.pkl
        qc_gate.pkl
        train_test_split.json
        training_metrics.json
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.evaluate import _train_stage1, _train_stage2, _hash_bucket   # noqa: E402
from hybrid.features import parse_swc, extract_feature_vector, FEATURE_NAMES  # noqa: E402
from hybrid.qc_input import OODDetector, QCGate                          # noqa: E402

# =============================================================================
# Pinned paths
# =============================================================================
QC_CSV     = ROOT / "paper" / "results" / "corpus_qc_v12_uncurated.csv"
OUT_DIR    = ROOT / "paper" / "models" / "v12"
SPLIT_JSON = OUT_DIR / "train_test_split.json"
METRICS    = OUT_DIR / "training_metrics.json"
S1_OUT     = OUT_DIR / "cell_type_classifier.pkl"
S2_OUT     = OUT_DIR / "branch_classifier.pkl"
QCGATE_OUT = OUT_DIR / "qc_gate.pkl"

SEED       = 42
TEST_SIZE  = 0.20         # fraction held out for test
OOD_QUANTILE = 0.99        # threshold for OOD detector


def _load_qc_passed() -> dict[str, list[Path]]:
    """Read QC CSV and return {cell_type: [Path, ...]} for qc_pass==True files."""
    if not QC_CSV.is_file():
        raise FileNotFoundError(
            f"{QC_CSV} not found — run `python -m paper._scan_corpus_qc` first."
        )
    out: dict[str, list[Path]] = {"pyramidal": [], "interneuron": []}
    n_total = 0
    n_pass = 0
    with QC_CSV.open("r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            n_total += 1
            if r.get("qc_pass", "").strip().lower() not in ("true", "1"):
                continue
            n_pass += 1
            ct = r["cell_type"]
            p = Path(r["path"])
            if not p.is_file():
                continue
            out.setdefault(ct, []).append(p)
    print(f"QC CSV: {n_total} files, {n_pass} passed QC")
    for ct, paths in out.items():
        print(f"  {ct}: {len(paths)}")
    return out


def _hash_split(
    qc_passed: dict[str, list[Path]],
    seed: int,
    test_size: float,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """Deterministic 80/20 hash-bucket split, same logic as hybrid.evaluate."""
    train: dict[str, list[Path]] = {}
    test: dict[str, list[Path]] = {}
    for ct, paths in qc_passed.items():
        train[ct] = []
        test[ct] = []
        for p in sorted(paths):
            bucket = _hash_bucket(seed, p.name)
            if bucket < test_size:
                test[ct].append(p)
            else:
                train[ct].append(p)
    return train, test


def _fit_ood_on_train(train_files: dict[str, list[Path]]) -> OODDetector:
    """Extract Stage 1 feature vectors for every training file and fit OOD."""
    print("\nFitting OOD detector on training Stage-1 features...")
    t0 = time.perf_counter()
    feats: list[np.ndarray] = []
    n_failed = 0
    n_total = sum(len(v) for v in train_files.values())
    done = 0
    for ct, paths in train_files.items():
        for p in paths:
            done += 1
            try:
                nodes = parse_swc(p)
                if not nodes:
                    n_failed += 1
                    continue
                fv = extract_feature_vector(nodes)
                feats.append(np.asarray(fv, dtype=np.float64))
            except Exception:
                n_failed += 1
            if done % 1000 == 0:
                print(f"  ... {done}/{n_total}")
    X = np.vstack(feats)
    ood = OODDetector.fit(X, FEATURE_NAMES, quantile=OOD_QUANTILE)
    print(f"  fit {ood.n_train} training cells in {(time.perf_counter()-t0)/60:.1f} min "
          f"(skipped {n_failed} due to parse errors)")
    print(f"  feature_dim={ood.mean.shape[0]}  threshold={ood.threshold:.3f}  "
          f"quantile={OOD_QUANTILE}")
    return ood


def _save_split(train_files, test_files):
    payload = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "train": {ct: sorted(p.name for p in paths) for ct, paths in train_files.items()},
        "test":  {ct: sorted(p.name for p in paths) for ct, paths in test_files.items()},
        "counts": {
            "train": {ct: len(paths) for ct, paths in train_files.items()},
            "test":  {ct: len(paths) for ct, paths in test_files.items()},
        },
    }
    SPLIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved train/test split -> {SPLIT_JSON}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overall_t0 = time.perf_counter()

    # 1) Load QC-passed files
    qc_passed = _load_qc_passed()

    # 2) Hash-bucket split
    train_files, test_files = _hash_split(qc_passed, SEED, TEST_SIZE)
    print()
    print("Split:")
    for ct in ("pyramidal", "interneuron"):
        n_tr = len(train_files.get(ct, []))
        n_te = len(test_files.get(ct, []))
        print(f"  {ct:12s}: train={n_tr}  test={n_te}")
    _save_split(train_files, test_files)

    # 3) Train Stage 1
    print(f"\nTraining Stage 1 -> {S1_OUT.name}")
    t0 = time.perf_counter()
    _train_stage1(train_files, S1_OUT)
    s1_min = (time.perf_counter() - t0) / 60.0
    print(f"  Stage 1 elapsed: {s1_min:.1f} min")

    # 4) Train Stage 2
    print(f"\nTraining Stage 2 -> {S2_OUT.name}")
    t0 = time.perf_counter()
    _train_stage2(train_files, S2_OUT)
    s2_min = (time.perf_counter() - t0) / 60.0
    print(f"  Stage 2 elapsed: {s2_min:.1f} min")

    # 5) Fit OOD detector on training features
    ood = _fit_ood_on_train(train_files)
    gate = QCGate(ood_detector=ood)
    gate.save(QCGATE_OUT)
    print(f"Saved QC gate (structural + OOD) -> {QCGATE_OUT}")

    # 6) Write training metrics
    total_min = (time.perf_counter() - overall_t0) / 60.0
    metrics = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "n_train": {ct: len(paths) for ct, paths in train_files.items()},
        "n_test":  {ct: len(paths) for ct, paths in test_files.items()},
        "stage1_train_min": s1_min,
        "stage2_train_min": s2_min,
        "total_min": total_min,
        "ood_n_train": ood.n_train,
        "ood_threshold": ood.threshold,
        "ood_quantile": OOD_QUANTILE,
    }
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nTOTAL ELAPSED: {total_min:.1f} min")
    print(f"Metrics -> {METRICS}")
    print()
    print("Next steps:")
    print("  1. Train v12 GNN apical/basal head:")
    print("     python -m paper.gnn_apical_basal --data-dir data/v12_uncurated "
          "--ckpt paper/models/v12/gnn_apical_basal.pt")
    print("  2. Calibrate confidence flag thresholds on test set:")
    print("     python -m paper._calibrate_flags")
    print("  3. Run end-to-end with QC + flags on a single SWC:")
    print("     python -m paper._predict_with_flags --input some.swc")
    return 0


if __name__ == "__main__":
    sys.exit(main())
