# Flag/Rejection Model — Handoff Document

> 2026-06-08 cleanup note: use `paper/CANONICAL_ARTIFACTS.md` and
> `paper/CANONICAL_PAPER_CHECKLIST.md` as the current source of truth.
> Some older files named in this handoff have been moved to `paper/archive/`
> because they are historical experiments rather than active pipeline files.

**Audience**: someone joining this project to build a per-cell flag/rejection model.
**Assumption**: you have not seen this codebase before. This doc gives you everything you need to start.

## Current Canonical Update (2026-05-28)

- Accepted labeler/model directory: `paper/models/v12_gentle_seed123/`
- Model recipe: full cleaned 11,862-cell QC corpus, seed=123, 80/20 hash split, conservative Branch3 enabled.
- Clean no-leak test split: 2,402 cells (1,745 pyramidal, 657 interneuron).
- GT-celltype evaluation, no flag: accuracy 0.9791, neurite macro F1 0.9443, axon/basal/apical F1 0.9912/0.9440/0.8979, per-cell F1 mean 0.9296, P10 0.6480.
- Current flag training data: `paper/results/heldout_per_cell_f1_seed123_branch3.csv` plus `paper/results/flag_confidence_features_seed123_branch3.csv`.
- Accepted flag model: `paper/models/flag_model_seed123_branch3_pyramidal/flag_model_f060.joblib`; use Stage1-pyramidal guarded flags.

Older references to `v12_pyramidal_only_seed2024_clean/` and `heldout_per_cell_f1_branch3.csv`
are historical unless a section explicitly says it is describing the old pyramidal-only experiment.

---

## 1. What this project is

This repo is an automatic neuron-morphology labeler. Input is an `.swc` file
(a neural reconstruction — points in 3D space with parent links). Output is a
per-node class label:

| Class ID | Name | Description |
|---|---|---|
| 1 | soma | cell body |
| 2 | axon | output process |
| 3 | basal | basal dendrite (input process, near soma) |
| 4 | apical | apical dendrite (input process, characteristic of pyramidal cells; long trunk reaching up) |

The pipeline (`hybrid/pipeline.py:run_pipeline_on_nodes`) has 5 stages:

1. **QC gate** — rejects malformed or disconnected reconstructions (bad rows, duplicate IDs, multiple roots, orphan parents, invalid radii/coordinates, OOD morphology, etc.); runtime input QC does not require pre-existing soma or neurite labels.
2. **Stage 1** — predicts cell type ("pyramidal" or "interneuron") from cell-level features (RF / XGBoost classifier)
3. **Stage 2** — predicts per-branch labels using a per-cell-type RandomForest on engineered features (one model for pyramidal, one for interneuron)
4. **Stage 3** — GraphSAGE GNN refines apical-vs-basal predictions (pyramidal cells only) + topology refinement rules
5. **Branch3 rescue** — a 3-class GraphSAGE branch head for pyramidals that can correct axon/basal/apical branches, including apical branches initially mislabeled as axon

**Current performance (clean pyramidal model + conservative Branch3 rescue, GT cell-type override, 4263 test cells):**
- Per-node accuracy: **0.9823**
- Apical F1 (corpus-pooled): 0.9081
- Basal F1 (corpus-pooled): 0.9437
- Per-cell F1 mean: 0.9209
- **Per-cell F1 P10: 0.6214** ← improved from 0.5908, but the bottom 10% is still the weak spot

An optional learned Branch3 gate exists (`branch3_gate.joblib`) and scores higher in aggregate
(P10 0.6463, apical F1 0.9117), but it is not the default because it causes too many
large regressions on cells that were already good.

## 2. What the flag model should do

Build a model that, given **ANY input SWC cell** (pyramidal OR interneuron) and the **full v12 pipeline's predictions**, outputs `P(bad)` — the probability that this cell's per-cell F1 is below a threshold (e.g. F1 < 0.5).

At inference time, you don't know the true F1 — that's what we want to predict.

**Why this matters**: the full v12 pipeline performs unevenly across cell types:
- **Pyramidals**: per-cell F1 P10 ≈ 0.50 (bottom 10% are catastrophic)
- **Interneurons**: per-cell F1 P10 ≈ 0.97 (almost all fine; tiny tail)

So in practice most "bad cells" the flag will catch are pyramidals — but the model should be **trained on the union of both cell types** so it generalizes and learns the failure signatures across cell types (e.g. Stage 1 mis-routes an interneuron to the pyramidal pipeline and produces apical predictions on a cell that physically can't have apical → these ARE catchable bad-interneuron cases).

**Use the full pipeline (Stage 1 active) as your prediction source.** Stage 1 sometimes mis-classifies cells, and those Stage 1 errors propagate downstream — those failures are part of what the flag model needs to catch. **Do NOT use GT cell-type override** at inference time (we don't have GT cell-type at deployment).

**Why this matters at deployment**:
- Bad cells get sent to human review (don't auto-publish wrong labels)
- The "auto-labeled" output excludes them, lifting headline metrics dramatically
- Users get a per-cell quality score alongside predictions

**Empirical ceilings** (per-cell F1 P10 lift from perfect "oracle" 10% rejection):
- Pyramidals: P10 0.50 → ~0.93 (+43 pp ceiling)
- Interneurons: P10 0.97 → ~1.00 (+3 pp ceiling — almost nothing to gain)
- **All cells combined**: P10 ~0.66 → ~0.99 (most of the lift comes from pyramidals)

---

## 3. Repo layout (what's on disk, post-cleanup as of 2026-05-27)

```
swc-autolabel-ml/
├── data/v12_uncurated/                      # raw SWC files (DO NOT MODIFY)
│   ├── pyramidal/swc/*.swc                  # ~11,000 pyramidal cells on disk
│   └── interneuron/swc/*.swc                # ~3,000 interneurons on disk
│
├── hybrid/                                  # core pipeline library
│   ├── pipeline.py                          # run_pipeline_on_nodes (main entry point)
│   ├── features.py                          # parse_swc, SWCNode, FEATURE_NAMES
│   ├── branch_features.py                   # BRANCH_FEATURE_NAMES, extract_branches
│   ├── evaluate.py                          # per_cell_neurite_f1 helper
│   ├── cell_type_detector.py                # Stage 1
│   ├── train_stage2.py                      # Stage 2
│   ├── stage3_refine.py                     # topology refinement after Stage 2/GNN
│   └── qc_input.py                          # OOD detector, QC gate (Mahalanobis)
│
├── paper/
│   ├── models/                              # 4 trained model directories
│   │   ├── baselines/                       # 4 external baselines on uncleaned seed=42 split
│   │   │   ├── neurom_rf.pkl
│   │   │   ├── lmeasure_rf.pkl              # ← USE for cross-model features
│   │   │   ├── sholl_rf.pkl
│   │   │   └── sholl_mlp.pkl
│   │   ├── v12_gentle_seed42/               # Full v12 pipeline, uncleaned seed=42 (80/20 split)
│   │   ├── v12_gentle_seed789/              # Full v12 pipeline, uncleaned seed=789 (80/20 split)
│   │   └── v12_pyramidal_only_seed2024_clean/   # ← CURRENT BEST for pyramidal work
│   │       ├── cell_type_classifier.pkl     # Stage 1 (frozen from seed=42)
│   │       ├── branch_classifier.pkl        # Stage 2 (pyramidal half re-trained on cleaned corpus; interneuron frozen)
│   │       ├── gnn_apical_basal.pt          # Stage 3 GNN (re-trained on cleaned corpus, focal γ=2)
│   │       ├── gnn_branch3_rescue.pt        # Branch3 rescue head (3-class pyramidal branch correction)
│   │       ├── branch3_gate.joblib           # Optional learned Branch3 accept/abstain gate (not default)
│   │       ├── branch3_cache/               # Cached Branch3 train/test graphs for threshold sweeps/restarts
│   │       ├── qc_gate.pkl                  # OOD detector (frozen)
│   │       ├── train_test_split.json        # 50/50 hash-bucket split on pyramidals, seed=2024
│   │       ├── eval_split_for_gnn.json      # GNN-format split
│   │       └── training_metrics.json
│   │
│   ├── results/                             # ALL outputs from training/eval/analysis
│   │   ├── corpus_qc_v12_uncurated.csv      # The CURRENT corpus catalog (11,862 QC-pass)
│   │   ├── *.csv / *.json / *.txt           # see Section 11.10 for the full canonical list
│   │   └── *.md                             # comparison tables, failure-mode taxonomy
│   │
│   ├── logs/                                # training/eval verbose logs (re-runnable, can be cleaned)
│   │
│   ├── _retrain_v12_gentle_seed.py          # main full-v12 retrain script
│   ├── _train_pyramidal_only.py             # pyramidal-only retrain (Stage 2 + GNN only)
│   ├── _eval_pyramidal_gt_celltype.py       # ← BEST starting reference for inference
│   ├── _eval_lmeasure_pyramidal_only.py     # L-Measure-RF eval (cross-model features)
│   ├── _eval_per_seed_own_test.py           # eval v12 on its seed's held-out test
│   ├── _eval_baselines_on_v12.py            # eval all 4 baselines together
│   ├── _eval_v12_gt_celltype.py             # GT cell-type override variant
│   ├── _eval_rejection_sweep.py             # rejection sweep (oracle + hand-rules)
│   ├── _compare_baselines.py                # rebuild the comparison report
│   ├── _analyze_failure_modes.py            # rebuild the failure-mode taxonomy
│   ├── _score_heldout_f1.py                 # the script that produced heldout_per_cell_f1.csv
│   ├── _scan_stage1_disagreement.py         # the T2 cross-seed scan
│   ├── _identify_hard_training_cells.py     # for hard-cell upweighting experiments
│   ├── _drop_bad_lab_cells.py               # bad-lab cleanup utility
│   ├── _scan_corpus_qc.py                   # rebuilds corpus_qc CSV from raw .swc dir
│   ├── _backup_gnns_before_retrain.py       # safety utility
│   ├── _restore_best_gnn.py                 # safety utility
│   ├── _remake_v11_figures.py               # paper figure generator
│   ├── _make_architecture_slides.py         # slide generator
│   ├── baselines.py                         # baseline framework
│   ├── external_baselines.py                # the 4 external baselines (NeuroM/L-Measure/Sholl)
│   ├── gnn_apical_basal.py                  # GNN model + training
│   ├── gnn_branch3_rescue.py                # 3-class pyramidal rescue GNN training
│   ├── gnn_branch3_inference.py             # Branch3 load/score helper
│   ├── gnn_dataset.py                       # GNN data loader
│   ├── gnn_inference.py                     # load_gnn(path) helper
│   ├── PIPELINE_EXPERIMENTS.md              # ← log of everything tried, results
│   ├── STAGE1_EXPERIMENTS.md                # Stage 1 ceiling analysis
│   ├── PHASE1_RESULTS.md / PHASE2_RESULTS.md / PHASE3A_RESULTS.md
│   ├── CACHING_POLICY.md
│   └── FLAG_MODEL_HANDOFF.md                # ← THIS DOCUMENT
│
└── README.md
```

**Quick orientation**:
- For `inference`, look at `_eval_pyramidal_gt_celltype.py` first
- For `current best model`, use `paper/models/v12_gentle_seed123/`
- For `current corpus`, use `paper/results/corpus_qc_v12_uncurated.csv` (filter on `qc_pass == True`)
- For `experiment history`, read `paper/PIPELINE_EXPERIMENTS.md`

---

## 4. The data you need (already on disk)

### A. Per-cell labeled data (READY TO USE)

These CSVs have one row per cell with the per-cell F1 score — these are your **labels** for training the flag model.

**Use the full-pipeline both-cell-type files as primary** (the flag model needs to work on the full pipeline output for any cell type):

| File | Cells | Cell types | Pipeline | Use |
|---|---|---|---|---|
| **`paper/results/heldout_per_cell_f1_seed123_branch3.csv`** | **2,402** | both (pyr + int) | seed=123 accepted conservative-Branch3 labeler, Stage 1 active, no-leak test split | **CURRENT PRIMARY flag-model labels.** |
| **`paper/results/flag_confidence_features_seed123_branch3.csv`** | **2,402** | both (pyr + int) | seed=123 accepted conservative-Branch3 labeler | **CURRENT PRIMARY feature companion.** Adds confidence, geometry, and Branch3-vs-no-Branch3 disagreement features. |
| `paper/results/heldout_per_cell_f1_branch3.csv` | 4,176 | both (pyr + int) | older conservative-Branch3 labeler, Stage 1 active | Historical labels after stale-row purge. |
| `paper/results/flag_confidence_features_branch3.csv` | 4,176 | both (pyr + int) | older conservative-Branch3 labeler | Historical feature companion. |
| `paper/results/heldout_per_cell_f1.csv` | 4,176 | both (pyr + int) | older seed42/seed789 full v12, Stage 1 active | Historical/reference labels after stale-row purge. |
| `paper/results/per_seed_own_test.csv` | 4,621 rows after purge | both (pyr + int) | full v12 (Stage 1 active) | Additional historical labels — has per-cell accuracy. |

**Optional supplementary data** (pyramidal-only or with-GT-override — useful for cross-model features, NOT for primary labels):

| File | Cells | Cell types | Pipeline | Use |
|---|---|---|---|---|
| `paper/results/v12_pyramidal_only_seed2024_clean_eval.csv` | 4,263 | pyramidal only | Stage 1 BYPASSED | Use if you want to also model "Stage 2+3 quality independent of Stage 1." Optional. |
| `paper/results/lmeasure_pyramidal_only_eval.csv` | 4,263 | pyramidal only | L-Measure-RF (no Stage 1) | Cross-model features for pyramidals (v12 vs L-Measure agreement) |
| `paper/results/baselines_on_v12.csv` | varies | both | 4 baselines (no Stage 1) | Cross-model features (NeuroM, L-Measure, Sholl-RF, Sholl-MLP predictions) on seed=42 split |

**Why the Branch3 files are primary now**: they match the accepted labeler that will be deployed, include BOTH cell types with Stage 1 active, and exclude the stale bad-GT rows that were removed from the active corpus.

**Schema of `v12_pyramidal_only_seed2024_clean_eval.csv`** (columns):
- `file` — SWC filename (e.g. `neuromorpho__abc.swc`)
- `n_nodes` — total nodes
- `n_apical_gt`, `n_basal_gt`, `n_axon_gt` — GT class counts (ONLY use for label, not as feature)
- `n_apical_pred`, `n_basal_pred`, `n_axon_pred` — predicted class counts (USE as feature)
- `accuracy` — per-cell node accuracy (label)
- `F1_neurite` — per-cell macro F1 over neurite classes (label — define "bad" from this)
- `axon_F1`, `basal_F1`, `apical_F1`, `soma_F1` — per-class F1 (label)

**Schema of `heldout_per_cell_f1_branch3.csv`** (columns):
- `file`, `cell_type_gt`, `n_nodes`, `seed_used`
- `held_out_F1` — same idea as above (label)
- `axon_F1`, `basal_F1`, `apical_F1`
- `stage1_pred`, `stage1_conf`, `stage1_correct` — Stage 1 outputs
- `n_components` — disconnected components in reconstruction
- `soma_z`, `apical_mean_z`, `apical_above_soma`, `apical_z_extent`
- `axon_frac`, `apical_frac`, `basal_frac` — GT class fractions (DON'T use as feature; use predicted instead)
- `pred_axon`, `pred_basal`, `pred_apical` — predicted class counts
- `source` — neuromorpho / allen / hpf_ca1

### B. Supplementary signals (helpful features)

| File | What it has | Use |
|---|---|---|
| `paper/results/gt_failure_modes.csv` | Per-cell categorical labels (which failure mode the cell falls in: APICAL_MISSED, BASAL_LOST, etc.) | Optional — can be used as categorical input feature OR to validate flag model on specific failure types |
| `paper/results/stage1_disagreement.csv` | Per-cell Stage 1 cross-seed agreement | Useful feature — cells where Stage 1 disagreed across seeds are likely problematic |
| `paper/results/purged_non_qc_rows_summary.json` | Audit summary for stale rows removed from result CSVs | Confirms no rows outside current `qc_pass=True` corpus remain in file-column CSVs |
| `paper/results/dropped_labs_summary.json` | List of 21 "bad labs" (already filtered out of corpus, but lab prefix from filename is still a useful feature) | Use lab prefix as a feature |
| `paper/results/corpus_qc_v12_uncurated.csv` | The full corpus catalog (12,484 cells originally → 11,862 after cleanup). One row per file with structural QC stats | Can join on filename to get pre-pipeline structural stats |

### C. Model + raw data (if you need to re-run inference)

| Path | What it is |
|---|---|
| `paper/models/v12_gentle_seed123/` | The production-equivalent model. Use it to compute features that require running inference (e.g. per-node confidences) |
| `data/v12_uncurated/{pyramidal,interneuron}/swc/*.swc` | Raw SWC files. Parse with `from hybrid.features import parse_swc; nodes = parse_swc(path)` |

---

## 5. Scripts to learn from (don't reinvent)

### Show you how to load a model and run inference

| Script | What it demonstrates |
|---|---|
| `paper/_eval_pyramidal_gt_celltype.py` | Complete example: loads a model dir, runs the full pipeline on test cells, computes per-cell metrics, writes JSON+CSV+pretty TXT. **Best starting reference** for inference-time code. |
| `paper/_eval_lmeasure_pyramidal_only.py` | Same idea but for the L-Measure baseline (different model). Useful if you want to compute cross-model features. |
| `paper/_score_heldout_f1.py` | Shows how to score per-cell F1 + structural fingerprints across many cells in batch. Has a watchdog timer for catching hangs. |

### Show you the pipeline internals

| Script / module | What it exposes |
|---|---|
| `hybrid/pipeline.py:run_pipeline_on_nodes()` | The main inference function. Takes parsed SWC nodes, returns a `PipelineResult` with `node_labels` (list of int) and `node_confidences` (list of float). Use `override_cell_type="pyramidal"` to bypass Stage 1. |
| `hybrid/features.py:parse_swc(path)` | Parse an SWC file into a list of `SWCNode` dataclasses. |
| `hybrid/evaluate.py:per_cell_neurite_f1(gt, pred, cell_type)` | The canonical per-cell F1 metric used in this project. Use it for labels. |

### Show you the corpus state

| Doc | Content |
|---|---|
| `paper/PIPELINE_EXPERIMENTS.md` | History of attempted improvements with results. Useful context for what's been tried. |
| `paper/results/gt_failure_modes.md` | Taxonomy of failure modes in the bottom 10% (APICAL_MISSED, BASAL_LOST, etc.) with rule definitions. **Useful for understanding what kinds of "bad" exist.** |
| `paper/results/comparison_v12_vs_lmeasure.txt` | v12 vs L-Measure side-by-side. The ensemble ceiling is +9.8 pp on P10, motivating cross-model features. |
| `paper/results/rejection_sweep.txt` | Earlier attempt at rejection with hand-crafted rules. **Failed badly (6% precision at 10% rejection)** — illustrates that learned model is needed. |

---

## 6. Feature design (recommended)

Per cell, compute these features. Group by data source needed.

### Free features (use the existing eval CSV columns directly)

- `n_nodes` — total reconstruction size
- `n_apical_pred / n_nodes` — predicted apical fraction
- `n_basal_pred / n_nodes` — predicted basal fraction
- `n_axon_pred / n_nodes` — predicted axon fraction
- Number of distinct predicted classes (1, 2, or 3 of {axon, basal, apical})
- Source (one-hot: allen / hpf_ca1 / neuromorpho)
- Lab prefix (one-hot or hashed embedding)

### Structural features (join with `heldout_per_cell_f1.csv` for these)

- `n_components` — disconnected components (broken reconstruction)
- `apical_above_soma` — direction of apical relative to soma (raw and signed)
- `apical_z_extent` — height of apical
- `soma_z` — soma position (for coordinate-convention normalization)

### Cross-model features (compute by re-running L-Measure prediction)

- For each cell, how many predicted nodes does v12 agree with L-Measure on? (% agreement)
- Does v12 dominant class match L-Measure dominant class?
- Predicted apical fraction delta between v12 and L-Measure

### Confidence features (REQUIRES re-running v12 inference and saving per-node confidence)

These are **the strongest predictors** — but they're not in the current CSVs. To get them:
1. Use `paper/_eval_pyramidal_gt_celltype.py` as a template
2. Modify it to save `pr.node_confidences` per cell (currently only `node_labels` is captured)
3. Re-run inference on the 4,263 test cells (~30 min)
4. Compute per-cell:
   - mean / std / min / 10th percentile of per-node confidence
   - entropy of per-node class probability distribution
   - fraction of nodes with confidence < 0.6, < 0.8
   - mean confidence within each predicted class (apical confidence, basal confidence, axon confidence)

### Pipeline-internal features (advanced)

- Stage 2 vs Stage 3 (GNN) disagreement rate per cell
- Per-cell Mahalanobis distance to training distribution (the existing QC gate already computes this)
- Stage 1 confidence margin (max prob − second max prob)

---

## 7. Training procedure (recommended)

```python
# 1. Build feature matrix X (cells × features) + label vector y
# 2. Define "bad" = (F1_neurite < 0.5)  → y is binary
# 3. Stratified split: 70% train, 15% val, 15% test
# 4. Train XGBoost:
import xgboost as xgb
clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=(n_neg / n_pos),  # handles class imbalance (~10% positive)
    objective='binary:logistic',
    eval_metric='aucpr',
    early_stopping_rounds=30,
)
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=20)
# 5. Calibrate probabilities (Platt or isotonic on val):
from sklearn.calibration import CalibratedClassifierCV
clf_cal = CalibratedClassifierCV(clf, cv='prefit', method='isotonic').fit(X_val, y_val)
# 6. Evaluate on test:
y_prob = clf_cal.predict_proba(X_test)[:, 1]
# Report: ROC AUC, PR AUC, precision @ recall=0.7, precision @ recall=0.8
```

### Success criteria

| Metric | Minimum | Good | Excellent |
|---|---|---|---|
| ROC AUC | > 0.75 | > 0.85 | > 0.90 |
| Precision @ recall=0.8 | > 0.40 | > 0.60 | > 0.75 |
| At 10% rejection rate, F1 P10 on kept cells | > 0.65 | > 0.75 | > 0.85 |

For reference, the **oracle ceiling at 10% rejection is F1 P10 = 0.93**. The hand-crafted rules previously tested produced 0.58 (worse than baseline 0.59). Anywhere between 0.65 and 0.93 means real value.

---

## 8. Getting started (concrete first steps)

### Day 1: MVP with existing features (no inference needed)

1. Load `v12_pyramidal_only_seed2024_clean_eval.csv` (the primary label source)
2. Optionally join with `heldout_per_cell_f1.csv` on `file` to add structural features (note: heldout has a slightly different cell set, so use as additional training data)
3. Define `y = (F1_neurite < 0.5).astype(int)`
4. Build feature matrix from already-present columns + lab-prefix encoding
5. Train XGBoost with 5-fold CV
6. Report ROC AUC + precision-recall curve
7. **Sanity check**: should achieve AUC > 0.70 just from free features. If not, something is broken in feature extraction.

### Day 2-3: Add inference-time features

1. Re-run inference saving per-node confidences (modify `_eval_pyramidal_gt_celltype.py` to dump confidences to a parquet/CSV)
2. Compute confidence statistics per cell
3. Retrain model with confidence features
4. Expected lift: ROC AUC from ~0.75 to ~0.85

### Day 4: Cross-model + OOD features

1. Re-run L-Measure prediction, save per-node predictions per cell
2. Compute v12 vs L-Measure agreement features
3. Compute Mahalanobis distance per cell using `hybrid/qc_input.py:OODDetector` (load from `qc_gate.pkl`)
4. Retrain. Expected lift: 0.85 → 0.88+

### Day 5: Operating point selection + final eval

1. PR curve analysis on test set
2. Pick threshold for the use case (high precision vs high recall)
3. Report performance at picked threshold
4. Build a small wrapper script `flag_cell.py` that takes an SWC path + model and outputs `P(bad)`

---

## 9. Gotchas / things to be careful about

1. **Don't use GT-derived features at inference**. The CSV has `n_apical_gt` which is the GT class count — this can't be known at inference. Only use `n_apical_pred`. (For training labels, GT is fine; just don't put it in X.)

2. **The corpus has heterogeneous coordinate conventions**. Some labs use Z-down, others Z-up. `apical_above_soma < 0` does NOT mean "broken cell" — for ~7% of cells it just means inverted Y axis. Use absolute values or robust scaling.

3. **Class imbalance is real but moderate** (~10-15% positive). Set `scale_pos_weight` in XGBoost or use class_weight in sklearn.

4. **Some cells will fail to parse / predict**. Handle exceptions; the eval CSVs already filter to successfully-predicted cells.

5. **Stage 1 cell-type was OVERRIDDEN in the baseline eval** (`override_cell_type="pyramidal"`). This isolates Stage 2+3 performance. If you want to model the FULL pipeline (Stage 1 included), use `heldout_per_cell_f1.csv` instead — those F1s are with Stage 1's prediction (which is sometimes wrong on pyramidals).

6. **Confidences are calibrated within each Stage 2 sub-model** (isotonic) but the GNN's output isn't calibrated. If you use raw GNN confidence, apply a separate calibration.

7. **The model dir `v12_gentle_seed123/` is the canonical reference**. It was trained on the cleaned 11,862-cell corpus with an 80/20 full-corpus split, seed=123. Use its `train_test_split.json` and `paper/results/leakage_check_seed123.txt` to make sure your train/test splits don't leak.

---

## 10. Quick links

- **Environment**: Python 3.13, PyTorch 2.5 with CUDA, XGBoost, sklearn. Virtualenv at `D:/Desktop/SWC-Studio/.venv/`.
- **Run any script**: `cd swc-autolabel-ml; python -m paper.<script_name>`
- **Inspect a model**: `pickle.load(open('paper/models/v12_gentle_seed123/branch_classifier.pkl', 'rb'))` returns a dict — see `hybrid/pipeline.py:_select_stage2_model` for keys.

For pipeline questions, the entry point is `hybrid/pipeline.py:run_pipeline_on_nodes`. Read it top-to-bottom (~250 lines) — it explains the full inference flow.

For label-generation questions, see `hybrid/evaluate.py:per_cell_neurite_f1`.

For corpus / split questions, every model dir contains a `train_test_split.json` that lists train and test files.

---

---

## 11. Current state of the project (as of 2026-05-28)

This section gives you the lay of the land — what's been done, what works, what didn't, and what numbers represent the current best.

### 11.1 Corpus state

- **Started**: 12,484 QC-passed cells (87% neuromorpho, 11% hpf_ca1, 2% allen)
- **Cleaning pass 1 (T2)**: dropped 89 cells where cross-seed Stage 1 strongly disagreed with GT cell-type → very likely whole-cell mislabels
- **Cleaning pass 2 (bad-lab)**: dropped 533 cells from 21 specific labs/studies within neuromorpho that had >40% per-cell failure rate
- **Current corpus**: **11,862 QC-pass cells (8,488 pyramidal + 3,374 interneuron)**
- The 622 dropped cells are gone from `paper/results/corpus_qc_v12_uncurated.csv` (rows removed entirely); the .swc files themselves are still on disk but unreferenced
- ⚠️ Note: the seed=42 and seed=789 models on disk were trained BEFORE this cleanup — they saw those 622 cells during training. The newer `v12_pyramidal_only_seed2024_clean` was trained on the cleaned corpus.

### 11.2 Current best models on disk

| Model dir | Use it for | Trained on |
|---|---|---|
| **`v12_gentle_seed123/`** | **PRIMARY full-pipeline model.** Stage 1 + Stage 2 + Stage 3 GNN + conservative Branch3 + QC. Use this at inference time. | Cleaned 11,862-cell corpus, 80/20 split, seed=123 |
| `v12_gentle_seed42/` | Historical full-pipeline model | Uncleaned 12,484-cell corpus, 80/20 split, seed=42 |
| `v12_gentle_seed789/` | Second seed, useful for cross-seed agreement features | Same corpus, seed=789 |
| `v12_pyramidal_only_seed2024_clean/` | Pyramidal-only, Stage 1 BYPASSED. Use only if you want to model "what would Stage 2+3 do if Stage 1 were perfect" as a feature. | Cleaned 11,862-corpus, 4,225 train pyramidals, seed=2024 |
| `baselines/` | 4 external baselines (NeuroM-RF, L-Measure-RF, Sholl-RF, Sholl-MLP) for cross-model agreement features | seed=123 80/20 split, both cell types |

**Why `v12_gentle_seed123` is primary**: at deployment, you'll run the full pipeline (Stage 1 active, both cell types supported) on incoming SWCs. The flag model needs to predict P(bad) on that pipeline's output.

### 11.3 Headline numbers (use these as benchmarks)

**Current production model quality: v12_gentle_seed123 + Branch3, GT cell-type override** on the clean held-out test (2,402 cells: 1,745 pyr + 657 int):

| Cell type | n cells | per-node acc | apical F1 corpus | basal F1 corpus | axon F1 corpus | **per-cell F1 mean** | **per-cell F1 P10** |
|---|---|---|---|---|---|---|---|
| **pyramidal** | 1,745 | 0.9809 | **0.9052** | 0.9363 | 0.9935 | **0.9175** | **0.6114** |
| **interneuron** | 657 | 0.9775 | n/a | 0.9574 | 0.9847 | **0.9617** | **0.9871** |
| **both pooled** | 2,402 | 0.9791 | 0.8979 | 0.9440 | 0.9912 | **0.9296** | **0.6480** |

Historical seed=42 full-pipeline numbers:

| Cell type | n cells | per-node acc | apical F1 corpus | basal F1 corpus | axon F1 corpus | **per-cell F1 mean** | **per-cell F1 P10** |
|---|---|---|---|---|---|---|---|
| **pyramidal** | 1,776 | 0.9804 | **0.8984** | 0.9273 | 0.9945 | **0.8947** | **0.5000** |
| **interneuron** | 692 | 0.9655 | n/a | 0.9379 | 0.9770 | **0.9592** | **0.9726** |
| **both pooled** | 2,468 | ≈0.9765 | — | — | — | ≈0.911 | ≈0.65 |

(Interneurons have no apical class — they're a 3-class problem.)

**Headline observation for the flag model**: pyramidals account for nearly all bad cells (P10=0.50). Interneurons rarely fail (P10=0.97). The flag model should mostly distinguish "is this a bad pyramidal?" but also occasionally catch "Stage 1 misrouted an interneuron through the pyramidal pipeline and produced apical predictions."

**Second seed (cross-seed reference)**: v12_gentle_seed789 → pyramidal P10 0.500, interneuron P10 0.912 (consistent with seed=42).

**Baselines on the same seed=42 held-out** (for cross-model features — these always use GT cell-type, no Stage 1):

| Baseline | pyr P10 | int P10 | pyr apical F1 corpus |
|---|---|---|---|
| L-Measure-RF | 0.473 | 0.831 | 0.916 |
| Sholl-RF | 0.401 | 0.697 | 0.895 |
| Sholl-MLP | 0.378 | 0.487 | 0.862 |
| NeuroM-RF | 0.498 | 0.649 | 0.693 |

**v12 vs L-Measure best-of-two oracle ceiling (pyramidals only)**:
- F1 P10 ceiling: **0.6887** (vs v12 alone 0.5908, +9.8 pp lift)
- Cross-model agreement (v12 ↔ L-Measure) is a strong feature for the flag model

Full comparison table: `paper/results/v12_vs_baselines_comparison.md`

### 11.4 What's known about the failure modes

From the T4 failure-mode analysis on bottom 10% of pyramidals (~435 cells):

| Category | Cells in bottom 10% | Enrichment vs corpus | Interpretation |
|---|---|---|---|
| APICAL_MISSED | 264 (62%) | 8.8x | GT has apical, model predicts 0 apical. Mix of GT errors and hard cells. |
| APICAL_UNDERPREDICTED | 276 (63%) | 8.9x | Same direction as MISSED, milder. Overlaps with MISSED. |
| STAGE1_PROPAGATION | 77 (18%) | 9.3x | Stage 1 wrong + F1 collapsed. Targets ~77 cells. |
| BASAL_LOST | 90 (21%) | 8.6x | Basal labeled as apical or axon. Often paired with APICAL_OVERPREDICTED. |
| AXON_DENDRITE_CONFUSION | 142 (33%) | 6.0x | Axon F1 < 0.95 (rare; usually small cells). |
| TRUNCATED_APICAL | 125 (29%) | 4.0x | Apical extent < 50µm. Legitimate partial reconstructions, not GT errors. |
| APICAL_INVERTED | 149 (34%) | **1.7x** ← weak signal | Apical mean Z below soma. Mostly just coordinate-convention difference, NOT GT errors. |
| APICAL_OVERPREDICTED | 9 | 10.0x | Strong signal but only 9 cells. Model calls basal "apical". |
| WHOLE_CELL_MISLABEL | 1 | 1.3x | T2 already cleaned most of these. |

**Practical implications for the flag model:**
- The biggest failure category is "apical recognition wrong" (MISSED + UNDERPREDICTED + OVERPREDICTED + BASAL_LOST) — same underlying axis. ~500+ cells affected.
- Stage 1 propagation is a clean 77-cell sub-target.
- `APICAL_INVERTED` looks like a great feature but actually isn't — too many false positives.
- Full details: `paper/results/gt_failure_modes.md`

### 11.5 What's been tried to improve apical/basal F1 (DON'T re-try these)

| Experiment | Result | Why it failed (or worked) |
|---|---|---|
| GNN focal loss γ=2 | ✅ **KEPT** (+1.26 pp P10) | Down-weighting easy examples helps tail; current default |
| GNN focal loss γ=3 | ✗ −1.15 pp | Over-focuses; loses on easy cells |
| Aggressive soft handoff (thr=0.99) | ✗ **−22 pp catastrophe** | "More confident pipeline" picked wrong-cell-type pipeline |
| Branch neighbor smoothing | ✗ 0 effect | Existing Stage 3 refinement already catches these |
| Aggressive island flipping (size 30) | ✗ 0 effect | Same — already caught |
| Stage 2 class balance power 2.0 | ✗ −0.21 pp | Existing 1.25 was well-tuned |
| Stage 2 class balance power 1.0 (no balance) | ✗ −1.38 pp on P10 | Even worse |
| **L-Measure features (all 4) added to v12** | ✗ **−3.6 pp P10 on apples-to-apples** | Added redundancy + noise; net negative |
| T2 cell-type mislabel cleanup (89 cells) | ✓ small positive | Real cleanup |
| Bad-lab cleanup (533 cells) | ✓ small positive | Real cleanup |
| Stage 1 iteration (6 attempts) | Plateau at 95.46% accuracy | Feature-limited ceiling, not data-limited |

Full log: `paper/PIPELINE_EXPERIMENTS.md`

### 11.6 Why the flag model is the high-leverage next step

The empirical rejection-sweep result (A1) shows:
- **Oracle rejection of bottom 10% by F1** lifts P10 from 0.59 → **0.93** (+34 pp)
- **Hand-crafted rejection rules** achieve **0.58 P10** (slightly worse than baseline — rules can't capture the right cells)
- **Conclusion: a learned flag model is the right tool.** The signal is there (oracle ceiling proves it); rules can't find it; ML can.

If your flag model achieves AUC > 0.85, you'd capture maybe half of the 34 pp oracle ceiling = **+15 pp on F1 P10** — bigger than any single architectural or data-cleanup change tried.

### 11.7 Why NOT to keep trying architectural changes for now

The flag model is more leverage than architecture changes because:
1. **All single hyperparameter / architecture tweaks have produced ≤+1.3 pp** on P10 (the focal γ=2 is the only winner across 18 attempts)
2. **The corpus has hit a noise floor** — adding more L-Measure features net-hurt; the model is already saturated on the features it has
3. **The remaining gap is split** between hard-but-real cells (~50%) and GT noise/ambiguity (~50%). A flag model addresses both: it can punt on either kind.
4. **Cross-model ensemble ceiling is +9.8 pp on P10** — even if you can't reach the oracle, an ensemble-aware flagger captures real value

### 11.8 Quick sanity checks before you start

When your flag model is trained, sanity-check by comparing to these reference operating points:

| Threshold strategy | Expected F1 P10 on kept cells | Expected % cells flagged |
|---|---|---|
| No flagging (baseline) | 0.5908 | 0% |
| Hand rules at 10% rejection | 0.5758 (WORSE!) | 10% |
| Oracle (uses GT — upper bound) at 10% rejection | 0.9331 | 10% |
| **A good ML flagger at 10% rejection** | **~0.70 - 0.85** | 10% |

If your flag model at 10% rejection lifts P10 above **0.70**, you're doing real work. Above **0.80** is excellent. Above **0.85** is approaching the oracle ceiling.

### 11.9 People + context

- Project: SWC-Studio neuron auto-labeler. Owner has been iterating on the v12 pipeline since ~May 2026.
- This handoff was written by Claude on 2026-05-27 during an extended experiment-sweep session.
- Previous handoff (T2/T4 results, L-Measure investigation, bad-lab cleanup) is documented in `paper/PIPELINE_EXPERIMENTS.md` and the results files referenced above.

### 11.10 Canonical artifacts — where each thing lives

**Models on disk (paper/models/)**

| Path | What it is | Trained on |
|---|---|---|
| `baselines/` | 4 external baselines: `neurom_rf.pkl`, `lmeasure_rf.pkl`, `sholl_rf.pkl`, `sholl_mlp.pkl` | cleaned 11,862-corpus, seed=123 80/20 split |
| **`v12_gentle_seed123/`** | **CURRENT CANONICAL full model.** Stage 1 + Stage 2 + apical/basal GNN + conservative Branch3 + QC | cleaned 11,862-corpus, seed=123, 80/20 split |
| `v12_gentle_seed42/` | Full v12 pipeline, all 4 stages trained from scratch | uncleaned 12,484-corpus, seed=42, 80/20 split |
| `v12_gentle_seed789/` | Same as above, different seed | uncleaned 12,484-corpus, seed=789, 80/20 split |
| `v12_pyramidal_only_seed2024_clean/` | Historical pyramidal-only model. Stage 2 pyr + apical/basal GNN + Branch3 rescue trained; Stage 1 + interneuron Stage 2 + QC inherited from seed=42 | cleaned 11,862-corpus, seed=2024, 50/50 split on pyramidals (4,225 train pyr) |

**Per-cell labeled data (paper/results/) — your training data for the flag model**

| File | Cells | Source / Config |
|---|---|---|
| `v12_gt_celltype_seed123.csv/json` | 2,402 | current v12+Branch3 seed=123, GT cell-type override, clean no-leak evaluation |
| `v12_seed123_vs_baselines_table.txt/json` | 2,402 | current v12+Branch3 vs rerun baselines on same seed=123 split |
| `heldout_per_cell_f1_seed123_branch3.csv` | 2,402 | current flag labels, Stage 1 active, clean no-leak test split |
| `flag_confidence_features_seed123_branch3.csv` | 2,402 | current flag features, including Branch3-vs-no-Branch3 disagreement |
| `v12_pyramidal_only_seed2024_clean_eval.csv` | 4,263 | v12 pyramidal-only, Stage 1 bypassed, cleaned corpus |
| `v12_pyramidal_only_seed2024_clean_eval.json` | (summary) | aggregate metrics, per-class P/R/F1, confusion matrix |
| `v12_pyramidal_only_seed2024_clean_branch3_eval.csv/json/txt` | 4,263 | same split with Branch3 rescue enabled (adopted default) |
| `v12_pyramidal_only_seed2024_clean_branch3_loose_eval.csv/json/txt` | 4,263 | same split with looser Branch3 thresholds; better P10, more regressions |
| `lmeasure_pyramidal_only_eval.csv` | 4,263 | L-Measure-RF on the same 4,263 cells (use for cross-model features) |
| `lmeasure_pyramidal_only_eval.json` | (summary) | same |
| `v12_pyramidal_only_seed2024_balance1_eval.csv` | 4,263 | v12 with class_balance=1.0 retrain (failed experiment; useful as model variant) |
| `heldout_per_cell_f1_branch3.csv` | 4,176 | historical conservative-Branch3 labeler, Stage 1 active |
| `flag_confidence_features_branch3.csv` | 4,176 | historical flag features, including Branch3-vs-no-Branch3 disagreement |
| `heldout_per_cell_f1.csv` | 4,176 | historical v12 FULL pipeline after stale-row purge |
| `heldout_per_cell_f1_summary.json` | (summary) | aggregate of the above |

**Supplementary signals (paper/results/)**

| File | What it has |
|---|---|
| `gt_failure_modes.csv` | Per-cell failure-mode category labels (APICAL_MISSED, BASAL_LOST, etc.) |
| `gt_failure_modes.md` | Failure-mode taxonomy + rules + examples |
| `stage1_disagreement.csv` | T2 cross-seed Stage 1 disagreement per cell (suspicion score) |
| `stage1_disagreement_summary.json` | T2 summary |
| `dropped_labs_summary.json` | The 21 bad labs already filtered out (lab prefix still useful as feature) |
| `corpus_qc_v12_uncurated.csv` | **The current corpus catalog** — filter on `qc_pass=True` to get the 11,862 active cells |
| `corpus_qc_v12_uncurated.json` | Original QC scan metadata |
| `rejection_sweep.csv/json/txt` | A1 hand-rule rejection results (proves rules don't work; ML needed) |
| `comparison_v12_vs_lmeasure.txt/json` | v12 vs L-Measure side-by-side, Config 2 (Stage 1 bypassed) |
| `v12_vs_baselines_comparison.md` | **Authoritative comparison** v12 vs 4 baselines, both configs, both cell types |
| `experiment_sweep.md` | Wave A/B experiment results (running log) |
| `baselines_on_v12.csv/json/txt` | 4-baseline eval on uncleaned seed=42 split (Config 1) |
| `per_seed_own_test.csv/json` | v12 full pipeline eval on seed 42 + 789 (Config 1) |

**Helper scripts (paper/*.py)**

| Script | What it does |
|---|---|
| `_eval_pyramidal_gt_celltype.py` | **BEST inference reference** — loads model, runs full pipeline on test cells, computes comprehensive metrics |
| `_eval_lmeasure_pyramidal_only.py` | L-Measure baseline inference — useful for cross-model features |
| `_score_heldout_f1.py` | Per-cell F1 + structural fingerprints — produced `heldout_per_cell_f1.csv` |
| `_score_flag_dataset_branch3.py` | Rebuilds current Branch3-aligned flag labels/features |
| `_purge_non_qc_result_rows.py` | Removes stale non-QC rows from result CSVs with a `file` column |
| `_scan_stage1_disagreement.py` | T2 cross-seed scan — produced `stage1_disagreement.csv` |
| `_eval_rejection_sweep.py` | Rejection-curve analysis (oracle vs hand-rules) |
| `_analyze_failure_modes.py` | Failure-mode taxonomy generator — produces `gt_failure_modes.{csv,md}` |
| `_compare_baselines.py` | Generates `comparison_v12_vs_lmeasure.{txt,json}` |
| `_identify_hard_training_cells.py` | Inference on train set → bottom-decile → JSON for hard-cell upweighting |

### 11.11 Reference metric numbers — your baselines/targets

**v12 PRODUCTION FULL PIPELINE (seed=42, Stage 1 active, both cell types)** — this is the model your flag will be applied to:

```
Per-cell-type breakdown (1,776 pyramidals + 692 interneurons on seed=42 held-out):

PYRAMIDAL CELLS (n = 1,776):
  per-node accuracy    = 0.9804
  apical F1 corpus     = 0.8984
  basal F1 corpus      = 0.9273
  axon F1 corpus       = 0.9945
  per-cell F1 mean     = 0.8947
  per-cell F1 P10      = 0.5000     ← weak spot
  per-cell acc P10     = 0.6443

INTERNEURON CELLS (n = 692):
  per-node accuracy    = 0.9655
  basal F1 corpus      = 0.9379     (no apical class for interneurons)
  axon F1 corpus       = 0.9770
  per-cell F1 mean     = 0.9592
  per-cell F1 P10      = 0.9726     ← almost no bad cells
  per-cell acc P10     = 0.9622

BOTH CELL TYPES POOLED (n = 2,468):
  per-cell F1 mean    ≈ 0.91
  per-cell F1 P10     ≈ 0.65   (between pyr 0.50 and int 0.97 — depends on cell-type mix in your test)
```

**Bad-cell base rate from `heldout_per_cell_f1_branch3.csv` (accepted conservative-Branch3 labeler, both cell types, 4,176 cells):**
- F1 < 0.5: 3.4% of cells
- F1 < 0.6: 5.2% of cells
- Per-cell F1 P10: 0.9652

About **80-90% of bad cells are pyramidals** when training/eval is mixed across cell types. Make sure your training data preserves this ratio (use the full 4,358-cell heldout file directly — don't subsample pyramidals only).

**Oracle rejection ceiling (perfect flagger) on the FULL pipeline both-cell-type pool:**

For pyramidals alone (~1,776 cells):
| % rejected | pyr F1 P10 on kept | pyr F1 mean on kept |
|---|---|---|
| 0% | 0.500 | 0.895 |
| 5% | ~0.65 | ~0.93 |
| 10% | ~0.90 | ~0.96 |
| 15% | ~1.00 | ~0.98 |

For interneurons alone (~692 cells):
| % rejected | int F1 P10 on kept | int F1 mean on kept |
|---|---|---|
| 0% | 0.972 | 0.959 |
| 5% | ~0.99 | ~0.98 |
| 10% | 1.00 | ~0.99 |

For both-cell-types pooled at 10% rejection: pyramidals carry essentially all the lift.

**Targets for your flag model** (on full both-cell-type test data at 10% rejection):
| Performance level | F1 P10 (pooled) | F1 P10 (pyramidals only) | Pyramidal recall of F1<0.5 |
|---|---|---|---|
| ✗ Hand-crafted rules (proven failure) | 0.58 | 0.58 | ~6% |
| Minimum acceptable | 0.78 | 0.70 | 60% |
| Good | 0.88 | 0.80 | 75% |
| Excellent | 0.95 | 0.90 | 85% |
| Oracle ceiling | 0.99 | 0.93 | 100% |

**v12 + L-Measure ensemble ceiling (pyramidals only — L-Measure doesn't add much for interneurons since v12 already nails them)**:
- best-of-two pyramidal F1 mean = 0.9452 (vs v12 alone 0.9148, +3.04 pp)
- best-of-two pyramidal F1 P10 = 0.6887 (vs v12 alone 0.5908, +9.79 pp)

→ Cross-model features carry signal for pyramidal flagging.

---

**Last updated**: 2026-05-27 by Claude during the Wave A/B experiment sweep.
