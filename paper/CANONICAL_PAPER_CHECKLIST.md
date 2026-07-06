# Canonical Paper Checklist

Date: 2026-06-08

This file is the paper-facing source of truth for the current SWC auto-labeling
model, the required artifacts to keep, and the computations still needed before
writing/submission. Older experiment logs can stay useful, but paper claims
should point back to this checklist and the artifacts listed here.

## 1. Current Accepted Model

Use this as the canonical labeler unless a newer run explicitly replaces it.

| Item | Canonical artifact | Status |
|---|---|---|
| Full v12 labeler | `paper/models/v12_gentle_seed123/` | Accepted current model |
| Stage 1 cell type | `paper/models/v12_gentle_seed123/cell_type_classifier.pkl` | Keep |
| Stage 2 branch classifier | `paper/models/v12_gentle_seed123/branch_classifier.pkl` | Keep |
| Apical/basal GNN | `paper/models/v12_gentle_seed123/gnn_apical_basal.pt` | Keep |
| Conservative Branch3 rescue | `paper/models/v12_gentle_seed123/gnn_branch3_rescue.pt` | Keep |
| QC gate | `paper/models/v12_gentle_seed123/qc_gate.pkl` | Keep |
| Split definition | `paper/models/v12_gentle_seed123/train_test_split.json` | Keep |
| GNN split definition | `paper/models/v12_gentle_seed123/eval_split_for_gnn.json` | Keep |

Current recipe:

- Corpus: cleaned QC-pass v12 corpus, both pyramidal and interneuron.
- Seed: 123.
- Split: 80/20 hash split.
- Default inference: conservative Branch3 enabled.
- Paper deployment behavior: unknown cell type uses Stage 1; user-provided cell
  type can bypass Stage 1 in SWC-Studio.

## 2. Current Flag Models

| Item | Artifact | Status |
|---|---|---|
| Primary compact pyramidal flagger | `paper/models/flag_model_seed123_branch3_pyramidal/flag_model_f060.joblib` | Current deployable compact flagger |
| All-cell compact flagger | `paper/models/flag_model_seed123_branch3/flag_model_f060.joblib` | Current fallback/all-cell compact flagger |
| Best research flagger | `paper/models/flag_improvement_sweep_seed123_branch3_v12ens/pyramidal_xmodel_cell_f060_r080.joblib` | Best paper result, heavier feature set |

Important distinction:

- The compact flagger can run from one v12+Branch3 inference pass.
- The best research flagger uses multi-v12 / cross-model disagreement features
  and is better for paper analysis, but it is not yet the lightweight SWC-Studio
  default.

## 3. Current Canonical Results Already Computed

| Claim area | Artifact | Status |
|---|---|---|
| Clean corpus/QC catalog | `paper/results/corpus_qc_v12_uncurated.csv` | Computed |
| Seed123 v12 GT-celltype eval | `paper/results/v12_gt_celltype_seed123.{json,csv}` | Computed |
| Seed123 vs baselines table | `paper/results/v12_seed123_vs_baselines_table.{txt,json}` | Computed |
| Baselines on v12 split | `paper/results/baselines_on_v12.{csv,json,txt}` | Computed |
| No-leak check | `paper/results/leakage_check_seed123.{txt,json}` | Computed |
| Seed123 per-cell F1 for flag | `paper/results/heldout_per_cell_f1_seed123_branch3.csv` | Computed |
| Seed123 flag features | `paper/results/flag_confidence_features_seed123_branch3.csv` | Computed |
| Seed123 flag dataset summary | `paper/results/flag_seed123_branch3_dataset_summary.json` | Computed |
| Flag trial summary table | `paper/results/flag_model_trials_summary_table.csv` | Computed |
| Current best flag summary | `paper/results/flag_model_current_best_table.csv` | Computed |
| Multi-seed v12 vs baselines | `paper/results/v12_multiseed_vs_baselines_summary.{txt,csv,json}` | Computed 2026-06-09 |
| Branch3 on/off ablation | `paper/results/v12_branch3_ablation_summary.{txt,csv,json}` | Computed 2026-06-09 |
| Seed42/789 baseline reruns | `paper/results/baselines_on_v12_seed{42,789}.{json,csv}` | Computed 2026-06-09 |
| Seed42/789 leakage checks | `paper/results/leakage_check_seed{42,789}.{txt,json}` | Computed 2026-06-09 |
| Multi-seed flag no-leak method | `paper/MULTISEED_FLAG_NO_LEAK_METHOD.md` | Defined 2026-06-09 |
| Final v12-only multi-seed table | `paper/results/final_multiseed_v12_branch3_summary.{csv,json}`, `paper/results/final_multiseed_v12_branch3_table.txt` | Compiled 2026-06-15 |
| Final multi-seed vs baselines table | `paper/results/final_multiseed_vs_baselines.{csv,json}`, `paper/results/final_multiseed_vs_baselines_table.txt` | Compiled 2026-06-15 |
| Final ablation ladder | `paper/results/final_ablation_table.{csv,json,txt}` | Computed 2026-06-09 |
| Final flag no-leak / before-after tables | `paper/results/final_flag_leave_one_seed_out*`, `paper/results/final_flag_before_after.csv` | Computed 2026-06-09 |
| Final flag feature ablation | `paper/results/final_flag_feature_ablation.{csv,json,txt}` | Compiled 2026-06-15 |
| Final dataset/QC table | `paper/results/final_dataset_corpus_table.{csv,json,md}` | Compiled 2026-06-15 |
| Final failure-mode analysis | `paper/results/final_failure_modes.{csv,json,md}` and `paper/results/final_failure_mode_flag_scores.csv` | Computed 2026-06-15 |
| Final paper figures | `paper/results/figures/final_figure_*` and `paper/results/figures/final_figures_manifest.json` | Generated 2026-06-15 |

## 4. Paper Computations Still Needed

Most compute-heavy paper artifacts are now complete. The remaining work is
mainly figure generation, table selection, and manuscript writing. The sections
below are kept as a checklist and note the current status.

### A. Multi-Seed Final Model Evaluation

Goal: avoid relying on one seed for the main model claim.

Status: complete for the current v12+Branch3 recipe. Canonical final aliases
now exist under `paper/results/final_multiseed_v12_branch3_*`.

Completed:

- Confirmed final v12+Branch3 recipe for seeds `42`, `123`, `789`.
- Used the same cleaned corpus rules and the same Branch3 default.
- Evaluated each seed on its own no-leak held-out split.
- Reported mean, standard deviation, min, and max across seeds.

Required outputs:

- `paper/results/final_multiseed_v12_branch3_summary.csv`
- `paper/results/final_multiseed_v12_branch3_summary.json`
- `paper/results/final_multiseed_v12_branch3_table.txt`

Metrics:

- node accuracy
- neurite macro F1
- axon F1
- basal F1
- apical F1
- per-cell F1 mean
- per-cell F1 P10/P25
- per-cell-type breakdown
- confusion matrices

### B. Final Baseline Comparison

Goal: compare final v12+Branch3 against external baselines on the same clean
splits.

Status: complete for seeds `123`, `42`, and `789`; see
`paper/results/final_multiseed_vs_baselines.*`,
`paper/results/v12_multiseed_vs_baselines_summary.*`, and the per-seed
`v12_seed*_vs_baselines_table.*` files.

Baselines:

- NeuroM RF
- L-Measure RF
- Sholl RF
- Sholl MLP

Completed:

- Ran each baseline on the same final held-out split per seed.
- Did not involve the flag model in this comparison.
- Fed GT cell type to baselines and to the v12 GT-celltype comparison when
  making the apples-to-apples labeler table.

Required outputs:

- `paper/results/final_multiseed_vs_baselines.csv`
- `paper/results/final_multiseed_vs_baselines.json`
- `paper/results/final_multiseed_vs_baselines_table.txt`

### C. Ablation Study

Goal: show which architecture parts matter.

Status: complete; see `paper/results/final_ablation_table.*`.

Minimum ablations:

| Ablation | Purpose |
|---|---|
| Stage 2 only | baseline branch classifier without GNN/refinement |
| Stage 2 + apical/basal GNN | isolate GNN gain |
| Stage 2 + GNN + Stage 3 topology refinement | isolate topology refinement |
| Stage 2 + GNN + Stage 3 + Branch3 rescue | accepted full labeler |
| Stage1 active vs GT cell-type override | deployment vs apples-to-apples comparison |
| Full labeler + flag rejection | downstream clean-label set result |

Required outputs:

- `paper/results/final_ablation_table.csv`
- `paper/results/final_ablation_table.json`
- `paper/results/final_ablation_table.txt`

### D. Multi-Seed Flag Dataset

Goal: make flag-model claims stronger and possibly train a better flag model.

Status: complete using the no-leak leave-one-seed-out design.

Completed:

- Scored held-out per-cell F1 for final seeds `42`, `123`, `789`.
- Built matching confidence/geometry/Branch3 and heavy disagreement features.
- Added `model_seed` and stable `(model_seed, file)` row IDs.

Recommended evaluation design:

- Train flagger on two seeds.
- Test flagger on the third seed.
- Rotate leave-one-seed-out.
- Also report a cell-deduplicated test if the same cell can appear across seed
  splits.
- Use the exact no-leak protocol in `paper/MULTISEED_FLAG_NO_LEAK_METHOD.md`.
- Do not split this dataset row-randomly; split by `file` groups.

Required outputs:

- `paper/results/final_flag_multiseed_cells.csv`
- `paper/results/final_flag_multiseed_features.csv`
- `paper/results/final_flag_multiseed_features_oof_heavy.csv`
- `paper/results/final_flag_multiseed_leakage_check.{json,txt}`
- `paper/results/final_flag_leave_one_seed_out.csv`
- `paper/results/final_flag_leave_one_seed_out_table.txt`

Flag metrics:

- bad target: per-cell F1 < 0.6
- precision
- recall
- F1
- flagged total
- true bad flagged
- false flags
- missed bad
- bad total
- rejection rate
- before/after retained-set labeler metrics

### E. Flag Feature Ablation

Goal: justify why the best flagger uses extra disagreement features.

Compare:

| Flag feature set | Purpose |
|---|---|
| confidence + geometry only | compact baseline |
| + Branch3/no-Branch3 disagreement | deployable compact flagger |
| + external baseline disagreement | stronger paper model |
| + multi-v12 disagreement | current best research flagger |

Required outputs:

- `paper/results/final_flag_feature_ablation.csv`
- `paper/results/final_flag_feature_ablation.txt`

Status: complete; JSON is also available at
`paper/results/final_flag_feature_ablation.json`.

### F. Failure-Mode Analysis

Goal: make the paper discussion concrete.

Needed:

- Count dominant failure modes among per-cell F1 < 0.6.
- Break down by cell type and source/lab prefix.
- Show which failures are caught or missed by the flagger.

Required outputs:

- `paper/results/final_failure_modes.csv`
- `paper/results/final_failure_modes.md`
- one or two example-cell figures if the paper needs qualitative examples.

Status: table/report complete. Qualitative example figures remain optional and
should be generated only after choosing which example cells to show.

### G. Remaining Non-Compute Paper Work

Recommended next steps:

- Choose the exact table subset for the manuscript from the canonical outputs.
- Review/refine generated figures in `paper/results/figures/`: confusion
  matrix, per-cell F1 before/after flag rejection, and flag precision-recall.
- Generate the remaining schematic/qualitative figures: pipeline schematic and
  optional example cells.
- Write Methods text for the no-leak split, Branch3 rescue, and flagger
  leave-one-seed-out protocol.
- Decide whether to include the heavier research flagger as a main result or as
  an optional/appendix result, since SWC-Studio deploys the compact path.

## 5. Cleanup/Archive Plan

Do not delete important artifacts until the final tables above exist. Use
archive folders first.

Suggested archive folders:

- `paper/archive/models/`
- `paper/archive/results/`
- `paper/archive/scripts/`

Safe to delete after archive/manifest:

- `paper/**/__pycache__/`
- `paper/results/tmp_*`
- `paper/results/logs/*.pid`
- smoke-test outputs

Archive candidates:

- older flag model generations: `flag_model_branch3*`, `flag_model_seed123_branch3_pyramidal_p02`, high-recall-only bundles
- old pyramidal-only experiments unless cited in ablation/history
- one-off scripts not used by the final pipeline or final paper tables
- stale docs that reference removed scripts

Keep until paper is stable:

- `paper/models/baselines/` because it is needed for baseline reruns and
  xmodel disagreement features.
- `paper/models/v12_gentle_seed123/`
- final seed folders once created/confirmed.
- all final paper result CSV/JSON/TXT files listed above.

## 6. Canonical Paper Tables/Figures

Target final paper tables:

1. Dataset/corpus table: cell counts by type/source before and after QC.
2. Main performance table: v12+Branch3 vs baselines, multi-seed mean/CI.
3. Ablation table: Stage2, GNN, Stage3, Branch3, flag rejection.
4. Flag model table: precision/recall/false flags/missed bad before/after.
5. Per-cell-type table: pyramidal/interneuron metrics.

Target final figures:

1. Pipeline architecture schematic. Needed.
2. Confusion matrix for accepted model. Generated:
   `paper/results/figures/final_figure_confusion_matrix.{png,svg}`.
3. Per-cell F1 distribution before and after flag rejection. Generated:
   `paper/results/figures/final_figure_f1_distribution.{png,svg}`.
4. Flag precision-recall curve. Generated:
   `paper/results/figures/final_figure_flag_pr_curve.{png,svg}`.
5. Optional qualitative examples of caught bad labels. Optional.

## 7. Decision Log

- Current deployable SWC-Studio flagger is compact because it can run from one
  v12 inference pass.
- Current best research flagger uses heavier multi-model disagreement features
  and should be presented as a paper result or optional advanced mode, not as
  the default lightweight app flagger unless those features are integrated.
- Multi-seed flag training is expected to help because it gives more bad-cell
  examples, but final evaluation must avoid leakage by holding out seeds/cells.
