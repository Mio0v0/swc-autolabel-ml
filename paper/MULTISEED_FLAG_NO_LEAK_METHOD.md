# Multi-Seed Flag Model, No-Leak Method

Date: 2026-06-09

This note defines the paper-safe way to expand the flag/rejection model dataset
from one seed to multiple v12+Branch3 seeds.

## What The Main V12 Tables Mean

The current `v12_gt_celltype_seed*.{json,csv}` and
`v12_seed*_vs_baselines_table.*` files measure the labeler only.

They do not include the flag model. They are before rejection/flagging.

The baseline comparison also has no flag model. Stage 1 is bypassed by feeding
GT cell type directly, which is correct for the apples-to-apples labeler
comparison.

Flag-model claims need separate before/after rejection tables.

## Goal

Train and evaluate a per-cell flagger for the event:

`bad = per_cell_neurite_F1 < 0.6`

The flagger should use only deployment-safe features from model predictions,
confidence summaries, geometry, and optional model-disagreement features. It
must not use GT-derived columns such as per-class F1, accuracy, GT class
fractions, or `stage1_correct` as features.

## Safe Row Definition

One row is:

`(model_seed, file)`

where `file` is in the held-out test split of `model_seed`.

This gives an out-of-fold F1 label: the v12 model that produced the prediction
did not train on that cell.

Canonical row sources to build:

| Seed | Model directory | Split | Output prefix |
|---|---|---|---|
| 123 | `paper/models/v12_gentle_seed123/` | test | `final_flag_seed123_*` |
| 42 | `paper/models/v12_gentle_seed42/` | test | `final_flag_seed42_*` |
| 789 | `paper/models/v12_gentle_seed789/` | test | `final_flag_seed789_*` |

Build each source with:

```powershell
python -m paper._score_model_split_for_flag --model-dir paper/models/v12_gentle_seed123 --split test --out-labels paper/results/final_flag_seed123_labels.csv --out-features paper/results/final_flag_seed123_features.csv --out-summary paper/results/final_flag_seed123_summary.json
python -m paper._score_model_split_for_flag --model-dir paper/models/v12_gentle_seed42 --split test --out-labels paper/results/final_flag_seed42_labels.csv --out-features paper/results/final_flag_seed42_features.csv --out-summary paper/results/final_flag_seed42_summary.json
python -m paper._score_model_split_for_flag --model-dir paper/models/v12_gentle_seed789 --split test --out-labels paper/results/final_flag_seed789_labels.csv --out-features paper/results/final_flag_seed789_features.csv --out-summary paper/results/final_flag_seed789_summary.json
```

## Leakage Rules

Use all of these checks before reporting flag metrics.

1. Labeler leakage check:
   For every row `(model_seed, file)`, `file` must be in that model seed's
   `test` split and must not be in that model seed's `train` split.

2. Flag evaluation cell-overlap check:
   The same `file` must not appear in both flag train/validation and flag test,
   even if it appears under a different `model_seed`.

3. Validation split check:
   The same `file` must not appear in both flag train and flag validation.

4. Feature leakage check:
   Exclude GT/label columns from the feature list:
   `cell_type_gt`, `held_out_F1`, `acc`, `axon_F1`, `basal_F1`, `apical_F1`,
   `stage1_correct`, GT class fractions, GT apical geometry, and timing/debug
   columns.

5. Heavy disagreement feature check:
   Multi-v12 or baseline-disagreement features are allowed for paper research
   only if the reported evaluation still obeys the file-overlap rule above.
   For the strictest setting, only use disagreement predictions from models
   that also held out the evaluated file, or report those features separately as
   a heavier research flagger.

## Evaluation Protocol

Use leave-one-seed-out, with cell de-duplication:

| Fold | Test rows | Train/validation rows |
|---|---|---|
| A | seed 123 rows | seed 42 + 789 rows, after dropping any file also in seed 123 test |
| B | seed 42 rows | seed 123 + 789 rows, after dropping any file also in seed 42 test |
| C | seed 789 rows | seed 123 + 42 rows, after dropping any file also in seed 789 test |

Within each train/validation pool, split by `file`, not by row, so repeated
cells cannot cross the train/validation boundary.

Report mean and standard deviation across the three held-out seed folds.

## Final Production Training

After leave-one-seed-out evaluation is complete, train the production flagger on
all no-leak rows from seeds 123, 42, and 789.

That final all-row flagger is for deployment. Its paper metrics should come
from the held-out folds above, not from evaluating on the same rows used to fit
the final production bundle.

## Required Paper Outputs

| Artifact | Purpose |
|---|---|
| `paper/results/final_flag_multiseed_labels.csv` | Combined out-of-fold flag labels |
| `paper/results/final_flag_multiseed_features.csv` | Combined deployment-safe flag features |
| `paper/results/final_flag_multiseed_leakage_check.{json,txt}` | Proof of no model-train/test or flag cell-overlap leakage |
| `paper/results/final_flag_leave_one_seed_out.csv` | Per-fold precision/recall/rejection metrics |
| `paper/results/final_flag_leave_one_seed_out_table.txt` | Human-readable flag table |
| `paper/results/final_flag_before_after.csv` | Labeler metrics before/after flag rejection |
| `paper/models/final_flag_multiseed/` | Final production flagger trained after evaluation |

## What To Compare

Report these feature sets:

1. Compact: confidence + predicted geometry.
2. Compact + Branch3/no-Branch3 disagreement.
3. Research: + external baseline disagreement.
4. Research: + multi-v12 disagreement.

The compact model is the deployable SWC-Studio path. The research models show
how much extra signal exists if heavier disagreement features are available.
