# Flag Model Trial Summary

Target bad label: `held_out_F1 < 0.60`. Precision means `actually bad among flagged / all flagged`; recall means `actually bad flagged / all actually bad`.

## Pyramidal Progression

| Flag model family | Target recall setting | Precision | Recall | Flagged | Actually bad flagged | False flags | Missed bad | Model |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Original confidence/geometry | 80.0% | 38.8% | 74.3% | 67 | 26 | 41 | 9 | `rf:full` |
| + morphology baseline disagreement | 80.0% | 52.8% | 80.0% | 53 | 28 | 25 | 7 | `rf:full` |
| + sanity/failure heads, best 70% target | 70.0% | 54.3% | 71.4% | 46 | 25 | 21 | 10 | `gb:full` |
| + sanity/failure heads, high recall | 80.0% | 33.7% | 91.4% | 95 | 32 | 63 | 3 | `rf:numeric_only` |
| + multi-v12 disagreement, recommended | 80.0% | 78.9% | 85.7% | 38 | 30 | 8 | 5 | `hgb:full` |
| + multi-v12 disagreement, high recall | 90.0% | 39.0% | 91.4% | 82 | 32 | 50 | 3 | `extra_trees:numeric_only` |

## Current Best Multi-V12 Operating Points

| Scope | Target recall setting | Precision | Recall | Flagged | Actually bad flagged | False flags | Missed bad | Actually bad total | Experiment/model |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pyramidal | 70.0% | 78.4% | 82.9% | 37 | 29 | 8 | 6 | 35 | `single_cell_f060/cell_f060 / hgb:full` |
| pyramidal | 80.0% | 78.9% | 85.7% | 38 | 30 | 8 | 5 | 35 | `single_cell_f060/cell_f060 / hgb:full` |
| pyramidal | 90.0% | 39.0% | 91.4% | 82 | 32 | 50 | 3 | 35 | `single_failure_head_ranked_against_cell_f060/basal_f060 / extra_trees:numeric_only` |
| interneuron | 90.0% | 60.0% | 85.7% | 10 | 6 | 4 | 1 | 7 | `single_failure_head_ranked_against_cell_f060/axon_f050 / rf:full` |
| all | 80.0% | 53.8% | 83.3% | 65 | 35 | 30 | 7 | 42 | `two_stage_max_failure_heads/cell_f060+axon_f050+basal_f060+apical_f050 / two_stage:max_validation_percentile` |

## Recommendation

Use the multi-v12 pyramidal `r080` bundle for the practical operating point: **78.9% precision, 85.7% recall**, 38 flagged cells, 30 actually bad, 8 false flags, 5 missed bad.

Bundle: `paper/models/flag_improvement_sweep_seed123_branch3_v12ens/pyramidal_xmodel_cell_f060_r080.joblib`

Full detailed trial table: `paper/results/flag_model_trials_clean_table.csv`
