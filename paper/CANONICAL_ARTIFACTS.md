# Canonical Artifacts

Date: 2026-06-08

This manifest defines what should remain in the active `paper/` tree for the
current paper/model story, and what can be archived. Use this file before
deleting or moving artifacts.

## Production Model Artifacts

Keep these in `paper/models/` unless a newer accepted model explicitly replaces
them.

| Artifact | Role | Notes |
|---|---|---|
| `paper/models/v12_gentle_seed123/` | Current accepted v12+Branch3 labeler | Seed 123, cleaned corpus, 80/20 split, conservative Branch3 |
| `paper/models/flag_model_seed123_branch3_pyramidal/` | Current compact pyramidal flagger | Deployable from one v12+Branch3 inference pass |
| `paper/models/flag_improvement_sweep_seed123_branch3_v12ens/` | Best research flagger | Uses multi-v12 / cross-model disagreement features |
| `paper/models/baselines/` | External baseline models | Needed for reproducible baseline comparison and disagreement features |
| `paper/models/flag_model_seed123_branch3/` | Supporting compact all-cell flagger | Keep as fallback/supporting artifact, not the primary paper flagger |

Model folders to keep for now, but not final-paper canonical:

| Artifact | Role | Notes |
|---|---|---|
| `paper/models/v12_gentle_seed42/` | Supporting final multi-seed model | Keep for paper reproducibility |
| `paper/models/v12_gentle_seed789/` | Supporting final multi-seed model | Keep for paper reproducibility |

## Paper Result Artifacts

Keep these active because they support the current paper narrative or are direct
inputs for future paper tables.

| Artifact | Role |
|---|---|
| `paper/results/corpus_qc_v12_uncurated.csv` | Clean corpus/QC catalog |
| `paper/results/corpus_qc_v12_uncurated.json` | QC summary |
| `paper/results/dropped_labs_summary.json` | Data-cleaning audit |
| `paper/results/purged_non_qc_rows_summary.json` | Result-row cleanup audit |
| `paper/results/v12_gt_celltype_seed123.csv` | Seed123 v12 GT-celltype per-cell eval |
| `paper/results/v12_gt_celltype_seed123.json` | Seed123 v12 GT-celltype summary |
| `paper/results/v12_seed123_vs_baselines_table.txt` | Current seed123 comparison table |
| `paper/results/v12_seed123_vs_baselines_table.json` | Current seed123 comparison data |
| `paper/results/baselines_on_v12.csv` | Baseline predictions/metrics on v12 split |
| `paper/results/baselines_on_v12.json` | Baseline summary on v12 split |
| `paper/results/baselines_on_v12_table.txt` | Baseline short table |
| `paper/results/leakage_check_seed123.txt` | Split leakage check |
| `paper/results/leakage_check_seed123.json` | Split leakage check data |
| `paper/results/heldout_per_cell_f1_seed123_branch3.csv` | Current flag labels |
| `paper/results/flag_confidence_features_seed123_branch3.csv` | Current compact flag features |
| `paper/results/flag_seed123_branch3_dataset_summary.json` | Current flag dataset summary |
| `paper/results/flag_model_seed123_branch3_pyramidal_eval.*` | Compact pyramidal flag eval |
| `paper/results/flag_model_current_best_table.csv` | Current best flag summary |
| `paper/results/flag_model_trials_summary_table.csv` | Human-readable flag trial summary |
| `paper/results/flag_model_trials_clean_table.csv` | Full flag trial table |
| `paper/results/flag_improvement_sweep_seed123_branch3_v12ens.*` | Best research flag sweep |
| `paper/results/flag_features_seed123_branch3_xmodel_v12ens.*` | Research flag feature matrix |
| `paper/results/flag_v12_ensemble_features_seed123_branch3.csv` | Multi-v12 disagreement features |
| `paper/results/flag_improvement_best_summary_seed123_branch3.txt` | Best flag narrative summary |
| `paper/results/gt_failure_modes.csv` | Failure-mode labels |
| `paper/results/gt_failure_modes.md` | Failure-mode analysis |
| `paper/results/stage1_disagreement.csv` | Stage1 disagreement feature source |
| `paper/results/stage1_disagreement_summary.json` | Stage1 disagreement summary |
| `paper/results/final_dataset_corpus_table.{csv,json,md}` | Final dataset/QC paper table |
| `paper/results/final_dataset_qc_fail_reasons.csv` | Final QC failure reason counts |
| `paper/results/final_multiseed_v12_branch3_summary.{csv,json}` | Final v12+Branch3 multi-seed summary |
| `paper/results/final_multiseed_v12_branch3_table.txt` | Final v12+Branch3 human-readable table |
| `paper/results/final_multiseed_vs_baselines.{csv,json}` | Final multi-seed baseline comparison |
| `paper/results/final_multiseed_vs_baselines_table.txt` | Final multi-seed baseline comparison table |
| `paper/results/final_ablation_table.{csv,json,txt}` | Final architecture ablation ladder |
| `paper/results/final_flag_multiseed_cells.csv` | Final no-leak flag labels/cells |
| `paper/results/final_flag_multiseed_features*.csv` | Final no-leak flag feature matrices |
| `paper/results/final_flag_leave_one_seed_out*` | Final no-leak flag evaluation |
| `paper/results/final_flag_feature_ablation.{csv,json,txt}` | Final flag feature ablation table |
| `paper/results/final_failure_modes.{csv,json,md}` | Final failure-mode analysis |
| `paper/results/final_failure_mode_flag_scores.csv` | Row-level flag decisions used by failure-mode analysis |
| `paper/results/figures/final_figure_confusion_matrix.{png,svg,csv}` | Final accepted-model confusion matrix figure |
| `paper/results/figures/final_figure_f1_distribution.{png,svg}` | Final per-cell F1 before/after flag rejection figure |
| `paper/results/figures/final_figure_f1_distribution_summary.json` | Summary statistics for the F1 rejection figure |
| `paper/results/figures/final_figure_flag_pr_curve.{png,svg,csv}` | Final no-leak flag precision-recall figure |
| `paper/results/figures/final_figures_manifest.json` | Final figure manifest |

Optional/history result artifacts moved to `paper/archive/results/`; use them
there only if the final ablation table needs historical context:

| Artifact family | Role |
|---|---|
| `paper/results/v12_pyramidal_only_seed2024_clean*` | Historical pyramidal-only/Branch3 ablation |
| `paper/results/lmeasure_pyramidal_only_eval.*` | Historical L-Measure pyramidal comparison |
| `paper/results/per_seed_own_test.*` | Older multi-seed evaluation reference |
| `paper/results/rejection_sweep.*` | Older rejection/oracle reference |

## Historical / Archivable Artifacts

Move these to `paper/archive/` instead of deleting immediately.

### Model Folders

| Artifact | Archive destination | Reason |
|---|---|---|
| `paper/models/flag_model/` | `paper/archive/models/flag_model/` | Older flag generation |
| `paper/models/flag_model_branch3/` | `paper/archive/models/flag_model_branch3/` | Older Branch3 flag generation |
| `paper/models/flag_model_branch3_p01/` | `paper/archive/models/flag_model_branch3_p01/` | Older Branch3 flag generation |
| `paper/models/flag_model_seed123_branch3_pyramidal_p02/` | `paper/archive/models/flag_model_seed123_branch3_pyramidal_p02/` | Superseded pyramidal flag variant |
| `paper/models/flag_model_seed123_branch3_pyramidal_highrecall/` | `paper/archive/models/flag_model_seed123_branch3_pyramidal_highrecall/` | High-recall variant, not canonical |
| `paper/models/flag_model_seed123_branch3_pyramidal_xmodel/` | `paper/archive/models/flag_model_seed123_branch3_pyramidal_xmodel/` | Superseded by v12ens research flagger |
| `paper/models/flag_model_seed123_branch3_pyramidal_xmodel_highrecall/` | `paper/archive/models/flag_model_seed123_branch3_pyramidal_xmodel_highrecall/` | High-recall variant, not canonical |
| `paper/models/flag_improvement_sweep_seed123_branch3/` | `paper/archive/models/flag_improvement_sweep_seed123_branch3/` | Superseded by v12ens sweep |
| `paper/models/v12_pyramidal_only_seed2024_clean/` | `paper/archive/models/v12_pyramidal_only_seed2024_clean/` | Historical pyramidal-only experiment; archive only after ablation references are copied |

### Result Files

Archive these result families:

| Result family | Archive destination | Reason |
|---|---|---|
| `paper/results/flag_model_branch3*` | `paper/archive/results/` | Older flag generation |
| `paper/results/flag_model_eval*` | `paper/archive/results/` | Older flag generation |
| `paper/results/flag_branch3_dataset_summary.json` | `paper/archive/results/` | Older flag dataset summary |
| `paper/results/flag_confidence_features.csv` | `paper/archive/results/` | Older flag feature set |
| `paper/results/flag_confidence_features_branch3.csv` | `paper/archive/results/` | Older flag feature set |
| `paper/results/flag_model_seed123_branch3_pyramidal_p02*` | `paper/archive/results/` | Superseded flag variant |
| `paper/results/flag_high_recall*` | `paper/archive/results/` | High-recall exploratory run |
| `paper/results/flag_candidate_high_recall*` | `paper/archive/results/` | High-recall exploratory run |
| `paper/results/flag_xmodel_high_recall*` | `paper/archive/results/` | High-recall exploratory run |
| `paper/results/flag_xmodel_highrecall*` | `paper/archive/results/` | High-recall exploratory run |
| `paper/results/v12_pyramidal_only_seed2024_*` | `paper/archive/results/` | Historical pyramidal-only results |
| `paper/results/apical_owner_prob_scan.csv` | `paper/archive/results/` | Historical Branch3/apical-owner scan |
| `paper/results/comparison_v12_vs_lmeasure.*` | `paper/archive/results/` | Historical L-Measure analysis |
| `paper/results/v12_lmeasure_only_fusion_sweep_seed123.*` | `paper/archive/results/` | Exploratory fusion sweep |

Safe to delete after this manifest exists:

| File family | Action |
|---|---|
| `paper/**/__pycache__/` | Delete |
| `paper/results/tmp_*` | Delete |
| `paper/results/logs/*.pid` | Delete |
| stale `.log` files | Archive or delete only when not needed for reproducibility |

### Scripts

Keep active scripts:

| Script | Role |
|---|---|
| `paper/_retrain_v12_gentle_seed.py` | Main v12 retraining |
| `paper/gnn_branch3_rescue.py` | Branch3 rescue training |
| `paper/train_branch3_gate.py` | Optional Branch3 gate training |
| `paper/_score_model_split_for_flag.py` | Current flag dataset scoring |
| `paper/_score_flag_dataset_branch3.py` | Helper source used by current flag scoring |
| `paper/_score_heldout_f1.py` | Helper source used by current flag scoring |
| `paper/_train_flag_model.py` | Current flag model training |
| `paper/_eval_v12_gt_celltype.py` | v12 GT-celltype evaluation |
| `paper/_eval_baselines_on_v12.py` | Baseline rerun on v12 split |
| `paper/_compare_v12_seed_to_baselines.py` | Seed-vs-baseline table builder |
| `paper/_check_split_leakage.py` | Leakage audit |
| `paper/_sweep_flag_improvements.py` | Flag improvement sweep |
| `paper/_build_flag_v12_ensemble_features.py` | Multi-v12 flag features |
| `paper/_build_flag_confidence_features.py` | Compact flag feature builder |
| `paper/_build_flag_cross_model_features.py` | Cross-model flag features |
| `paper/_build_final_flag_oof_features.py` | No-leak out-of-fold heavy flag features |
| `paper/_combine_final_flag_multiseed.py` | Final multi-seed flag dataset combiner |
| `paper/_train_eval_final_flag_multiseed.py` | Final leave-one-seed-out flag training/evaluation |
| `paper/_compile_final_flag_tables.py` | Final flag table compiler |
| `paper/_compile_multiseed_paper_tables.py` | Final multi-seed v12/baseline table compiler |
| `paper/_compile_canonical_paper_outputs.py` | Canonical result alias and dataset table compiler |
| `paper/_analyze_final_failure_modes.py` | Final failure-mode and flag-catch analysis |
| `paper/_make_final_paper_figures.py` | Final result figure generator |
| `paper/_run_paired_paper_stats.py` | Paired seed-level statistics |
| `paper/_run_v12_architecture_ablation.py` | Final architecture ablation runner |
| `paper/_scan_corpus_qc.py` | QC catalog builder |
| `paper/_purge_non_qc_result_rows.py` | QC-row cleanup utility |
| `paper/baselines.py` | Baseline framework |
| `paper/external_baselines.py` | External baseline implementation |

Archive one-off scripts only after checking references with `rg`:

| Script | Archive reason |
|---|---|
| `_backup_gnns_before_retrain.py` | Safety utility, not paper pipeline |
| `_restore_best_gnn.py` | Safety utility, not paper pipeline |
| `_eval_lmeasure_pyramidal_only.py` | Historical L-Measure/pyramidal-only eval |
| `_eval_pyramidal_gt_celltype.py` | Historical pyramidal-only eval |
| `_eval_v12_lmeasure_fusion.py` | Exploratory fusion |
| `_identify_hard_training_cells.py` | Exploratory hard-cell analysis |
| `_make_architecture_slides.py` | Slide generator |
| `_remake_v11_figures.py` | Old v11 figures |
| `_train_pyramidal_only.py` | Historical pyramidal-only training |
