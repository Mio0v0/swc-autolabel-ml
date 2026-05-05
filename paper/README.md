# Paper deliverables

This directory holds everything needed to reproduce the algorithm-paper
comparison tables. Snapshots in `paper/results/snapshots/` are the
canonical numbers cited in the manuscript; the scripts below regenerate
them from the v9 training data.

## Layout

| Path | What it is |
|---|---|
| `baselines.py` | Floor baselines: random / majority / heuristic + a few legacy ablations (no-stage3, etc.). |
| `external_baselines.py` | SOTA-style external baselines on the v9 test split. **Closest competitor:** Emissah/Tecuatl/Ascoli (bioRxiv 2026), GCN-on-Sholl per-subtree. We mirror their architecture with `sholl_rf` and `sholl_mlp` baselines, plus a NeuroM-style per-branch RF. |
| `significance_tests.py` | Paired-Wilcoxon signed-rank tests across every method pair, Bonferroni-corrected. |
| `run_ablations.py` | Driver for the 5-row ablation grid (no-pca / no-soft-handoff / no-trunk / no-subtree-stage2 / no-gnn) plus a multi-seed stability check. |
| `compile_paper_table.py` | Merges every results JSON into one paper-ready table. |
| `cross_dataset_eval.py` | Zero-shot eval on a different SWC corpus (used for the cross-dataset row in the paper). |
| `bootstrap_ci.py` | 95% CI bootstrap on per-file F1 for the headline numbers. |
| `per_source_breakdown.py` | Splits the v9 per-file F1 distribution by data source (allen / lab / neuromorpho / hpf_ca1) and cell type. |

## Reproducing the comparison table

Run in this order. Each step writes a snapshot JSON / CSV under
`paper/results/snapshots/`.

```bash
# 1. Floor baselines (random / majority / heuristic)         ~5 min
python -m paper.baselines floor

# 2. External baselines (NeuroM-RF, Sholl-RF, Sholl-MLP)    ~30-45 min
python -m paper.external_baselines all \
    --data-dir data/v9_merged_dataset

# 3. Pipeline progression (already snapshotted; re-run if data changed)
#    v6 / v7 / v7b / v8 / v9 — see hybrid/evaluate.py        ~3 hr each

# 4. Bootstrap CIs on v9                                      ~2 min
python -m paper.bootstrap_ci

# 5. Per-source breakdown                                     ~1 min
python -m paper.per_source_breakdown

# 6. Paired Wilcoxon tests across every pair                  ~10 sec
python -m paper.significance_tests

# 7. Compile the unified paper table                          ~5 sec
python -m paper.compile_paper_table
```

The compiled table lands at `paper/results/paper_table.txt` and is
ready to drop into the manuscript.

## Ablation + multi-seed grid

These need full retraining and are budgeted separately. Plan ~20 hr of
CPU compute (or proportionally less with a GPU available for Stage 2).

```bash
# Single ablation
python -m paper.run_ablations no-soft-handoff   # ~5 min, no retrain
python -m paper.run_ablations no-pca            # ~3 hr,  retrain
python -m paper.run_ablations no-trunk          # ~3 hr,  retrain
python -m paper.run_ablations no-subtree-stage2 # ~5 min, no retrain
python -m paper.run_ablations no-gnn            # ~5 min, no retrain

# Multi-seed stability (3 retrains)
python -m paper.run_ablations multi-seed --seeds 42,123,456   # ~9 hr

# All of the above at once
python -m paper.run_ablations all   # ~20 hr
```

### Engine-side patches required for no-PCA and no-trunk

`paper/run_ablations.py` invokes `hybrid/evaluate.py` with environment
variables (`SWCAL_NO_PCA`, `SWCAL_NO_TRUNK`) that the feature
extractors must respect. The patches are documented at the top of
`run_ablations.py`. `no-soft-handoff` is already wired up via the
`SWCAL_NO_SOFT_HANDOFF` env-var hook in `hybrid/pipeline.py`.

## Closest related work — what reviewers will compare us against

| Paper | Year | Architecture | Task | Their reported |
|---|---|---|---|---|
| Emissah, Tecuatl, Ascoli (bioRxiv) | 2026-03 | GCN on Sholl-derived per-subtree features | apical/basal/other on pyramidals only | 99.51% file-level acc, weighted F1 = 0.977 |
| He, Li, Song (Sci Reports) | 2024 | Sugeno fuzzy + multi-classifier fusion | morphological cell-type classification | (different task) |
| Su, Chou, Huang (Neuroinformatics) | 2021 | per-node ML on local features | neuronal polarity (axon/dendrite) | (subset of our task) |
| Vecchi et al. (J Neurosci Methods) | 2021 | NeuriteNet CNN | morphological parameter estimation | (different task) |
| Lu, Zhao, Xie et al. (arXiv) | 2020 | 3D VGG/ResNet on volumetric | tracing-error detection | 74.7% sensitivity / 98.6% spec |

**Ascoli 2026** is the most direct comparison. Our `sholl_rf` and
`sholl_mlp` baselines mirror their feature set on the same v9 test
split so reviewers can see how that recipe lands on this benchmark.

## Snapshots index

`paper/results/snapshots/` is the canonical archive of every number
cited in the paper:

| File | Source step |
|---|---|
| `v6_full_pipeline.json` (+ `_per_file.csv`) | Step 3, v6 baseline |
| `v7_gnn_branch.json` | Step 3, v7 GNN |
| `v7b_gnn_after_s3.json` | Step 3, A/B null-result |
| `v8_subtree_gnn.json` | Step 3, v8 subtree-Stage 2 |
| `v8_cross_dataset_hpf_ca1*.json` | Cross-dataset zero-shot |
| `v9_baseline_no_gnn.json` (+ `.csv`) | v9 without GNN |
| `v9_final_subtree_gnn.json` (+ `.csv`) | **v9 final — current SOTA** |
| `v9_floor_baselines.{json,txt}` | Step 1 |
| `v9_per_source_breakdown.{json,txt}` | Step 5 |
| `v9_bootstrap_ci.{json,txt}` | Step 4 |
| `v9_eval_split.json` | hash-bucket test split |
| `external_baselines.json` (+ `_per_file.csv`) | Step 2 — produced by `paper.external_baselines` |
| `significance_tests.{json,txt}` | Step 6 — produced by `paper.significance_tests` |
| `gnn_sweep_18configs.json` | GNN hyperparameter sweep |

## Citation hooks

When citing the closest related work in the paper:

- Emissah H, Tecuatl C, Ascoli GA. *Automated Proofreading of Digitally Reconstructed Neural Morphology Enhances Accuracy, Scalability, and Standardization.* bioRxiv 2026.03.27.714818. https://doi.org/10.64898/2026.03.27.714818
- He F, Li G, Song H. *Morphological classification of neurons based on Sugeno fuzzy integration and multi-classifier fusion.* Sci Reports 2024;14:16003. https://doi.org/10.1038/s41598-024-66797-1
- Su CZ, Chou KT, Huang HP, et al. *Identification of neuronal polarity by node-based machine learning.* Neuroinformatics 2021;19:669–684. https://doi.org/10.1007/s12021-021-09513-y
- Vecchi JT, Mullan S, Lopez JA, Schultz SR. *NeuriteNet: a convolutional neural network for assessing morphological parameters of neurite growth.* J Neurosci Methods 2021;363:109349.
- Lu D, Zhao S, Xie P, et al. *Quality Control of Neuron Reconstruction Based on Deep Learning.* arXiv:2003.08556, 2020.
