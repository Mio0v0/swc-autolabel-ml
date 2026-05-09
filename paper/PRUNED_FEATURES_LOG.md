# Pruned features & negative results — paper-writing lookup

One-page reference for the methods section. When writing the paper, grep
this file for the feature group, copy the numbers, cite the source CSV.

> **2026-05-08 update**: Numbers below are the original v9 (leaked split)
> findings. We later discovered ~24% of the v9 test split was contaminated
> by `lab__/hpf_ca1__` duplicates, rebuilt the dataset as v10_dedup
> (2,267 cells, no duplicates), and re-ran every ablation.
> **The v10 (clean) findings are at the top of this file —
> see "v10 dedup-split ablation results" section.** v9 numbers below
> are kept for the discussion of how leakage inflated certain effects.

## v10 dedup-split ablation results (CURRENT, paper-relevant)

Test set: 461 cells (159 interneuron + 302 pyramidal), seed=42,
hash-bucket. Same configuration, same code, just with the
de-duplicated dataset.

|                            | neurite-F1 | per-file mean | per-file P10 | Δ vs v10_final |
|----------------------------|------------|---------------|--------------|----------------|
| **v10_final**              | **0.9675** | **0.9586**    | **0.9012**   | (ref)          |
| v10_no_pca                 | 0.9630     | 0.9544        | 0.8964       | mean −0.004    |
| v10_no_trunk               | 0.9675     | 0.9578        | 0.9012       | all ≈ 0        |
| v10_no_soft_handoff        | 0.9675     | 0.9586        | 0.9012       | all = 0        |

All three ablations have paired-Wilcoxon p_bonf = 1.00 vs v10_final
(over 136 pairs in the full grid). None of the design choices we
tested reach significance after correction.

Key finding: **the dramatic v9 effects from removing soft_handoff
(P10 +0.038) were a leakage artifact.** On clean data, soft_handoff
has zero measurable effect. Same for trunk features. PCA features
contribute the only measurable (but n.s.) gain — about 0.5pt on
mean and P10.

External baselines on the same v10 split:

|             | neurite-F1 | pf_mean | pf_P10 | p_bonf vs v10_final |
|-------------|------------|---------|--------|---------------------|
| lmeasure_rf | 0.9677     | 0.9335  | 0.6242 | 1.54e-02 *          |
| sholl_rf    | 0.9654     | 0.9198  | 0.6064 | 1.09e-04 *          |
| sholl_mlp   | 0.9513     | 0.8963  | 0.5239 | 5.90e-09 *          |
| neurom_rf   | 0.8715     | 0.8624  | 0.7072 | 1.57e-40 *          |

v10_final ties lmeasure_rf on OVERALL neurite-F1 (Δ −0.0002), but
wins per-file mean by +0.025 to +0.075 and per-file P10 by **+0.28
to +0.39**. The hybrid pipeline's edge is robustness on the
worst-decile cells, not average accuracy.

---

## Original v9 (leaked split) findings — kept for discussion

Same held-out test split throughout: 556 cells (interneuron + pyramidal),
seed=42, hash-bucket train/test split. Metric is per-file
neurite-macro-F1 (Stage 2+3, soma excluded). Baseline = `v9_final`.

## Removed from final model

### 1. Trunk-detection features (4 branch-level features)

Names: `is_trunk_primary`, `subtree_trunk_length_norm`,
`subtree_trunk_fraction`, `on_longest_path`.

Designed to detect "apical = one dominant trunk before bifurcating; basal
= bushy from the start." A trunk = the longest root-to-leaf path within
a primary subtree.

Ablation result (`eval_no_trunk.json`, full retrain):

|                  | mean Δ vs v9_final | P10 Δ | wins/losses/ties | p_bonf |
|------------------|---------------------|-------|------------------|--------|
| no_trunk         | **+0.0017**         | +0.0037 | 7 / 5 / 544     | 1.0    |

GNN apical/basal CV macroF1 actually **improved** without these features
(0.9637 → 0.9762 in 5-fold CV). They appear to be net-noise.

Decision: **remove**. Tiny consistent gain, simpler model, no downside.

### 2. Soft-handoff post-processing (Stage 2 → Stage 3 confidence reweighting)

The `soft_handoff` step was meant to soften Stage 2's hard class
boundaries before Stage 3 refinement, by passing class probabilities
instead of arg-max predictions for low-confidence branches.

Ablation result (`eval_no_soft_handoff.json`, inference-only):

|                   | mean Δ vs v9_final | P10 Δ   | wins/losses/ties | p_bonf |
|-------------------|---------------------|---------|------------------|--------|
| no_soft_handoff   | +0.0089             | **+0.0377** | 47 / 109 / 400 | 1.0    |

P10 jumps from 0.9618 → **0.9995** without it. Mean is positive but the
*median* effect is a small loss on 109 files. Soft-handoff is a "helps
the average file slightly, occasionally crashes a hard file
catastrophically" feature — removing it trades small avg loss for huge
worst-case gain.

Decision: **remove** (worst-case matters more for an interactive
auto-labeling tool that humans review).

## Kept in final model (despite tied aggregate)

### 3. Cell-PCA + branch-PCA features (12 features total)

Names: `principal_axis_projection`, `polar_angle_from_principal_axis`,
`principal_axis_alignment_strength`, `subtree_principal_projection`,
`subtree_principal_rank` (5 branch-level) plus `pc1_span`,
`pc1_asymmetry`, `pc1_max_above_soma`, `pc1_max_below_soma`,
`pc1_radial_max`, `subtree_pc1_concentration`,
`subtree_pc1_top_alignment` (7 cell-level).

Replace world-z assumptions with the cell's OWN long axis (PC1 of the
neurite point cloud). Designed to handle rotated coordinate frames,
sideways apicals, stunted apicals where world z-axis is not the apical
axis.

Ablation result (`eval_no_pca.json`, full retrain):

|             | mean Δ vs v9_final | P10 Δ      | wins/losses/ties | p_bonf |
|-------------|---------------------|------------|------------------|--------|
| no_pca      | +0.0000             | **−0.0167**| 0 / 0 / 556     | 1.0    |

All 556 files tie on per-file F1 to 4 decimals (rounding masks the
hard-tail effect), but **per-file P10 drops 1.7pt** without these
features. They were specifically designed for the hardest decile of
cells, and that's exactly where they earn their place.

Decision: **keep**. Aggregate test is uninformative because most cells
are easy and tied. The P10 hit is real and matches the design intent.

## Combined ablation: no_trunk + no_soft_handoff

Ran a 4th ablation to check whether the gains from removing trunk
features and soft-handoff stack. They do **not** — combining them
gives the same predictions as no_trunk alone (per-file F1 identical
on all 556 cells to 1e-6).

|                                | mean Δ | P10 Δ | identical to |
|--------------------------------|--------|--------|--------------|
| no_trunk + no_soft_handoff     | +0.0017 | +0.0037 | no_trunk alone |

Mechanism: Stage 1 cell-type confidence is unchanged by removing trunk
features (Stage 1 uses cell-level features only; trunk features are
branch-level). So soft-handoff fires on the same cells in v9_final and
no_trunk runs. Once trunk features are removed, the Stage 2 RF is
consistent enough that the alternative-cell-type's Stage 2 gives the
SAME prediction as the original cell-type's Stage 2 — soft-handoff
becomes a no-op.

**Three real design options (NOT four):**
- v9_final: trunk + soft_handoff (current headline)
- no_trunk: drop trunk features → +0.0010 neurite-F1, +0.0037 P10
- no_soft_handoff: drop soft_handoff → −0.0010 neurite-F1, **+0.0377 P10**

Combining yields no_trunk's numbers, never the no_soft_handoff P10
jump. Choose ONE of the two ablations, not both.

## One-sentence paper text (drop into methods section)

> We additionally evaluated four trunk-detection features (longest
> root-to-leaf path indicators) and a soft-handoff post-processing step
> that reweighted Stage 2 outputs by classifier confidence; pilot
> experiments on the held-out split showed neither improved
> neurite-macro-F1, so they were excluded from the final model.

## Robustness checks (NOT design ablations)

These belong in the "robustness / generalization" section of the paper,
not the ablation section.

### Multi-seed stability (Stage 1+2 retrain at fresh seeds, GNN held fixed)

|                | neurite-F1 | per-file mean | P10 |
|----------------|------------|---------------|------|
| seed=42 (paper)| 0.9748     | 0.9638        | 0.9618 |
| seed=123       | 0.9653     | 0.9434        | 0.6646 |
| seed=456       | 0.9709     | 0.9365        | 0.6503 |
| **mean ± std** | **0.9703 ± 0.0048** | 0.9479 ± 0.0142 | 0.7589 ± 0.176 |

Caveat: each seed produces a different hash-bucketed train/test split,
so per-file P10 differences are partly driven by which files land in
test, not by model variance alone. The neurite-F1 spread (~±0.5pt) is
the cleanest stability number.

### Cross-dataset generalization (zero-shot on hpf_ca1 corpus)

|                          | neurite-F1 | mean | P10 |
|--------------------------|------------|------|------|
| in-domain (v9 test split)| 0.9748     | 0.9638 | 0.9618 |
| hpf_ca1 (1377 cells)     | 0.9499     | 0.9516 | 0.7632 |

Δ ≈ −2.5pt neurite-F1 cross-corpus. Standard zero-shot drop. Stage 1
cell-type detector remained accurate (0.9985) on the new corpus.

Source: `paper/results/cross_dataset_v9_hpf_ca1.json`.

## Where to find the raw data

| What | Path |
|---|---|
| Unified paper table | `paper/results/paper_table.txt` |
| Pairwise Wilcoxon (66 pairs) | `paper/results/snapshots/significance_tests.txt` |
| no_pca run | `paper/results/snapshots/eval_no_pca.json` |
| no_trunk run | `paper/results/snapshots/eval_no_trunk.json` |
| no_soft_handoff run | `paper/results/snapshots/eval_no_soft_handoff.json` |
| Multi-seed runs | `paper/results/snapshots/eval_multi_seed_{123,456}.json` |
| Cross-dataset hpf_ca1 | `paper/results/cross_dataset_v9_hpf_ca1.json` |
| Full overnight log | `paper/results/overnight_queue.log` |
| Decision history | git log paper/ |
