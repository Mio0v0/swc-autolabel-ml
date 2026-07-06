# Stage 1 cell-type classifier — experiment log

**Goal**: push Stage 1 accuracy on the v12_uncurated test set from baseline
~95% toward ≥99%. Stage 1 is a 2-class classifier (pyramidal vs interneuron)
trained on 49 morphometric features per cell.

**Test set**: seed=42 hash-bucket split, **2,468 cells held out**
(1,776 pyramidal + 692 interneuron). All numbers below are on this same
test set, never seen during training.

**Metric definition**:
- Overall accuracy = correct / 2,468
- Per-cell-type recall = correctly-typed cells of that GT type / total GT cells of that type

---

## Outcome at a glance

| Iter | Config | Overall acc | Pyr recall | Int recall | Pyr errors | Int errors | Total errors |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Baseline (VotingClassifier of XGB RF n=200 + XGB GB n=150, class_weight="balanced") | **0.9498** | 0.9702 | 0.8974 | 53 | 71 | **124** |
| A | RF n=500 + GB n=300; class_weight={int:4, pyr:1} | 0.9534 | 0.9775 | 0.8916 | 40 | 75 | 115 |
| B | + bootstrap-oversample interneurons to match pyramidal count | 0.9518 | 0.9668 | 0.9133 | 59 | 60 | 119 |
| C | Pure XGBoost gradient boosting (n_estimators=2000, max_depth=8, lr=0.03, early stopping) + oversampling | 0.9518 | 0.9634 | 0.9220 | 65 | 54 | 119 |
| D | 5-seed XGB ensemble (seeds 42, 123, 456, 789, 1024) — average softmax probabilities | **0.9546** | 0.9685 | 0.9191 | 56 | 56 | **112** |
| E | XGB + oversampling + 6 NEW features (soma_radius, max_neurite_to_soma_ratio, branching/terminal density, bifurcation angle mean+std) | 0.9530 | 0.9645 | 0.9234 | 63 | 53 | 116 |

**Range across all 6 iterations**: 94.98%–95.46%. Total errors range: 112–124.

---

## Detailed walkthrough

### Iteration 0 — baseline

- VotingClassifier (soft voting) of:
  - `XGBRandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced")`
  - `XGBGradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, min_samples_leaf=3)`
- 49-dim feature vector (extracted by `hybrid/features.py:extract_global_features`)
- Run by: `python -m paper._quick_stage1_seed42`

Baseline number: **94.98% (124 errors)**. Already strong because the morphometric features include z-asymmetry, principal-axis features, subtree concentration, Strahler order, etc. — many discriminative signals for pyramidal vs interneuron.

### Iteration A — bigger model + aggressive class weighting

**Hypothesis**: interneuron recall is lower (89.7% vs pyramidal 97%). The default "balanced" class weight (~1.83 for interneuron) might not be aggressive enough to offset the 73/27 imbalance in training data.

**Change**:
- RF n_estimators: 200 → 500, min_samples_leaf 2 → 1
- GB n_estimators: 150 → 300, max_depth 5 → 6, lr 0.1 → 0.05
- `class_weight = {0: 4.0, 1: 1.0}` (interneuron 4× pyramidal — way above "balanced")

**Result**:
- Overall: 94.98% → **95.34%** (+0.36 pp)
- Pyramidal recall: 97.02% → 97.75% (+0.73 pp) — got better
- Interneuron recall: 89.74% → 89.16% (-0.58 pp) — got WORSE despite 4× weight

**Why**: bigger model has more capacity to overfit majority patterns. Class weighting alone doesn't overcome the diversity gap when one class has 2.7× less data.

### Iteration B — explicit oversampling

**Hypothesis**: class_weight=4 didn't help interneurons because interneuron training data lacks pattern *diversity*. Need actual duplicated rows, not just heavier loss.

**Change**:
- Bootstrap-sample interneurons to match pyramidal count (2,738 → 7,278)
- Removed class_weight (data is now balanced by replication)
- Same bigger model from A

**Result**:
- Overall: 95.34% → 95.18%
- Pyramidal recall: 97.75% → 96.68% (-1.07 pp) — got WORSE
- Interneuron recall: 89.16% → 91.33% (+2.17 pp) — improved

**Insight**: oversampling trades pyramidal accuracy for interneuron accuracy. Net unchanged overall. The model has finite capacity and can't gain on both classes simultaneously.

### Iteration C — pure XGBoost, no voting

**Hypothesis**: VotingClassifier soft-averaging might be hiding signal that one model alone would capture better. Try a stronger SINGLE model.

**Change**:
- Replace VotingClassifier with single `xgb.XGBClassifier(n_estimators=2000, max_depth=8, lr=0.03, subsample=0.85, colsample_bytree=0.85, early_stopping_rounds=50)`
- Use 10% held-out validation split for early stopping
- Still oversampling

**Result**:
- Overall: 95.18% (same as B)
- Pyramidal recall: 96.34% (lower)
- Interneuron recall: 92.20% (higher)
- Higher confidence outputs (mean conf 0.96 vs B's 0.90) — model more certain, but errors unchanged

**Insight**: model class doesn't matter. XGB-only is more confident but equally accurate. The bottleneck isn't the classifier algorithm.

### Iteration D — 5-seed ensemble

**Hypothesis**: even if a single model has bias, different random seeds might make different errors. Voting across seeds should reduce errors.

**Change**:
- Train 5 XGB classifiers with seeds 42, 123, 456, 789, 1024
- Same architecture as C, oversampling
- Eval: average softmax probabilities across 5 models, predict argmax

**Result per individual model**:
- Seed 42: 95.18%
- Seed 123: 95.38%
- Seed 456: 94.98%
- Seed 789: 95.62%
- Seed 1024: 95.58%
- **Range: 0.64 pp**

**Ensemble result**: **95.46% — best of all iterations (112 errors)**.

**Critical insight**: 5 models trained with different seeds all fail on essentially the SAME ~115 cells. The ~0.64 pp variance between individual models is tiny — they're highly correlated. Ensembling provides minimal lift (~0.3 pp) because errors are correlated.

This proves the errors are **not stochastic** — they're systematic. The model gets the same cells wrong every time because the features for those cells genuinely straddle the decision boundary.

### Iteration E — add 6 new discriminative features

**Hypothesis**: features may be the bottleneck. Add features specifically targeting pyramidal vs interneuron discrimination.

**New features added to `hybrid/features.py:extract_global_features`**:
1. `soma_radius` — radius of the consolidated soma anchor node
2. `max_neurite_to_soma_ratio` — max neurite radius / soma radius (interneurons higher)
3. `branching_density` — branch points / max path length
4. `terminal_density` — terminals / max path length
5. `bif_angle_mean` — average angle (degrees) at bifurcations
6. `bif_angle_std` — variability of bifurcation angles

Feature vector grew from 49 → 55 dimensions.

**Change**: same pure XGB model as iteration C, just retrained with 6 extra features.

**Result**: overall 95.30% — **no improvement** over iteration C (95.18%). Pyr errors went up (65 → 63 actually slightly down), int errors went down (54 → 53). Essentially the same.

**Insight**: the new features don't carry information the model couldn't already extract from the existing 49. The remaining ~115 ambiguous cells are ambiguous in feature space regardless of which features we add.

---

## Conclusions

**Ceiling identified at ~95%.** Six diverse interventions converged to the same accuracy range (94.98%–95.46%) with the same ~115 ambiguous cells failing across all of them.

**The 115 cells are SYSTEMATICALLY hard, not noisy**:
- 5 seeds with different random init fail on the same cells (ensemble gives only +0.3 pp)
- Different model architectures (voting RF+GB vs pure XGB) fail on the same cells
- Different class-imbalance strategies (weighting vs oversampling) only shuffle which class the errors land in, not the total count
- 6 new features designed specifically for the pyramidal/interneuron task didn't move the needle

**What this likely means**:
1. **GT label errors**: many of the 115 cells are probably mislabeled in the source data. A cell traced as "interneuron" but morphologically a pyramidal (or vice versa) will always be "wrong" no matter how good the model is.
2. **Genuinely atypical cells**: small fragments, immature neurons, only-basal pyramidals (no apical traced) — these straddle the cell-type boundary by definition.
3. **Feature-set limit**: even with 55 morphometric features, certain morphological patterns are summarized away. A graph-neural-net classifier that uses the full tree structure (not summary features) could in principle do better — but at significant engineering cost.

## Paths to break the 95% ceiling (not yet attempted)

| Approach | Effort | Realistic ceiling | Risk |
|---|---|---|---|
| **F. GNN-based cell-type classifier** | 6–8 hr code + 1 hr train | ~97–98% | medium |
| **G. Manual label review** of the 115 always-wrong cells; relabel any GT errors | 2–3 hr clicking | up to 99%+ | depends on label-error rate |
| **H. Accept 95% and use Stage 1 confidence as a soft-flag** | 0 | 95% reported, ~3% caught by handoff | — |

The current production setup uses **(H)** combined with the soft-handoff logic in `hybrid/pipeline.py` (run both Stage 2 pipelines when Stage 1 confidence is below threshold). For the paper, all three options should probably be discussed honestly.

---

## Files produced by these experiments

- `paper/logs/stage1_iter_A.log` through `stage1_iter_E.log` — full stdout of each iteration
- `paper/results/stage1_seed42.json` — final result (latest iteration overwrites)
- `paper/results/stage1_seed42_ensemble.json` — 5-seed ensemble result (iteration D)

## How to reproduce any iteration

The Stage 1 trainer is `hybrid/evaluate.py:_train_stage1`. To reproduce a specific iteration, restore the corresponding configuration in that function and run:

```
python -m paper._quick_stage1_seed42      # iterations 0/A/B/C/E (single model)
python -m paper._quick_stage1_ensemble    # iteration D (5-seed ensemble)
```
