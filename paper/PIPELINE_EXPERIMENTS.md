# Pipeline experiments — possible improvements catalog + status log

## Status summary

| Phase | When | Best result | Best experiment | Δ vs prev best |
|---|---|---:|---|---:|
| 1 | 2026-05-24 | mean P10 = **0.5768** | `gnn_focal_loss_gamma_2` | +1.26 pp |
| 2 | 2026-05-25 | mean P10 = **0.5768** (unchanged) | (Phase 1 best held) | 0 |
| Cleanup pass | 2026-05-26 | (data-side) | T2 + bad-lab drop | small positive |
| L-Measure A/B | 2026-05-26 | (negative result) | L-Measure features hurt | −3.6 pp P10 |
| Branch3 rescue | 2026-05-27 | P10 = **0.6214** | 3-class pyramidal branch rescue GNN | +3.06 pp vs clean baseline |

Baseline mean P10 across seeds {42, 789}: **0.5642**.
Target: ≥ 0.70.
**Current gap: 0.123 — unchanged after Phase 2.**

### Phase 1 experiments completed (results table)

| # | Experiment | Tier | Retrain | mean P10 | Δ baseline | Status |
|---|---|---|---|---:|---:|---|
| 0 | baseline | — | — | 0.5642 | — | reference |
| 1 | aggressive_soft_handoff_thr_0.99 | 1 | no | 0.3484 | **−21.58 pp** | ✗ HURT (reverted) |
| 3 | branch_neighbor_smoothing | 1 | no | 0.5642 | 0.0000 | ✗ no effect |
| 4 | aggressive_island_flipping_size_30 | 1 | no | 0.5642 | 0.0000 | ✗ no effect |
| 6 | gnn_focal_loss_gamma_2 | 2 | GNN | **0.5768** | **+1.26 pp** | ✓ KEPT |

Detailed results: `paper/PHASE1_RESULTS.md`.

### Phase 2 experiments completed (results table)

| # | Experiment | Tier | Retrain | mean P10 | Δ vs Phase 1 best | Status |
|---|---|---|---|---:|---:|---|
| — | phase2_baseline (focal γ=2 already kept) | — | no | 0.5768 | 0.0000 | reference |
| 7 | gnn_focal_loss_gamma_3 | 2 | GNN | 0.5653 | **−1.15 pp** | ✗ HURT (reverted) |
| 11-lite | stage2_aggressive_class_balance_power_2.0 (+ focal γ=2) | 3 | Stage 2 + GNN | 0.5747 | −0.21 pp | ✗ HURT (reverted) |

Detailed results: `paper/PHASE2_RESULTS.md`.

### Key learning from Phase 2

- **Focal γ=3 over-focuses** — model loses 1.15 pp vs γ=2. The γ=2 from Phase 1 was already at the sweet spot.
- **Stage 2 power 1.25 → 2.0 hurts slightly** — the default class balancing was already well-tuned; pushing harder on apicals over-corrects.
- **Combined Phase 1 + Phase 2 (8 experiments)**: peak lift is +1.26 pp (Phase 1's focal γ=2). The marginal hyperparameter space is exhausted.

### Key learning from Phase 1

- **Aggressive soft handoff was a disaster (−22 pp)**. Picking "the more confident" cell-type pipeline often picks the WRONG-CELL-TYPE pipeline producing a confidently-wrong output. The default threshold (0.65) is actually well-tuned. Lesson: confidence ≠ correctness when models are mis-routed.
- **Branch smoothing & island flipping gave 0 lift**. The existing Stage 3 refinement already catches most isolated mislabels; the marginal cases targeted by the new rules don't exist (or the rules don't fire as designed).
- **Focal loss GNN works** — modest but consistent +1.26 pp. Targeting hard apical-vs-basal cases in the GNN loss helps the tail of cells.
- **Big lesson**: small inference-time tweaks didn't help. The bottleneck is in TRAINING (Stage 2 / GNN learning hard cases) or in DATA (label errors). Structural changes likely needed.

---

### 2026-05-26 session experiments (data-side + L-Measure A/B)

| # | Experiment | Type | Result | Status |
|---|---|---|---:|---|
| T2 | Cross-seed Stage 1 disagreement scan → drop 89 cells | data | small positive | ✓ KEPT (cells removed) |
| BadLab | Lab-prefix failure-rate scan → drop 533 cells from 21 high-failure labs | data | small positive | ✓ KEPT (cells removed) |
| L-Measure-RF | Add 4 L-Measure features (partition_asymmetry mean/max, branch_contraction, subtree_volume_norm) to Stage 2 + GNN | model | **−3.6 pp on apples-to-apples** (apical F1 −1.6 pp, basal F1 −0.25 pp) | ✗ HURT (reverted) |

Detailed analysis (T3 + T4): `paper/results/heldout_per_cell_f1.csv`, `paper/results/gt_failure_modes.md`.

**Key learning from this session**:

- **GT cleanup (cell-type mislabel + bad-lab) gives small but real lift**. Combined ~622 cells dropped from 12,484 → 11,862. Pyramidal pool: 9,057 → 8,488 (505 dropped). Headline numbers move ~+0.3 pp on per-node accuracy, slightly more on per-file mean F1; P10 doesn't move much because the dropped cells aren't all bottom-decile.
- **L-Measure features added globally hurt overall (−3.6 pp P10)**. Despite L-Measure-RF baseline beating v12 on standalone apical F1 (0.91 vs 0.88), adding L-Measure features to v12 didn't transfer — the existing features already encode most of the signal, and the new features added redundancy + noise. 177 cells improved by ≥0.05, but 209 cells got worse by ≥0.05 → net negative.
- **Apical recognition is the central problem** (per `paper/results/gt_failure_modes.md` — T4 analysis): ~540 cells in bottom 10% have APICAL_MISSED or APICAL_UNDERPREDICTED patterns. The model's apical-vs-basal decision boundary is poorly learned in both directions. Adding raw morphometry features didn't fix it; the issue may need architectural changes or different features (TMD / persistence diagrams).
- **APICAL_INVERTED is NOT a meaningful failure signal** (enrichment only 1.7x). 897 cells have apical-below-soma but most are fine — different coordinate conventions across labs. Skipped as a cleanup heuristic.
- **The "neuromorpho is bad" intuition was too coarse**: only 1.06x failure enrichment vs allen/hpf_ca1 overall. Within neuromorpho, ~21 specific labs concentrate the failures (533 cells total). Most of neuromorpho (~10k cells) is well-curated.

---

### 2026-05-27 session experiment: Branch3 rescue head

| Experiment | Type | Result | Status |
|---|---|---:|---|
| Branch3 conservative gate | pyramidal-only 3-class GraphSAGE correction head | F1 P10 0.5908 -> **0.6214**, apical F1 0.9008 -> **0.9081** | **KEPT / adopted** |
| Branch3 loose gate | threshold sweep + full eval | F1 P10 0.5908 -> **0.6377**, apical F1 0.9008 -> **0.9075** | optional; not default due more regressions on already-good cells |
| Branch3 learned gate | HistGradientBoosting accept/abstain gate over Branch3 proposals | F1 P10 0.5908 -> **0.6463**, apical F1 0.9008 -> **0.9117** | opt-in only; not default due 60 already-good cells falling below F1 0.8 |

Key learning: the main apical issue was not just basal-vs-apical inside known dendrite branches. A sizeable chunk of GT apical was being locked in as axon. The 3-class rescue head can correct axon/basal/apical directly while leaving the existing Stage 1, Stage 2, old GNN, and all interneuron artifacts frozen.

Learned-gate follow-up: aggregate metrics improved further, but the gate was poorly calibrated for the "do not damage good cells" objective. Candidate accept rate was 0.986 on gate-train, 0.963 on gate-val, but only 0.598 on the held-out test cache, and its scores were saturated near 1.0. Full eval had 250 cells improved by >=0.05 F1, but 125 cells worsened by <=-0.05; 60 cells with baseline F1 >= 0.8 dropped below 0.8. Keep this artifact for analysis/ablation, not production default.

Artifacts:
- `paper/models/v12_pyramidal_only_seed2024_clean/gnn_branch3_rescue.pt`
- `paper/models/v12_pyramidal_only_seed2024_clean/branch3_gate.joblib`
- `paper/results/v12_pyramidal_only_seed2024_clean_branch3_eval.{json,csv,txt}`
- `paper/results/v12_pyramidal_only_seed2024_clean_branch3_loose_eval.{json,csv,txt}`
- `paper/results/v12_pyramidal_only_seed2024_clean_branch3_gate_eval.{json,csv,txt}`

---


**Goal**: improve v12 per-cell F1 P10 (currently 0.5905 on full pipeline,
0.6237 with GT cell-type override) toward 1.0. Each experiment is
documented with its target, expected lift, cost, and risk.

**Test set**: seeds 42 and 789, each on its own held-out 20% split
(~2,500 cells each, contamination-free). Mean P10 across both seeds is
the headline metric.

**Baseline reference numbers (v12 GT cell-type, seed=42)**:
- Per-cell F1 mean: 0.9261
- Per-cell F1 **P10: 0.6237**
- Pyramidal P10: 0.5803
- Interneuron P10: 0.999
- Corpus apical F1: 0.9137

---

## Tier 1: Inference-only changes (no retraining)

Cheapest to test (~30 min per experiment, just inference). The trained
models on disk are unchanged.

### 1. Aggressive soft handoff (threshold = 0.99)

**Mechanism**: env var `SWCAL_SOFT_HANDOFF_THRESHOLD=0.99` makes the
pipeline run BOTH cell-type Stage 2 pipelines for nearly every cell
(default threshold is 0.65), then pick whichever produces higher mean
per-node confidence. Targets the Stage 1 cascade failures.

**Code**: already wired in `hybrid/pipeline.py:DEFAULT_SOFT_HANDOFF_THRESHOLD`.

**Expected**: +3-5 pp on P10. **Risk**: low.

### 2. Probability-weighted dual-pipeline averaging

**Mechanism**: instead of picking ONE pipeline at handoff, run both and
**average per-node predictions** weighted by Stage 1 softmax probabilities.
Smoother handling of borderline cells; no hard winner-takes-all.

**Code**: NEW — needs modification of `_run_stage23`'s soft-handoff
branch in `hybrid/pipeline.py`.

**Expected**: +2-4 pp. **Risk**: low.

### 3. Branch-level neighbor smoothing in Stage 4

**Mechanism**: post-process Stage 3 output. For each branch, look at
its parent and immediate child branches. If both neighbors agree on a
class but THIS branch predicts something else, flip this branch.
Catches isolated mislabeled branches.

**Code**: NEW — function added to `hybrid/stage3_refine.py`, env-gated
by `SWCAL_BRANCH_NEIGHBOR_SMOOTH=1`.

**Expected**: +1-3 pp. **Risk**: very low.

### 4. Aggressive island flipping (max_island_size 8 → 30)

**Mechanism**: existing `_island_flipping` in `stage3_refine.py` looks
for isolated small mislabeled segments (≤ 8 nodes) and flips them.
Bumping the size to 30 catches larger isolated mislabels.

**Code**: parameterize the existing constant via env var
`SWCAL_MAX_ISLAND_SIZE`.

**Expected**: +0.5-2 pp. **Risk**: very low (might over-smooth on
truly multi-class branches).

### 5. Confidence-based fallback

**Mechanism**: if all per-node confidences in a branch are below
a threshold (say 0.55), default the whole branch to basal (the dominant
neurite class). Reduces catastrophic class failures where the model
predicts the wrong class with high commitment.

**Code**: NEW — post-process in pipeline.py.

**Expected**: +0.5-2 pp. **Risk**: low.

---

## Tier 2: Medium-cost retraining (~50-100 min per experiment)

### 6. Focal loss GNN (γ=2)

**Mechanism**: replace cross-entropy loss in `paper/gnn_apical_basal.py`
with focal loss `(1 - p_correct)^γ × CE`. Down-weights easy examples,
concentrates gradient on hard apical-vs-basal cases.

**Code**: already wired via `SWCAL_GNN_FOCAL_GAMMA=2.0` env var.

**Expected**: +1-3 pp. **Risk**: low.

### 7. Focal loss GNN (γ=3)

Same as #6 but more aggressive tail focus. Risk: may over-focus on
hard examples and lose performance on easy ones.

### 8. GraphSAGE → GAT (graph attention)

**Mechanism**: replace the GraphSAGE conv layer in
`paper/gnn_apical_basal.py:ApicalBasalSAGE` with a GAT layer
(`torch_geometric.nn.GATConv`). The model learns *which* neighbors
matter most for the apical decision — useful when apicals branch off
in characteristic patterns.

**Code**: NEW — small change to model definition.

**Expected**: +2-4 pp. **Risk**: medium (different convergence
behavior; may need new learning rate tuning).

### 9. GNN curriculum learning

**Mechanism**: train GNN initially on all training cells, then fine-tune
on cells the first model misclassified (with higher weight). Two-pass
training.

**Code**: NEW — wraps the existing GNN trainer.

**Expected**: +1-3 pp. **Risk**: medium.

### 10. Stage 1 ensemble (5-seed vote)

**Mechanism**: train 5 Stage 1 models with different seeds, average
softmax. Should reduce Stage 1 error rate from 5% to 4.5% → fewer
cascades.

**Code**: scaffolding exists in `_quick_stage1_ensemble.py`. Need to
update `detect_cell_type_from_nodes` to load multiple models.

**Expected**: +1-2 pp on full-pipeline P10 (Stage 1 alone plateaued
at 95.46% in prior experiments). **Risk**: low.

---

## Tier 3: Heavy retraining (~3-5 hours per experiment)

### 11. Focal loss Stage 2

**Mechanism**: custom XGBoost objective implementing focal loss for
the per-branch RF classifier. Retrain Stage 2 + GNN (GNN depends on
Stage 2's subtree-owner features).

**Code**: NEW — XGBoost custom objective in
`hybrid/train_stage2.py`, env-gated by `SWCAL_STAGE2_FOCAL_GAMMA`.

**Expected**: +1-3 pp. **Risk**: low.

### 12. Cell-type-as-input-feature Stage 2

**Mechanism**: instead of two separate per-cell-type Stage 2 models
(`models_by_cell_type` dict), train ONE model that takes cell-type
as a binary input feature. When Stage 1 misclassifies, the wrong
cell-type input just nudges the output (rather than swapping to an
entirely different model). Eliminates the cascade catastrophe.

**Code**: SIGNIFICANT — rearchitect `hybrid/train_stage2.py` and the
Stage 2 inference path in `hybrid/pipeline.py`.

**Expected**: +3-7 pp (structural fix for Stage 1 cascade).
**Risk**: medium-high (lots of code change; loses the structural
separation that made each model specialized).

### 13. Hard-cell 2-pass training

**Mechanism**: train Stage 2 + GNN on full corpus, then evaluate on
training data, identify worst-decile cells, retrain with those weighted
5x. Direct attack on the failure tail.

**Code**: NEW — wraps the training scripts.

**Expected**: +2-4 pp. **Risk**: low.

### 14. Per-source class weighting

**Mechanism**: the corpus is ~87% NeuroMorpho. Upweight Allen
(~1.7% of cells) and hpf_ca1 (~11%) during training. May help
generalization to lab-style data.

**Code**: small change to sample weights in the trainers.

**Expected**: +0.5-2 pp. **Risk**: low (may help cross-dataset but
hurt aggregate).

---

## Tier 4: Architecture changes (~1-2 days each)

### 15. GNN-based Stage 1

**Mechanism**: replace summary-feature RF Stage 1 with a graph neural
net that takes the full tree as input. Stage 1's 95% ceiling on
summary features was confirmed in 6 iterations (see
`STAGE1_EXPERIMENTS.md`); a GNN might break that.

**Code**: SIGNIFICANT — new model architecture; needs feature
extraction at the graph level instead of cell level.

**Expected**: Stage 1 accuracy 95% → 97-98% → fewer cascades →
+2-4 pp on P10. **Risk**: high (lots of code; uncertain payoff).

### 16. End-to-end pipeline trained jointly

**Mechanism**: one model from raw SWC → per-node labels. Gradients
flow through all stages. Eliminates cascade failures by construction.

**Code**: months of work — too speculative for now.

**Expected**: +2-5 pp. **Risk**: very high.

---

## Tier 5: Data-side (no model change)

### 17. Manual GT label cleanup

**Mechanism**: inspect the ~115 cells Stage 1 always misclassifies
(documented in `STAGE1_EXPERIMENTS.md`). 30-70% are likely GT label
errors (cell traced as "interneuron" but morphologically pyramidal,
or vice versa). Fix the labels, retrain.

**Code**: human work, no scripting needed beyond surfacing the cells.

**Expected**: up to +4 pp on Stage 1 → ~+3 pp on P10.
**Risk**: requires hours of expert human time.

### 18. Confidence-based selective rejection

**Mechanism**: don't label cells where overall pipeline confidence is
below a threshold. Mark them "human review needed" and exclude from
the metric.

**Code**: small change to inference + metric code.

**Expected**: P10 jumps to 0.85+ on accepted cells (the worst-decile
cells become "rejected" instead of counted as failures). The price:
~5% of cells excluded from the labeled output.

**Risk**: requires honest disclosure in paper that we're computing
metrics on a self-selected subset of cells.

---

## Phase 1 plan (running now)

Five experiments selected as the cheapest high-expected-value combination:

| # | Name | Tier | Retrain | Est. lift |
|---|---|---|---|---|
| 0 | baseline | — | — | reference |
| 1 | aggressive_soft_handoff_thr_0.99 | 1 | no | +3-5 pp |
| 3 | branch_neighbor_smoothing | 1 | no | +1-3 pp |
| 4 | aggressive_island_flipping (30) | 1 | no | +0.5-2 pp |
| 6 | gnn_focal_loss_gamma_2 | 2 | GNN both seeds | +1-3 pp |

Combined expected lift if additive: +5-13 pp on per-cell F1 P10.
Realistic combined (since interventions overlap): +4-8 pp.

Total wall time: ~2.5-3 hours.

Output:
- `paper/results/p10_iteration_log.json` — full timeline with kept/reverted decisions
- `paper/PHASE1_RESULTS.md` — written by `_generate_phase1_report.py` after iteration ends
