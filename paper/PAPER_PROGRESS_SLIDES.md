# Auto-typing paper — progress and pending work

Meeting deck format: one section = one slide. Read top-to-bottom for the
full story; jump to "Pending" if you want the action items.

---

## Slide 1 — Project in one sentence

A 4-stage hybrid ML pipeline (cell-type RF → per-branch RF → subtree
GraphSAGE GNN → topology refinement) that auto-labels SWC neuron
morphologies (soma / axon / basal-dendrite / apical) at production
quality, packaged as a CLI + GUI app with a bundled model.

---

## Slide 2 — Headline result

Held-out test split: **556 cells**, hash-bucket train/test (seed=42).

| metric | v9 final | best external |
|---|---|---|
| neurite-macro-F1 (Stage 2+3) | **0.9748** | 0.9645 (lmeasure_rf) |
| per-file mean F1 | **0.9638** | 0.9379 |
| per-file P10 (worst-10% cells) | **0.9618** | 0.6522 |
| accuracy | **0.9946** | 0.9913 |
| apical F1 | **0.9548** | 0.9433 |

v9 significantly beats every external baseline (paired Wilcoxon,
Bonferroni-corrected over 66 pairs). All four external baselines are
re-implementations on the SAME train/test split (NeuroM-RF, Sholl-RF,
Sholl-MLP, L-Measure-RF).

---

## Slide 3 — What's done: pipeline architecture

Iterative architecture story with clean intermediate snapshots:

| version | what it adds | neurite-F1 |
|---|---|---|
| heuristic | rule-based topology only | 0.4580 |
| v6 | + per-branch RF (Stage 2) | 0.9682 |
| v7 | + GNN apical/basal head (Stage 2b) | 0.9745 |
| v8 | + subtree-aware Stage 2 | 0.9745 |
| v9_no_gnn | drop GNN ablation | 0.9719 |
| **v9 final** | **all stages** | **0.9748** |

Each version has its own snapshot JSON + per-file CSV in
`paper/results/snapshots/`.

---

## Slide 4 — What's done: comparisons

**Floor baselines (3):** random, majority, heuristic — establish lower
bound. neurite-F1 in 0.31-0.46 range.

**External baselines (4):** all re-implementations of published methods,
trained + evaluated on the same 556-cell test split for an apples-to-
apples comparison.
- NeuroM-RF (Emissah/Ascoli-style features)
- Sholl-RF
- Sholl-MLP
- L-Measure-RF

**Significance tests:** paired Wilcoxon signed-rank on per-file F1,
Bonferroni-corrected over 66 pairs. v9 vs every external baseline is
significant after correction.

**Bootstrap CIs:** 95% CIs on per-file F1 for every method
(`paper/results/bootstrap_ci.json`).

---

## Slide 5 — What's done: ablations (5 + 1 in progress)

Same held-out test split (seed=42) for all five.

| ablation | mean Δ vs v9 | P10 Δ | wins/losses/ties | Bonferroni p | take-away |
|---|---|---|---|---|---|
| no_pca (drop 12 PCA features) | +0.0000 | −0.0167 | 0/0/556 | 1.0 | aggregate tied, P10 hurt — keep |
| no_trunk (drop 4 trunk feats) | +0.0017 | +0.0037 | 7/5/544 | 1.0 | tiny clean gain — remove |
| no_soft_handoff | +0.0089 | +0.0377 | 47/109/400 | 1.0 | fewer catastrophic fails — remove |
| multi_seed_123 | −0.0134 | −0.30 | (different split) | 1.0 | stability ±0.5pt |
| multi_seed_456 | −0.0118 | −0.31 | (different split) | 1.0 | stability ±0.5pt |
| **no_trunk + no_soft_handoff (RUNNING)** | TBD | TBD | TBD | TBD | confirms the gains stack |

No internal ablation reaches significance after Bonferroni correction
over 66 pairs. The pipeline is robust — the GNN + Stage 3 absorb most
feature variations.

Decisions captured in `paper/PRUNED_FEATURES_LOG.md`.

---

## Slide 6 — What's done: robustness checks (NOT ablations)

These belong in the "generalization" section, separate from the
ablation table.

**Multi-seed stability** (Stage 1+2 retrain at seeds 42/123/456, GNN
held fixed):
- neurite-F1 = **0.9703 ± 0.0048** (mean ± std across 3 seeds)
- per-file P10 varies more (0.65-0.96) because each seed produces a
  different hash-bucketed test split with different difficulty mix.

**Cross-dataset zero-shot** (hpf_ca1 corpus, **1,377 cells** — completely
held-out, never seen during training):
- neurite-F1 = **0.9499** (Δ −2.5pt vs in-domain 0.9748)
- accuracy = 0.9923, P10 = 0.7632
- Stage 1 cell-type detection still 99.85% accurate
- Standard zero-shot drop, no catastrophic failure mode

---

## Slide 7 — What's done: novel sub-analyses

Beyond the standard table, we have three deeper analyses:

**1. No-apical-pyramidal census** (`paper/no_apical_pyramidal_census.py`)
- 36 pyramidal cells with NO apical ground-truth (atypical / immature)
- v9 predicted **zero false apicals** on all 36 → 100% specificity for
  the apical class on truly-non-apical cells
- Strong evidence the model isn't just over-predicting "apical"

**2. Apical-recall sub-analysis**
(`paper/apical_recall_subanalysis.py`)
- Per-source × cell-type breakdown of where apical detection is hardest
- Identifies the long tail (~10% of files dominate the failure budget)
- Worst-source: pyramidal/neuromorpho (mean F1=0.879, P10=0.589)

**3. Computational cost analysis** — partially done
- Per-cell timing recorded by `paper.eval_engine_on_test`
- Need: aggregate to median ms/cell + breakdown by stage

---

## Slide 8 — Current paper table

`paper/results/paper_table.txt` — 16 rows (3 floor + 4 external + 5
pipeline versions + 5 ablations) with consistent columns:

```
method                acc      macro-F1   neurite-F1   pf_mean   pf_P10   p_bonf
v9_final              0.9946   0.9811     0.9748       0.9638    0.9618   (ref)
no_trunk              0.9948   0.9819     0.9758       0.9655    0.9655   1.00
no_pca                0.9944   0.9805     0.9739       0.9639    0.9451   1.00
no_soft_handoff       0.9945   0.9803     0.9738       0.9727    0.9995   1.00
sholl_rf              0.9907   0.9727     0.9636       0.9282    0.6232   8.79e-06 *
neurom_rf             0.9722   0.9175     0.8900       0.8821    0.6912   4.87e-35 *
heuristic             0.5761   0.5935     0.4580       0.5250    0.1331       -
```

(Truncated — full version in `paper/results/paper_table.txt`.)

---

## Slide 9 — What's pending: decisions

**Headline model decision** (do BEFORE writing methods):
1. Should v9 final be re-defined as "v9 minus trunk features minus
   soft-handoff"? Both ablations show net improvements.
2. If yes → retrain Stage 1+2 + GNN with trunk features physically
   removed from `branch_features.py`. ~5 hr compute.
3. If no → stick with current v9 final, mention the pruned-but-tested
   features in the methods section as design choices.

**Currently running** (started 10:27 today, ETA ~15:00):
- `no_trunk + no_soft_handoff` combo ablation. Confirms whether the
  individual gains stack when both are removed together.

---

## Slide 10 — What's pending: writing

Sections to draft, in priority order:

| section | status | notes |
|---|---|---|
| Methods — pipeline | not started | Architecture diagram + per-stage description |
| Methods — features | not started | Feature catalog (110+ features). Reference `PRUNED_FEATURES_LOG.md` for the one-liner about pruned features |
| Results — main table | data ready | `paper_table.txt` is paper-ready format; needs LaTeX conversion |
| Results — significance | data ready | Wilcoxon table at `snapshots/significance_tests.txt` |
| Results — robustness | data ready | Multi-seed + cross-dataset (Slide 6) |
| Discussion — failure modes | sub-analysis ready | Worst-decile breakdown, apical sub-analysis |
| Figures | not started | Need: pipeline schematic, per-class F1 bar chart, P10 distribution, cross-dataset comparison |

---

## Slide 11 — What's pending: optional experiments

Not blockers, but worth considering:

- **Second cross-dataset corpus** (e.g. Allen Cell Types beyond what's
  in v9_merged_dataset training). Strengthens generalization claim.
- **Per-source cell-type bias analysis** — does the model over-rely on
  source-specific noise? (e.g. NeuroMorpho files have characteristic
  resampling artifacts.)
- **Ablation of feature groups** rather than individual feature sets:
  group-level "drop all topology features" vs "drop all geometric
  features". More interpretable for readers.
- **Speed comparison vs external baselines** — wall-clock per cell.

---

## Slide 12 — What's pending: tooling

- Plumb `eval_engine_on_test` timing into a clean cost-analysis
  table (median ms/cell, P95, breakdown by stage).
- Generate the figure list — pipeline schematic still hand-drawn somewhere?
- LaTeX conversion of `paper_table.txt` (could just `pandoc` it).

---

## Slide 13 — Bugs caught and fixed during this work

(Useful for the meeting — shows engineering rigor.)

1. **GNN feature-dim mismatch under env-var ablations**. Cached GNN
   trained with 61 features rejected eval extracting 56. Fix: per-stage
   GNN retrain.
2. **Δ unicode crash** on Windows cp1252 stdout in `hybrid/evaluate.py`.
   Crashed AFTER metrics computed but BEFORE JSON write. Fix:
   replace Δ with "delta", reconfigure stdout to UTF-8.
3. **eval_tmp cache silently reused across env vars**. The previous
   overnight queue's no_trunk/multi_seed runs all reused a no_pca-trained
   Stage 2 model — produced superficially-valid metrics that were
   nonsense. Fix: fingerprint sidecar (`fingerprint.json`) covering
   seed + env vars; cache invalidates when fingerprint changes.
4. **Snapshot-on-success-only missed crashed-but-valid results**.
   Combined with bug #2, no_pca's metrics were never written to disk
   despite being correctly computed. Fix: mtime-based snapshot detection.

---

## Slide 14 — Next 3-step plan

1. **Today**: combo ablation finishes (~15:00). Compare to individual
   no_trunk and no_soft_handoff to confirm gains stack.
2. **This week**: decide headline model (current v9 vs trimmed v9).
   If trimming, kick off final retrain.
3. **Next week**: start writing Methods + Results sections using
   existing data. Figures parallel to writing.

---

## Where everything lives

```
paper/
├── PRUNED_FEATURES_LOG.md          # paper-writer's lookup for pruned features
├── PAPER_PROGRESS_SLIDES.md        # this file
├── compile_paper_table.py          # rebuilds the unified table
├── significance_tests.py           # paired Wilcoxon (66 pairs)
├── apical_recall_subanalysis.py    # tail breakdown
├── no_apical_pyramidal_census.py   # 100% specificity check
├── overnight_queue.py              # ablation orchestrator
└── results/
    ├── paper_table.txt             # the unified table
    ├── paper_table_full.json       # machine-readable version
    ├── overnight_queue.log         # full compute log
    ├── cross_dataset_v9_hpf_ca1.json # 1377-cell zero-shot eval
    └── snapshots/
        ├── eval_no_pca.json
        ├── eval_no_trunk.json
        ├── eval_no_soft_handoff.json
        ├── eval_multi_seed_{123,456}.json
        ├── significance_tests.{json,txt}
        ├── v9_final_subtree_gnn.json     # current headline
        └── apical_recall_subanalysis.{json,txt}
```
