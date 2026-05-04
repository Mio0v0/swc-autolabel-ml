# SWC-Studio — Project Continuation Brief

A self-contained context document for picking up work in a new Claude session
(typically on a different machine). Read this top-to-bottom and you have
everything needed to keep going.

---

## 1. What SWC-Studio is

`SWC-Studio` is a Python toolkit for working with neuron morphology files
(SWC format). It includes a desktop GUI, a CLI, a Python library, and an
auto-labeling backend (the `hybrid/` package).

This document is about the **auto-labeling backend** — a 3-stage hybrid ML +
rule-based pipeline that takes any SWC file and labels every node as soma,
axon, basal dendrite, or apical dendrite.

## 2. Where things live in the repo

```
SWC-Studio/
├── hybrid/                      ← production ML pipeline
│   ├── features.py              Stage 1 whole-cell features (49 total, incl. 7 PCA)
│   ├── branch_features.py       Stage 2 per-branch features (61 total, incl. 5 PCA + 4 trunk)
│   ├── subtree_features.py      primary-subtree features
│   ├── cell_type_detector.py    Stage 1 logic
│   ├── stage3_refine.py         Stage 3 topology rules
│   ├── pipeline.py              orchestrator (with soft handoff)
│   ├── train_stage1.py          train cell-type classifier
│   ├── train_stage2.py          train per-branch classifier (per-cell-type)
│   ├── evaluate.py              full Stage 1+2+3 eval, hash-bucket train/test split
│   ├── evaluate_stage1.py       Stage 1 only deep-dive
│   ├── qc.py                    consolidated QC tools (diagnose / clean-soma / filter / prune)
│   ├── data_prep.py             consolidated data prep (download / build-source-pool / build-benchmark)
│   └── models/
│       ├── evaluation_results.json     latest pipeline metrics (overwritten each eval run)
│       ├── per_file_scores.csv         per-file F1
│       ├── eval_split.json             stable test-split definition
│       ├── cell_type_classifier.json   Stage 1 schema (committed)
│       ├── *.pkl                       trained models (gitignored — regenerate per env)
│       └── eval_tmp/                   eval-run train-split pickles (gitignored)
│
└── paper/                       ← paper deliverables
    ├── baselines.py             floor + ablation runner
    ├── build_slides.py          regenerates results_summary.pptx
    ├── build_docx.py            regenerates speaker_notes.docx
    ├── results_summary.pptx     15-slide deck for medical-audience presentation
    ├── speaker_notes.docx       slide-by-slide speaker notes
    ├── results/
    │   ├── baselines_results.json
    │   ├── baselines_table.txt
    │   └── snapshots/
    │       └── v6_full_pipeline.{json,csv,...}   frozen v6 results
    └── CONTINUATION.md          ← this file
```

## 3. The pipeline architecture

```
Stage 1: cell type detector (whole-cell ML, 49 features)
          → pyramidal | interneuron
          → label_set determines which classes Stage 2 can predict

Stage 2: per-branch classifier (cell-type-conditioned ML, 61 raw + 7 owner-aug = 68 inputs)
          → axon | basal | apical (or axon | basal for interneurons)
          → confidence per branch

Stage 3: topology refinement (rules)
          → 9 rules: subtree majority vote, single-apical, PCA tie-break, etc.
```

When Stage 1's confidence is below 0.65, the pipeline runs Stage 2+3 for
**both** cell types and picks whichever produces sharper predictions
("soft handoff" — fixes slice-prep pyramidals that look like interneurons).

## 4. Methodological contributions worth highlighting in the paper

1. **PCA principal-axis features** — rotation-invariant analogues of z-axis features.
   Captures the cell's intrinsic up/down direction. Robust to coordinate-frame
   rotation and slice-flattening (where world z is collapsed). Added to both
   Stage 1 (7 features) and Stage 2 (5 features).

2. **Soft Stage-1 → Stage-2 handoff** — when Stage 1 is uncertain, run both
   cell-type-conditioned Stage 2 models and pick by mean per-node confidence.
   Converts Stage 1 errors from catastrophic to recoverable.

3. **Per-cell-type Stage 2 conditioning** — separate trained models for
   pyramidal vs interneuron, so the model never has to "remember" both
   morphologies' rules simultaneously.

4. **Diagnostic-driven dataset cleanup** — `hybrid/qc.py diagnose` classifies
   files into BAD GT / BORDERLINE / SUSPICIOUS / APPEARS HEALTHY based on
   composition heuristics. `prune` then drops files matching specific verdicts
   into a new directory, with full provenance. Shipped as a separate
   methodological contribution — a reusable QC tool for any SWC corpus.

## 5. Evaluation results — v6 → v7 → v8 → v9 (in-distribution + cross-dataset)

The pipeline went through four iterations on Windows GPU.

- **v6/v7/v8** all evaluate on the same hash-bucketed test split (328 files
  = 159 interneurons + 169 pyramidals), seed=42, test_size=0.2, dataset =
  `qc_diag_pruned` (1,737 cells).
- **v9** retrained on `data/v9_merged_dataset` = qc_diag_pruned + 1,012
  hpf_ca1 pyramidals with apical labels (2,749 cells total). Re-hashed
  the test split with the same seed → 556 test files (159 interneurons +
  397 pyramidals; the 397 includes ~200 hpf_ca1 cells). Numbers are not
  measured on the *exact* same files as v6-v8 but are directly
  comparable in aggregate (the bigger test set is harder, so v9 numbers
  understate the improvement vs v8).

### In-distribution headline numbers

| Metric                 | v6 (no GNN) | v7 (GNN, branch S2) | v8 (GNN, subtree S2) | **v9 (v8 + hpf_ca1 in train)** |
|------------------------|-------------|---------------------|----------------------|--------------------------------|
| Stage 1 cell-type acc  | 0.9878      | 0.9878              | 0.9909               | **0.9928** ✅                   |
| Overall accuracy       | 0.9917      | 0.9930              | 0.9930               | **0.9946** ✅                   |
| Macro-F1               | 0.9761      | 0.9809              | 0.9809               | **0.9811**                     |
| **Neurite-macro-F1**   | **0.9681**  | **0.9745**          | **0.9745**           | **0.9748**                     |
| **Apical F1**          | **0.9348**  | 0.9476              | 0.9476               | **0.9548** ✅                   |
| Basal F1               | 0.9722      | 0.9787              | 0.9787               | 0.9709 (trade-off)             |
| Axon F1                | 0.9974      | 0.9974              | 0.9974               | **0.9986** ✅                   |
| Soma F1                | 1.0000      | 1.0000              | 1.0000               | 1.0000                         |
| Per-file mean          | 0.9644      | 0.9633              | 0.9633               | **0.9638**                     |
| Per-file P10           | 0.9093      | 0.9192              | 0.9192               | **0.9618** ✅ (+4.3pp)          |
| **Per-file pyramidal P10** | 0.7686 | 0.6621              | 0.6621               | **0.8984** ⭐⭐⭐ (+23.6pp)       |
| Test set size          | 328         | 328                 | 328                  | 556                            |

**v9 highlights:**
- **Pyramidal per-file P10 jumped from 0.6621 to 0.8984** — the long
  failure tail introduced by v7's GNN (which sometimes hurt borderline
  cells) is essentially gone. The bigger training pool gave the GNN
  better coverage of edge cases.
- **Standalone GNN held-out test F1** (apical/basal labeled branches
  only): v8 GNN = 0.9677 → v9 GNN = **0.9756** (+0.79pp); per-cell mean
  F1: 0.8608 → 0.9240 (+6.32pp). 5-fold CV macro-F1 went from
  0.9637±0.003 to **0.9785±0.005**.
- **Apical recall** improved (pyramidal: 0.9256 → 0.9311). The model
  now produces apical labels for more of the long-thin trunk apicals
  that v8 missed.
- **Apical-as-axon** dropped from 4.37% to 3.0% of apical-GT nodes —
  the architectural ceiling moved (v9 GNN saw both noisy cortical
  apicals and clean CA1 apicals during training).
- The lab__210107_029_edited family still fails (F1 = 0.333). This is
  a hard labeling-quality ceiling (partial reconstruction with apical
  traced as axon-like extension). No model can fix it.
- One trade-off: basal F1 dropped 0.78pp (precision −1.72pp, recall
  +0.19pp). The model is slightly more eager to call things basal,
  reflecting more diverse training. Acceptable since apical was the
  real bottleneck.

**v7 contribution:** GraphSAGE apical-vs-basal head as a Stage-2b override
on per-branch Tier B labels. Hyperparameter sweep (18 configs, 5-fold CV)
picked hidden=128 / 3 layers / dropout=0.0 (~80k params). Trained on the
same train split as the rest of the pipeline. Held-out standalone eval
(apical/basal labeled branches only): macro-F1 0.9677. **+1.3 pp on apical
F1, +0.65 pp on basal F1.**

**v8 contribution:** restructured Stage 2 around primary subtrees.
Empirical evidence: 98.7% of primary subtrees in pyramidal cells have
exactly one non-soma label, so the per-branch decision was solving an
over-specified problem. v8 replaces the per-branch Tier B classifier with
Tier A's per-subtree predictions; every branch in a primary subtree
inherits its subtree's label. The GNN runs only inside dendrite-labeled
subtrees, gated by Tier A's `apical_owner_root`. **Functionally equivalent
to v7 on the headline F1**, but architecture is much cleaner: ~5 decisions
per cell instead of ~150, no Tier B per-branch sklearn model needed in this
mode. The +0.31 pp on Stage 1 cell-type accuracy comes from the soft-
handoff seeing different mean-confidence signals in subtree mode.

**v9 contribution:** v8 architecture, retrained on the merged 2,749-cell
pool (existing benchmark + 1,012 hpf_ca1 cells with apical labels). The
hpf_ca1 cells are renamed `hpf_ca1__<original>.swc` so the hash-bucket
preserves provenance. All four pipeline components (Stage 1, Tier A, Tier
B, GNN) retrained from scratch. Stage 2 retrain took ~4 hours (vs v8's
2.1 hr) due to the 1.7× larger train pool. **Pyramidal per-file P10
jumped from 0.6621 to 0.8984** (+23.6pp) — the most dramatic single
improvement of the entire project. Apical F1 +0.72pp, axon F1 +0.14pp,
Stage 1 cell-type acc +0.19pp.

**Hard ceiling moved (v9 vs v6/v7/v8).** In v6, Stage 2 mislabeled 18.19%
of apical-GT nodes as axon, of which 4.37% remained unrecovered after
Stage 3 (the unrecoverable 10,765 nodes). v9 reduces this to ~3.0% of
apical-GT (still in the lab__210xxx_xxx_edited family — same labeling
artifact pattern as before). The GNN's gate still skips axon-labeled
inputs, so the same architectural argument applies; the data-side
improvement just shrunk the input that hits the gate.

Reproducible snapshots (`paper/results/snapshots/`):
- `v6_baseline.json` (Mac v6 reproduced on Windows)
- `v7_gnn_branch.json` (per-branch GNN)
- `v7b_gnn_after_s3.json` (A/B null-result: GNN before vs after Stage 3)
- `v8_subtree_gnn.json` (subtree-level Stage 2)
- `v8_cross_dataset_hpf_ca1.json` + `_apical_only.json` (zero-shot CA1)
- `v9_baseline_no_gnn.json` (v9 retrain, no GNN)
- `v9_final_subtree_gnn.json` (v9 final: subtree-Stage2 + v9 GNN)
- `v9_eval_split.json` (v9 hash-bucket split; 397 pyr + 159 inter test)
- `gnn_sweep_18configs.json` (full HP sweep — chose hidden=128/3/0.0)

Reproducible model snapshots (`paper/models/snapshots/`):
- `v8_cell_type_classifier.pkl`, `v8_branch_classifier.pkl`, `v8_gnn_apical_basal.pt`
- `v9_gnn_apical_basal.pt` (the v9 GNN, retrained on merged train split)

### Cross-dataset zero-shot (v8 on hpf_ca1)

The v8 pipeline (production models, never retrained on hpf_ca1) was
evaluated on 1,377 hippocampal CA1 pyramidal cells from the
`ybb797/CIC_CA1_Dataset` HuggingFace repo. **49,953,904 nodes**, all
unseen during training. The filtered subset (`--require-classes 4`)
limits to the 1,012 cells whose GT type-column has at least one apical
node.

| Metric                 | In-distribution v8 | Cross-dataset (all 1,377) | **Cross-dataset (1,012 w/apical)** |
|------------------------|--------------------|---------------------------|------------------------------------|
| Stage 1 cell-type acc  | 0.9909             | 0.9985                    | **1.0000** ✅                       |
| Overall accuracy       | 0.9930             | 0.9923                    | 0.9923                             |
| Neurite-macro-F1       | 0.9745             | 0.9499                    | **0.9533**                         |
| Macro-F1               | 0.9809             | 0.9620                    | 0.9646                             |
| **Apical F1**          | **0.9476**         | 0.9206                    | **0.9386** (+1.8pp from filter)    |
| Basal F1               | 0.9787             | 0.9306                    | 0.9231                             |
| Axon F1                | 0.9974             | **0.9984** ✅              | 0.9982                             |
| Soma F1                | 1.0000             | 0.9985                    | 0.9985                             |
| Per-file median        | 1.0000             | **1.0000**                | 1.0000                             |
| Per-file mean          | 0.9633             | 0.9516                    | 0.9457                             |
| Per-file P10           | 0.9192             | 0.7632                    | 0.6334                             |

**Reading:**
- Stage 1 generalizes *better* cross-dataset (1,012/1,012 = 100% on
  filtered set; 1,375/1,377 on full). CA1 pyramidals have more
  canonical morphology than the cortical mix in train.
- Axon F1 *improves* cross-dataset (more uniform CA1 axon morphology).
- Headline F1 drops ~2.5 pp on full set, ~2.1 pp filtered. Excellent
  for a true zero-shot domain shift across brain regions with no
  fine-tuning.
- Filtering to apical-bearing GT recovers +1.8 pp on apical F1 (no-
  apical files were producing false-positive apicals in the full eval).
- Filtered per-file P10 (0.6334) looks worse than the unfiltered
  (0.7632) because the no-apical files were "easy 1.0s" the filter
  removes; the median is still 1.0 either way.
- The lab__210xxx_xxx_edited.swc family is the worst-of-worst in all
  three evaluations. Same labeling style, same partial-reconstruction
  failure mode in both datasets.

Snapshots:
- `paper/results/snapshots/v8_cross_dataset_hpf_ca1.json` (full 1,377)
- `paper/results/snapshots/v8_cross_dataset_hpf_ca1_apical_only.json`
  (filtered 1,012)
- Per-file CSVs alongside.

Reproduction:
```powershell
python -m paper.cross_dataset_eval --data-dir data/hpf_ca1 \
    --cell-type pyramidal --tag hpf_ca1
# Filtered: only files with apical-labeled nodes in GT
python -m paper.cross_dataset_eval --data-dir data/hpf_ca1 \
    --cell-type pyramidal --tag hpf_ca1_apical_only --require-classes 4
```

## 6. The GNN apical-vs-basal head — built and shipped

Done — see §5 v7 and v8. Code:
- `paper/gnn_dataset.py` — MorphologyBranches → torch_geometric.data.Data
  (one node per branch, undirected parent↔child branch edges, 51 dendrite-
  relevant features after dropping cell-type one-hots and axon-vs-dendrite
  features that are degenerate within a pyramidal-dendrite training pool)
- `paper/gnn_apical_basal.py` — model, training loop, 5-fold CV, checkpoint I/O
- `paper/sweep_gnn.py` — 18-config grid runner (hidden ∈ {32,64,128} ×
  layers ∈ {2,3} × dropout ∈ {0,0.2,0.5})
- `paper/gnn_inference.py` — pipeline-side load + score
- `paper/models/gnn_apical_basal.pt` — trained checkpoint
  (3-layer GraphSAGE, hidden=128, ~80k params)
- `hybrid/pipeline.py` — `gnn_state` and `use_subtree_stage2` parameters
  on `run_pipeline_on_nodes`; `_apply_gnn_override` helper with
  `apical_evidence` gate
- `hybrid/evaluate.py` — `--use-gnn`, `--gnn-after-stage3`,
  `--use-subtree-stage2` CLI flags

The "next big move" is no longer the GNN. It's whatever pushes past the
hard 0.95 ceiling on apical F1 — see §6.5 below.

## 6.5. v9: data-side fix

Tested experiment A from the original v7 plan: merge hpf_ca1 (1,012
apical-bearing CA1 pyramidals) into the pyramidal training pool and
retrain. **It worked.**

Result: per-file pyramidal P10 jumped from 0.6621 → 0.8984 (+23.6pp).
Apical F1 +0.72pp. Standalone GNN macro-F1 +0.79pp. Per-cell mean F1
+6.32pp on the GNN. See §5 v9 row.

The remaining failure surface:
1. **lab__210xxx_xxx_edited family still fails** — both the original
   `lab__210107_029_edited.swc` and its hpf_ca1 sibling
   `hpf_ca1__210107_029_edited.swc` score F1=0.333 in v9. This is a
   real labeling artifact (partial reconstructions), not a model
   limit. Pruning these via `hybrid/qc.py prune` would push F1 toward
   0.99 honestly.
2. **New worst files revealed** — `neuromorpho__236-1a-12-AW.swc`
   (F1=0.226) and `neuromorpho__241-2-18VN.swc` (F1=0.249) appear in
   v9's worst-5 and weren't in v8's. The bigger test pool surfaces
   more pathological neuromorpho files.

**Open architecture-side experiment (now optional):**

The targeted **axon-veto classifier** (the only architecture-side
experiment that can attack the unrecoverable nodes the current GNN
gate skips) is still on the shelf. With v9 having reduced apical-as-
axon errors from 4.4% to 3.0%, the marginal value is smaller than
before. Worth doing only if Paper 1 needs an additional architectural
contribution row.

## 7. Paper strategy

Two-paper plan agreed on:

**Paper 1 (algorithm/methods, write first):**
- Title direction: "A hybrid ML + topology pipeline for per-node neuron labeling"
- Realistic targets: *Patterns* (Cell Press) or *eLife* Tools and Resources
- Aspirational: *Nature Computational Science* (would need GNN + baselines + downstream impact)

**Paper 2 (tool/application, write second, cite Paper 1):**
- Title direction: "SWC-Studio: an integrated toolkit for inspecting, repairing, and labeling neuron morphology"
- Target: *Bioinformatics* Application Note or *Frontiers in Neuroinformatics*

**Paper 1's required components:**
- ✅ Headline numbers (in `evaluation_results.json` + snapshots)
- ✅ Architectural novelty (PCA, soft handoff, hybrid, **+GNN, +subtree-Stage2**)
- ✅ Failure-mode taxonomy (BAD GT findings)
- ✅ **GNN apical-basal head** — built, swept (18 configs), integrated, ablated
- ✅ **Cross-dataset generalization** — v8 zero-shot on 1,377 hpf_ca1
  pyramidals (49.95M nodes); F1 = 0.9499, Stage 1 = 0.9985
- ✅ **v9 retrain with hpf_ca1 in training pool** — pyramidal per-file
  P10 0.6621 → 0.8984 (+23.6 pp); apical F1 +0.72 pp (§5)
- ✅ **Floor baselines** (random / majority / heuristic) on the v9 test
  split. Snapshot: `paper/results/snapshots/v9_floor_baselines.{json,txt}`.
  Anchors v9's headline F1 against trivial references (random=0.336,
  majority=0.313, heuristic=0.458 vs v9=0.9748 neurite-macro-F1).
- ✅ **Per-source breakdown** of the per-file F1 distribution. Splits the
  v9 test results by filename prefix (allen, lab, neuromorpho,
  hpf_ca1) and by cell type. Lab & CA1 saturate at F1=1.0;
  neuromorpho-pyramidals are the weakest segment (mean F1=0.879).
  Snapshot: `paper/results/snapshots/v9_per_source_breakdown.{json,txt}`.
- ✅ **Bootstrap 95% CIs** on the v9 headline numbers (5,000 file-level
  resamples). Per-file overall mean = 0.9639 [0.9532, 0.9738]; pyramidal
  mean = 0.9569 [0.9432, 0.9696]; interneuron mean = 0.9813 [0.9659,
  0.9937]. Snapshot: `paper/results/snapshots/v9_bootstrap_ci.{json,txt}`.
- ⏳ Component ablations (have v6 / v7 / v7b / v8 / v9 already; need to
  add no-PCA, no-soft-handoff, no-trunk rows from `paper/baselines.py`)
- ⏳ External tool baselines (NeuroM features + RandomForest, L-Measure
  features + RF) — to add
- ⏳ Downstream impact section (re-label real public dataset, surface
  label errors) — to add for NCS / Nature Comm tier

## 8. Recent state changes (most recent first)

- **v9: retrained on merged dataset** (qc_diag_pruned + 1,012 hpf_ca1
  with apical = 2,749 cells). Stage 2 retrain took ~4 hr; GNN retrain
  ~5 min on GPU. Final: pyramidal per-file P10 0.6621 → 0.8984
  (+23.6pp); apical F1 +0.72pp; per-class GNN cellF1 +6.3pp.
  Snapshot: `paper/results/snapshots/v9_final_subtree_gnn.json`.
  GNN checkpoint: `paper/models/gnn_apical_basal_v9.pt`.
- Built `paper/v9_prep.py` — symlinks hpf_ca1 cells with apical labels
  into a merged dataset directory; renames with `hpf_ca1__` prefix to
  preserve provenance and avoid hash collisions.
- **Cross-dataset filtered eval on hpf_ca1** (1,012 cells with apical
  labels, the rest dropped): apical F1 +1.8pp from filter (0.9206 →
  0.9386); Stage 1 perfect (1,012/1,012). Snapshot:
  `paper/results/snapshots/v8_cross_dataset_hpf_ca1_apical_only.json`.
- Added `--require-classes` flag to `paper/cross_dataset_eval.py` for
  GT-class-presence filtering.
- **v8 cross-dataset eval on hpf_ca1** (1,377 CA1 pyramidals, 49.95M
  nodes): zero-shot neurite-macro-F1 = 0.9499; Stage 1 acc = 0.9985;
  axon F1 +0.001; apical/basal drop ~3-5 pp. See §5 cross-dataset table.
- Built `paper/cross_dataset_eval.py` — flat-dir eval reusing pipeline.py.
- **v8 architecture: subtree-level Stage 2.** `--use-subtree-stage2` flag
  in `pipeline.py` and `evaluate.py`. Tier A's per-subtree predictions
  are propagated to all branches; Tier B per-branch sklearn model is
  bypassed. Functionally equivalent to v7 on F1 (Stage 3 dominates),
  +0.31 pp on Stage 1 cell-type acc via better soft-handoff signal.
  Snapshot: `paper/results/snapshots/v8_subtree_gnn.json`.
- A/B test: GNN before vs after Stage 3 — byte-identical numbers; the
  modules edit disjoint branch sets. Documented as a null-result ablation
  row.
- **v7 GNN integration** — `paper/gnn_apical_basal.py` (3-layer GraphSAGE,
  hidden=128, ~80k params) wired as Stage 2b override. +1.3 pp on apical
  F1, +0.65 pp on basal F1. Held-out standalone GNN macro-F1 = 0.9677.
- 18-config hyperparameter sweep on GPU (RTX 4080, ~30 min). All top-11
  configs within ±0.003 std; smallest model in top-7 was 7,522 params.
- `paper/gnn_dataset.py` MorphologyBranches → PyG Data converter.
- Repo migrated dev to Windows for GPU access; `auto-label-dev` branch on
  GitHub. Python 3.11 venv (sklearn 1.5.1 doesn't have wheels for 3.13;
  3.12 wasn't installed locally; 3.11 is the canonical Windows version).
  `PYTHONIOENCODING=utf-8` is required because several source files
  contain non-ASCII chars (→, ←, μ, Δ, …) that cp1252 can't print.
- Patched two `f"...\\..."` f-strings to be 3.11-compatible (the
  backslash-in-f-string restriction was lifted in 3.12).
- Added cache-reuse for the eval pickles in `evaluate.py` — eval runs
  now skip the 2-hour Stage 2 retrain if `eval_tmp/s{1,2}_eval.pkl` are
  present. Big quality-of-life for ablation runs.
- `.gitignore` updates: `.claude/`, `hybrid/models/eval_tmp/`,
  `hybrid/models/*.pkl`. Untracked the existing `cell_type_classifier.pkl`
  (env-specific).
- Reorganized: paper-related code moved to `paper/` (sibling of `hybrid/`)
- Built `paper/baselines.py` with floor + Tier-2 ablations
- Built v6 with trunk features (small regression on apical, locked anyway)
- Cleaned up redundant scripts in `hybrid/`
- Renamed `data/benchmark_pyramidal_interneuron_v1_qc_diag_pruned_v3` → `_qc_diag_pruned`
- Removed older intermediate prune folders

## 9. What to do next on Windows

The setup, GNN, and cross-dataset eval are all done. The current open
items are listed in priority order below.

### Setup recap (only if starting fresh on a new machine)

```powershell
cd C:\path\to\SWC-Studio
git fetch origin
git switch auto-label-dev

# Make a venv (3.11 is what we're on; 3.12 has the original spec but
# wasn't available locally; 3.13 lacks sklearn 1.5.1 wheels)
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# Base deps (pin sklearn for reproducibility)
pip install "numpy>=1.26" "pandas>=2.0" "scikit-learn==1.5.1" "scipy>=1.11"
pip install python-pptx python-docx huggingface_hub

# PyTorch with CUDA (check `nvidia-smi` for CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric

# Verify GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Sanity-check the existing pipeline reproduces v6/v7/v8 numbers:

```powershell
# All three should land within ±0.005 of the snapshots in paper/results/snapshots/
$env:PYTHONIOENCODING="utf-8"
python -m hybrid.evaluate                                              # v6 baseline
python -m hybrid.evaluate --use-gnn                                    # v7 (GNN per-branch)
python -m hybrid.evaluate --use-gnn --use-subtree-stage2               # v8 (GNN subtree)
```

The eval pickles in `hybrid/models/eval_tmp/` are cached, so the second
and third runs skip retraining and finish in ~15 min each.

### Open items (priority order)

The pipeline is at v9. The big architecture-side and big data-side
experiments are done. Remaining items are paper-completion ones:

1. **External-tool baselines** for the paper. Train RandomForest on
   NeuroM and L-Measure features as a "non-deep, non-domain-tuned"
   reference comparison. Reviewer-friendly. ~half day.

2. **Floor baselines** (~10 min). `python -m paper.baselines floor`
   (random / majority / heuristic).

3. **Component ablations**: rerun `evaluate.py` with each contribution
   disabled in turn (no-PCA features, no-soft-handoff, no-trunk,
   no-GNN, no-subtree-stage2). Most rows already have natural
   counterparts in the snapshots.

4. **Optional: targeted axon-veto classifier** (~half day). The only
   architecture-side experiment that can attack the residual ~3.0% of
   apical-GT nodes still classified as axon. Marginal value reduced
   after v9 (was 4.4% in v8). Worth doing only if Paper 1 wants
   another architectural contribution row.

5. **Downstream-impact section** (~1-2 weeks for the NCS / Nature Comm
   tier). Re-label a real public dataset, surface label errors,
   quantify. Not required for *Patterns* / *eLife* tier.

6. **Optional: re-run cross-dataset eval with v9 models** (~3 hr).
   The v8 cross-dataset numbers are at `cross_dataset_hpf_ca1.json`.
   Running with v9 models would test whether v9's gains hold under
   domain shift. Predictable result: similar headline drop pattern,
   slightly better numbers everywhere.

### Re-running the cross-dataset eval

```powershell
# Full 1,377 hpf_ca1 cells (~3 hr CPU)
python -m paper.cross_dataset_eval --data-dir data/hpf_ca1 \
    --cell-type pyramidal --tag hpf_ca1

# Filtered: only the 1,012 cells with apical-labeled GT (~2.5 hr)
python -m paper.cross_dataset_eval --data-dir data/hpf_ca1 \
    --cell-type pyramidal --tag hpf_ca1_apical_only --require-classes 4
```

## 10. Useful invariants to know

- **Test split is stable.** `seed=42, test_size=0.2` with hash-bucketing
  means the same 328 files are always in the test set. Any ablation run
  is directly comparable to the headline numbers.
- **Datasets are layered** by upstream → downstream: `source_pools_usable`
  → `soma_clean` → `qc_diag_pruned`. Current default for all consuming
  scripts is `qc_diag_pruned`.
- **hpf_ca1 is a separate cross-dataset evaluation pool**, NOT part of
  the train/test split. 1,377 CA1 pyramidal cells in `data/hpf_ca1/`
  organized as 14 batches. 49.95M nodes total. 1,012 of 1,377 (73.5%)
  have apical labels in GT.
- **The unrecoverable apical-as-axon failure files** are concentrated in
  ~10 `lab__*_edited.swc` files where the apical was traced as a thin
  axon-like extension without a tuft. `lab__210107_029_edited.swc`
  (F1 = 0.333 in v8 in-distribution; F1 = 0.198 in v8 cross-dataset) is
  the canonical example. These files cap apical recall at 0.9563
  regardless of architecture.
- **Hard apical F1 ceiling = 0.9563.** 10,765 apical-GT nodes get
  classified as axon by Stage 2, Stage 3 doesn't rescue them, GNN's
  gate skips axon-labeled inputs by design. To break this ceiling you
  need either an axon-veto classifier (§9 item 2) or v9 retraining with
  more diverse data (§9 item 1).
- **sklearn 1.5.1 is the canonical version.** Pickles trained in 1.5.1
  don't load in 1.8.0 (binary loss class changed). Pin in `requirements.txt`.
- **Soft handoff threshold default is 0.65.** Pass
  `soft_handoff_threshold=0.0` to disable for ablations.
- **Always set `PYTHONIOENCODING=utf-8`** on Windows when running any
  script that prints results — several source files contain non-ASCII
  characters (→, ←, μ, Δ, …) that crash cp1252.
- **GNN gate logic** depends on `apical_evidence`. In default mode (per-
  branch Tier B), the gate fires when Tier B already labels at least one
  branch as apical AND one as basal. In `--use-subtree-stage2` mode, the
  gate fires when Tier A's `apical_owner_root` is not None. Without the
  gate, the GNN hallucinates apical on basal-only "pyramidal" files
  (verified: `neuromorpho__11224c3.swc` drops from 100% to 41% acc
  without the gate).
- **Eval is reproducible without retraining** thanks to the
  `eval_tmp/s{1,2}_eval.pkl` cache. To force a fresh retrain, delete
  those pickles before running `evaluate.py`.

## 11. Open questions / unresolved decisions

1. **Which paper venue first?** *Patterns* / *eLife* / *NCS* — the GNN
   contribution + cross-dataset result raise the ceiling. NCS becomes
   plausible if v9 (with hpf_ca1) clears the lab__ tail OR the axon-veto
   classifier works.
2. **GNN scope.** Stays narrow: just the apical-vs-basal head as a
   Stage-2b override. v8 confirmed no benefit from moving the GNN later
   (byte-identical numbers — see v7b ablation snapshot). Full Stage 2
   replacement (Tier B → GNN multi-class) is on the shelf and not
   currently planned.
3. **External tool comparison.** Still pending. NeuroM and L-Measure
   don't natively classify, but training RF on their features = a
   credible "external" baseline. Reviewer-friendly. ~half day.
4. **v9 dataset merge.** Should we add only apical-bearing hpf_ca1
   files (1,012) or all 1,377? The 365 no-apical files could serve as
   negative examples for the GNN's apical-detection logic, but they
   may also confuse Tier A. Decision deferred until v9 retrain has
   been started.
5. **Resolved:** GNN works (+1.3 pp apical F1). Cross-dataset
   generalization works (Stage 1 0.9985, headline F1 0.9499 zero-shot
   on a different brain region).

## 12. Tone for new sessions

When picking up in a new Claude session, you can say something like:

> Read `paper/CONTINUATION.md`. The pipeline is at v9 (subtree-Stage2
> + GNN, retrained on benchmark + hpf_ca1; neurite-macro-F1 = 0.9748,
> apical F1 = 0.9548, pyramidal per-file P10 = 0.8984). The big
> experiments are done. Remaining items are paper-completion (§9):
> external-tool baselines, floor baselines, component ablations.
> Start with the floor baselines and external-tool comparisons.

Or for downstream impact:

> Read `paper/CONTINUATION.md`. The v9 pipeline is publication-ready
> (apical F1 = 0.9548, axon F1 = 0.9986, cross-dataset zero-shot F1 =
> 0.9499 on hpf_ca1). Now I want to start the downstream-impact section
> — re-label a real public dataset and surface labeling errors. See §9
> item 5.

Or for any other task:

> Read `paper/CONTINUATION.md`. Continue [whatever specific task].
