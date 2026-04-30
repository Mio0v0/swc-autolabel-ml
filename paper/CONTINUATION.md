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

## 5. Latest evaluation results (v6, dataset = qc_diag_pruned, 1,737 cells)

| Metric                 | Value   |
|------------------------|---------|
| Overall accuracy       | 0.9918  |
| Pooled F1              | 0.9762  |
| Neurite F1             | 0.9682  |
| Per-file mean F1       | 0.9644  |
| **Per-file P10**       | **0.9093** |
| Stage 1 cell-type acc  | 0.9817  |
| **Apical F1**          | **0.9348** |
| **Basal F1**           | **0.9725** |
| Soma F1                | 1.0000  |
| Axon F1                | 0.9974  |

Big jump from v4 → v5: per-file P10 went from 0.679 → 0.901 (PCA features +
soft handoff fixed the slice-prep tail). v6 added trunk features but they
**did not improve apical F1** (slight regression of 0.0017) — likely
redundant with existing subtree-rank features.

Snapshot of v6 lives at `paper/results/snapshots/v6_full_pipeline.json`.

## 6. The next big move: small GNN for apical vs basal

Apical-vs-basal is fundamentally a graph-topology problem. Hand-crafted
features have saturated (~F1 0.93-0.94 apical, 0.97 basal). A small GNN
that operates on the branch graph could capture topology patterns that no
scalar feature encodes.

**Proposed architecture (not yet implemented):**
- 2-layer GraphSAGE, hidden dim 64, ~12k parameters
- Input: 47 dendrite-relevant per-branch features
- Edges: parent ↔ child branches in the cell tree
- Output: 2-class softmax (apical vs basal), only used for pyramidal dendrite branches

**Integration plan:**
- Stage 2 stays unchanged for axon-vs-dendrite (current ensemble works well there: F1 0.997)
- New Stage 2b: when Stage 2 says "dendrite" for a pyramidal branch, the GNN re-decides apical vs basal
- Drop-in replacement; falls back to Stage 2 if GNN regresses

**Expected outcome:** ~50% chance of +2–4 pp on apical F1; ~30% modest help; ~20% no change.
Worst case is a publishable negative result.

**Effort:** ~5–7 days of focused work, all on Windows GPU.

**Required deps (Windows):**
- PyTorch with CUDA (cu121 or cu118 depending on driver)
- torch-geometric
- See `paper/CONTINUATION.md` step 4 below for install commands

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
- ✅ Headline numbers (in `evaluation_results.json`)
- ✅ Architectural novelty (PCA, soft handoff, hybrid)
- ✅ Failure-mode taxonomy (BAD GT findings)
- ⏳ Component ablations (paper/baselines.py provides — see step 9 below)
- ⏳ Floor baselines (random / majority / heuristic — paper/baselines.py provides)
- ⏳ External tool baselines (NeuroM features + RandomForest, L-Measure features + RF) — to add
- ⏳ Cross-dataset generalization (train Allen, test NeuroMorpho) — to add
- ⏳ Downstream impact section (re-label real public dataset, surface label errors) — to add for NCS/Nature Comm tier
- ⏳ GNN apical-basal head — to build on Windows GPU

## 8. Recent state changes (most recent first)

- Migrated dev to Windows for GPU access (NVIDIA card available)
- Pushed `auto-label-dev` branch to GitHub
- Updated `.gitignore`: `.claude/`, `hybrid/models/eval_tmp/`, `hybrid/models/*.pkl`
- Untracked the existing `cell_type_classifier.pkl` (env-specific)
- Reorganized: paper-related code moved to `paper/` (sibling of `hybrid/`)
- Built `paper/baselines.py` with floor + Tier-2 ablations
- Built v6 with trunk features (small regression on apical, locked anyway)
- Cleaned up redundant scripts in `hybrid/`
- Renamed `data/benchmark_pyramidal_interneuron_v1_qc_diag_pruned_v3` → `_qc_diag_pruned`
- Removed older intermediate prune folders

## 9. What to do next on Windows

In rough order:

### Setup (one-time, ~2-3 hours including data transfer)

```powershell
cd C:\path\to\SWC-Studio
git fetch origin
git switch auto-label-dev

# Make a venv
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1

# Base deps (pin sklearn to match Mac build for reproducibility)
pip install "numpy>=1.26" "pandas>=2.0" "scikit-learn==1.5.1" "scipy>=1.11"
pip install python-pptx python-docx

# PyTorch with CUDA (check `nvidia-smi` for CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric

# Verify GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Manually transfer `data/` (4.5 GB, not in git) from Mac via external SSD,
rsync, or cloud storage.

### Regenerate models (~30-45 min total)

```powershell
python -m hybrid.train_stage1
python -m hybrid.train_stage2
python -m hybrid.evaluate
```

Sanity-check: pooled neurite F1 should be ≈ 0.968 (within ±0.005 of Mac
v6 value). If it differs more, sklearn version is probably wrong.

### Then build the GNN (~5-7 days)

1. **Day 1 — data module.** Build `paper/gnn_dataset.py` that takes a
   `MorphologyBranches` and produces a `torch_geometric.data.Data` graph
   (one node per branch, edges between parent/child branches, node features
   = subset of `BRANCH_FEATURE_NAMES`).

2. **Day 2-3 — model + training.** Build `paper/gnn_apical_basal.py` with
   a 2-layer GraphSAGE, training loop with 5-fold CV, save best checkpoint
   to `paper/models/gnn_apical_basal.pt`.

3. **Day 4-5 — hyperparameter sweep.** Try {hidden dim 32/64/128}, {layers
   2/3}, {dropout 0/0.2/0.5}. Pick best per-CV-mean F1.

4. **Day 5-6 — pipeline integration.** Modify `pipeline.py` so that for
   pyramidal cells, dendrite-classified branches go through the GNN to
   re-decide apical vs basal. Behind a flag (`use_gnn=True`).

5. **Day 6-7 — re-evaluate + writeup.** Run `python -m hybrid.evaluate
   --use-gnn`. Compare to v6 numbers. If it helps, incorporate into the
   paper. If it doesn't, document as ablation row.

### Then run the remaining baselines

```powershell
python -m paper.baselines floor          # random / majority / heuristic (~10 min)
python -m paper.baselines all            # aggregates everything available
```

For heavier ablations (no-pca, no-trunk, no-cell-type), each requires a
fresh ~30 min eval run with that component disabled. See
`paper/baselines.py` docstring for the procedure.

## 10. Useful invariants to know

- **Test split is stable.** `seed=42, test_size=0.2` with hash-bucketing means the same 328 files are always in the test set. Any ablation run is directly comparable to the headline numbers.
- **Datasets are layered** by upstream → downstream: `source_pools_usable` → `soma_clean` → `qc_diag_pruned`. Current default for all consuming scripts is `qc_diag_pruned`.
- **The 5 worst v6 files** are mostly BORDERLINE (atypical morphology, valid GT) plus 1 SUSPICIOUS (z-axis-inverted apical) plus 1 APPEARS HEALTHY (real model failure on `homo-7.swc`). Diagnostic findings: ~6% of files are at the morphological ceiling; further dataset cleanup won't help.
- **sklearn 1.5.1 is the canonical version.** Pickles trained in 1.5.1 don't load in 1.8.0 (binary loss class changed). Pin in `requirements.txt`.
- **Soft handoff threshold default is 0.65.** Pass `soft_handoff_threshold=0.0` to disable for ablations.

## 11. Open questions / unresolved decisions

1. **Which paper venue first?** *Patterns* / *eLife* / *NCS* — depends on whether GNN works and how much downstream-impact section gets built.
2. **GNN scope.** Just apical-vs-basal head, or full Stage 2 replacement? Currently planned: just the head, lower risk.
3. **Cross-dataset generalization test.** Worth running for Paper 1 — train on Allen+lab, test on NeuroMorpho only.
4. **External tool comparison.** NeuroM and L-Measure don't natively classify, but training RF on their features = a credible "external" baseline. Reviewer-friendly.

## 12. Tone for new sessions

When picking up in a new Claude session, you can say something like:

> Read `paper/CONTINUATION.md`. We're building the GNN apical-vs-basal head described in section 6. Start with section 9's GNN day-1 task: build `paper/gnn_dataset.py`.

Or for any other task:

> Read `paper/CONTINUATION.md`. Continue [whatever specific task].
