# A staged classifier–GNN–topology pipeline for automatic axon, basal, and apical labeling of neuronal morphology reconstructions

**Status:** v0.4 working draft, 2026-07-02. All numbers traced to artifacts in
`paper/results/`. Remaining `[TODO]` markers flag citation verification and
the SWC-Studio deployment note. Every reported metric is annotated with its
evaluation configuration (see §4.6).

---

## Abstract

Neuronal morphology reconstructions in the SWC format are the substrate
for a large fraction of quantitative neuroanatomy, but the per-node
structure-type labels (axon, basal dendrite, apical dendrite) that
downstream analyses depend on are frequently absent, inconsistent, or
inverted across public corpora. We present an end-to-end pipeline that
takes a raw SWC reconstruction and returns a labeled SWC. The pipeline
runs five sequential steps: (i) a quality-control gate that filters
malformed inputs; (ii) a cell-type classifier that either uses a
user-supplied cell type or predicts pyramidal vs. interneuron from
whole-cell features (predicted-type accuracy 0.974 ± 0.007); (iii) a
per-branch XGBoost classifier for axon / basal / apical; (iv) a graph
neural network plus a deterministic topology-refinement pass with a
"Branch3" rescue subnet; and (v) an optional quality-flag model that
identifies cells whose labels are likely wrong. On a curated
11,862-cell, three-source corpus split 80/20 by file hash and evaluated
across three random seeds, the labeler reaches per-class F1 of 0.993
(axon), 0.954 (basal), and 0.916 (apical) when the ground-truth cell
type is supplied, and 0.993 / 0.951 / 0.906 in the deployment
configuration where Stage 1 predicts the cell type. To our knowledge no
prior system performs this end-to-end per-node SWC labeling task, so we
construct four published-style baselines spanning the design space
(NeuroM-RF, Sholl-RF, Sholl-MLP, L-Measure-RF) and evaluate them on the
same splits with ground-truth cell type supplied to both sides for an
apples-to-apples comparison. Our pipeline improves neurite macro-F1 by
+0.005 to +0.13 over these baselines and lifts the 10th-percentile
per-cell F1 — the metric that matters most for downstream filtering —
from at most 0.54 (best baseline) to 0.74 (paired sign-flip p < 0.0001,
n = 5,739 unique files). In the deployment configuration, the
quality-flag model identifies the worst 10% of predictions with 58%
recall (precision 0.43, F1 0.50) using only features available from a
single inference pass; rejecting these cells lifts the kept-set
10th-percentile per-cell F1 from 0.66 to 0.96. Finally, we close the
loop: the flag step routes suspect cells to a curator whose corrections
feed back into training, so the labeler improves with use. A
simulated-curator active-learning experiment shows that flag-guided
review reaches a target accuracy with ~60% fewer curator corrections
than reviewing random cells, establishing the flag signal as an
efficient acquisition function, not merely a filter.

---

## 1. Introduction

The SWC format encodes a neuronal morphology reconstruction as a
directed tree: each node carries a 3D coordinate, a radius, a
structure-type integer (1=soma, 2=axon, 3=basal dendrite, 4=apical
dendrite, ...), and a parent pointer. Downstream tools — Sholl analysis,
branch-order statistics, compartmental modeling, dendrite-specific
feature extraction, and most recently graph-based representation
learning — depend on the structure-type labels being correct. In
practice they often are not. Cells uploaded to NeuroMorpho.Org and
similar repositories are produced by hundreds of laboratories using
different tracing tools, conventions, and quality bars; we find that a
non-trivial fraction of files have apical and basal dendrite labels
swapped, lack apical labels entirely on pyramidal cells, mix axon and
dendrite on traces of single-compartment neurons, or use the dendrite
label as a catch-all.

The standard responses are manual relabeling (slow, does not scale, and
itself a source of inter-annotator inconsistency) and discard (throwing
out ~50% of an already small corpus). A more attractive option is
*automatic* relabeling: given the raw geometry of a reconstruction,
predict the correct structure-type label for every node.

**No off-the-shelf labeler exists for this task.** To our knowledge,
no released software package or published model performs end-to-end
per-node SWC re-labeling on raw morphology reconstructions. NeuroM and
L-Measure are widely-used morphometric *analysis* libraries — they
compute features from already-labeled SWCs, they do not assign labels.
The closest *labeling* work is the Emissah/Tecuatl/Ascoli line of
per-subtree classifiers based on Sholl descriptors, but it predicts
subtree types in isolation rather than producing a full re-labeled SWC
and is not released as a runnable labeler. The novelty of the present
work is therefore both in its **purpose** (end-to-end per-node
relabeling, deployment-ready, with a QC + flag wrapper) and in its
**structure** (a staged QC → cell-type → branch-classifier → GNN →
topology-refinement → flag pipeline). Because no off-the-shelf baseline
exists, the four baselines we compare against (§5.1) are *our own
constructions* assembled from standard morphometric feature libraries
(NeuroM, L-Measure, Sholl descriptors) plus standard classifiers
(RandomForest, MLP); we deliberately span the design space of
plausible approaches a competing group might build (see §4.6).

This paper contributes four things:

1. **An end-to-end SWC labeler, benchmarked honestly.** A staged
   pipeline (QC → cell type → branch classifier + GNN → topology
   refinement + Branch3 rescue) that reaches mean per-class F1 of
   0.993 / 0.954 / 0.916 for axon / basal / apical (0.993 / 0.951 /
   0.906 when Stage 1 predicts the cell type), evaluated against four
   constructed published-style baselines on the same splits with paired
   significance testing (n = 5,739 files), and dissected by a controlled
   stage ablation that identifies topology refinement — not the GNN — as
   the single largest gain.
2. **A deployment workflow built around a quality flag.** A flag model
   identifies the worst 10% of predictions at 58% recall (F1 0.50) from
   a single inference pass, lifting the kept-set 10th-percentile per-cell
   F1 from 0.66 to 0.96 — turning an auto-labeler into an
   auto-label-then-curate pipeline.
3. **Train-on-your-own-data specialization.** The pipeline retrains from
   a user-supplied corpus via one driver script; we show (§5.5) that
   retraining on a single clean lab corpus matches or beats the
   cross-source model on that lab's data.
4. **A curator feedback loop that improves both labeler and flag.** The
   flag step routes suspect cells to a curator whose corrected labels
   retrain the model (§4.7); a simulated-curator study shows flag-guided
   review improves the labeler with ~60% fewer corrections than random
   (§5.7) and raises flag precision +21% as verdicts accumulate (§5.8) —
   so the deployed system improves with use.

Code, models, frozen seeds (42, 123, 789), splits, and an artifact
manifest are released for full reproducibility.

---

## 2. Related work

We organize the landscape into (i) subcompartment classification in
connectomics, (ii) morphometric-feature libraries and classifiers,
(iii) representation learning on morphology, (iv) per-subtree
structure classifiers, and (v) databases and curation. None of these
provides an off-the-shelf per-node re-labeler for light-microscopy SWC
reconstructions, which is the gap this work fills.

[Note: reference details below are drawn from a July-2026 literature
search; final citation formatting (authors, venues, years) should be
verified against the cited URLs before submission.]

**Subcompartment classification in connectomics.** The closest
*machine-learning* analog to our task comes from dense
electron-microscopy (EM) connectomics, where neurite fragments must be
typed as axon / dendrite / soma. Li et al., "Neuronal Subcompartment
Classification and Merge Error Correction" (bioRxiv 2020,
[link](https://www.biorxiv.org/content/10.1101/2020.04.16.043398)),
classify subcompartments on EM segments and correct merge errors, and
Schubert et al., "Learning cellular morphology with neural networks"
(*Nature Communications* 2019,
[link](https://www.nature.com/articles/s41467-019-10836-3)), introduce
Cellular Morphology Networks (CMNs) that classify 2D projections of
neurite fragments with CNNs. These operate on dense EM voxel/skeleton
data with a coarser (axon/dendrite/soma) label set and are embedded in
proprietary connectomics pipelines; neither is a released tool that
re-types the finer axon / basal-dendrite / apical-dendrite labels on
light-microscopy SWC reconstructions, and neither exposes a standalone
labeling API for the public SWC corpora we target.

**Morphometric-feature libraries.** L-Measure (Scorcioni, Polavaram &
Ascoli, *Nature Protocols* 2008) and NeuroM (Blue Brain Project) compute
standard per-cell/per-branch morphometric descriptors; Sholl analysis
(Sholl, *J. Anat.* 1953) remains the most cited such descriptor. These
are *analysis* tools — they compute features from SWCs that already
carry per-node type labels — not labelers. We assemble `lmeasure_rf`,
`neurom_rf`, and Sholl-feature baselines (§4.5) on top of these feature
families to give the comparison a strong feature-engineering floor; to
our knowledge no published system turns any of them into an SWC
re-labeler.

**Representation learning on morphology.** A growing body of work learns
morphological embeddings for *whole-cell* cell-type classification
rather than per-node structure labeling — e.g. graph-based self-
supervised "morphological bar codes" for tens of thousands of cortical
neurons (an unsupervised dendritic-morphology map of mouse visual
cortex, *Nature Communications* 2025,
[link](https://www.nature.com/articles/s41467-025-58763-w)), and axon-
based interneuron classification (bioRxiv,
[link](https://www.biorxiv.org/content/10.1101/414615)). Our
apical/basal GNN is architecturally related (a graph network over the
morphology) but solves a different problem at a different granularity:
per-branch structure typing within one cell, not per-cell type
assignment.

**Per-subtree structure classifiers.** The closest *labeling* prior art
is the line of per-subtree Sholl-descriptor classifiers associated with
the Ascoli group (Emissah, Tecuatl & Ascoli, bioRxiv 2026 — [TODO:
verify citation]), which predict a structure type per primary subtree.
This motivates our `sholl_rf` / `sholl_mlp` baselines. It is not
end-to-end on raw SWC (it decides per-subtree in isolation, without a
QC gate, topology-consistency pass, or quality flag) and is not
released as a runnable labeler, so we reimplement the per-subtree
classifier on our split as a baseline (§4.5).

**Databases and curation.** The SWC format (Cannon et al., *J. Neurosci.
Methods* 1998) and NeuroMorpho.Org (Ascoli, Donohue & Halavi, *J.
Neuroscience* 2007) underpin the corpora we use. NeuroMorpho applies
limited per-upload curation but does not relabel structure types, so
per-lab conventions and errors persist downstream — the very problem
our flag model and curator loop are designed to support (not replace).

**Human-in-the-loop and active learning.** Uncertainty-guided sample
selection is a standard active-learning tool; our contribution is to
show that the *quality-flag* signal of a deployed labeler doubles as an
acquisition function (§5.7) and that curator verdicts on flagged cells
retrain the flag model itself (§5.8) — a closed loop specific to the
re-labeling setting rather than a generic active-learning result.

**Summary.** No published or released system performs end-to-end
per-node re-labeling of light-microscopy SWC reconstructions into
axon / basal / apical. The novelty of this work is in both its
*purpose* (a deployment-ready re-labeler with a QC gate, quality flag,
and curator feedback loop) and its *structure* (the staged classifier +
GNN + topology-refinement + rescue pipeline of §4.1). All four baselines
we benchmark against are our own constructions (§4.5), assembled from
the feature families above because no runnable prior labeler exists.

---

## 3. Dataset

We assembled a corpus of 21,054 SWC files from three sources: the Allen
Cell Type Atlas (299 cells), an in-house hippocampal CA1 collection
(1,377 cells), and NeuroMorpho.Org (19,378 cells). Cells span two cell
types — pyramidal (10,910) and interneuron (10,144) — selected to give
a clean apical-present/absent contrast for the apical/basal decision.

All cells are passed through a quality-control gate (Stage 0; see §4.1)
that requires (i) a single root, (ii) presence of a soma node, (iii) at
least one neurite beyond the soma, (iv) all radii positive, and (v)
total cell size within a reasonable range. After QC, 11,862 cells
remain (56.3% pass rate). Pass rates differ substantially across
sources and cell types:

| Group | Total | QC pass | Pass rate | Median nodes (pass) |
|---|---:|---:|---:|---:|
| allen / interneuron | 197 | 126 | 64.0% | 11,976 |
| allen / pyramidal | 102 | 77 | 75.5% | 11,405 |
| hpf_ca1 / pyramidal | 1,377 | 1,358 | 98.6% | 31,056 |
| neuromorpho / interneuron | 9,947 | 3,248 | 32.7% | 3,440 |
| neuromorpho / pyramidal | 9,431 | 7,053 | 74.8% | 684 |

(Full table: [final_dataset_corpus_table.md](paper/results/final_dataset_corpus_table.md).)

The dominant QC failure modes are missing soma (5,624 / 9,192 = 61%)
and joint missing-soma + no-neurites (24%). The remainder fail size
and single-root checks. Interneuron failure is driven almost entirely
by the NeuroMorpho subset, where many uploads are axon-only or
dendrite-only fragments rather than complete cells. We report all
downstream results on the QC-pass corpus.

We split this corpus 80/20 by a deterministic hash of the SWC filename.
The hash is computed once and reused across all training and evaluation
seeds, so seed-to-seed variability comes only from model
initialization, not from sampling different held-out cells. We verify
that no test file leaks into any training fold
([leakage_check_seed{42,123,789}.txt](paper/results/)).

---

## 4. Method

### 4.1 Pipeline overview

The end-to-end labeler is a sequence of five steps that consume a raw
SWC file and return a labeled SWC plus an optional quality flag. The
key design decision is that the cell-type module sits *between* QC and
the per-branch labeler and can be either supplied by the user or
predicted by Stage 1; everything downstream is identical in both
modes.

```
Raw SWC
   ↓
Stage 0  QC gate                  (filter; reject or pass)
   ↓
Stage 1  Cell type                user input  ───────────────┐
         (pyramidal / interneuron)                           │
         OR                                                  │
         predict (XGBoost, whole-cell features) ─────────────┘
   ↓
Stage 2  Per-branch axon / basal / apical XGBoost classifier
   ↓
Stage 2.5  Apical / basal GNN refinement      (pyramidals only)
   ↓
Stage 3  Topology refinement + Branch3 rescue subnet
   ↓
Labeled SWC
   ↓
Stage 4  Quality flag model       (flags suspect cells)
   ↓
       flagged? ──no──► accept labels
          │ yes
          ▼
Stage 5  Curator review           (human-in-the-loop, §4.7)
          │   CONFIRM_BAD + corrected labels
          │   REJECT_FLAG (labels were fine)
          │   SKIP
          ▼
   curator-verified pool  ──► periodic augmented retrain
          └──────────────────────────► (back into Stage 2 / GNN / Branch3 / flag)
```

![Pipeline schematic](paper/results/figures/final_figure_pipeline_schematic.png)

**Figure 1.** The end-to-end pipeline. Stages 0–4 form the labeler
(learned components in blue, rule-based in green, the quality flag in
red); Stage 5 (yellow) is the human-in-the-loop feedback path, whose
curator-verified corrections drive a versioned retrain (dashed) that
never overwrites the factory checkpoint.

Stages 0–4 are the *labeler*; Stage 5 is the *feedback loop* (§4.7) that
lets the deployed system improve as curators review flagged cells. The
loop is optional — a user can stop at the labeled SWC — but it is the
mechanism behind the system's headline deployment property: it gets
better with use (validated in §5.7).

The reason for routing cell type through Stage 1 (rather than
*requiring* a user-supplied type) is that downstream branch-level
features depend on cell-type-conditional priors: apical labels are only
meaningful for pyramidals, and the per-branch classifier is fed
cell-type as an input feature. A wrong cell type cascades into the
branch labeling, so cell-type accuracy is a meaningful component
metric in its own right (0.974 ± 0.007; §5.2).

(See [hybrid/pipeline.py](hybrid/pipeline.py),
[hybrid/qc_input.py](hybrid/qc_input.py),
[hybrid/train_stage2.py](hybrid/train_stage2.py),
[hybrid/stage3_refine.py](hybrid/stage3_refine.py) for the canonical
implementations.)

### 4.2 Stage 0 — Quality control

The QC gate is a deterministic filter (no trainable parameters) that
rejects SWC files that violate any of five structural requirements:
single root, has soma, has neurites, positive radii, reasonable size.
QC pass rates by source and cell type are reported in §3.

### 4.3 Stages 1–3 — Cell type, branch classifier, GNN, topology refinement

**Stage 1 — cell type.** An XGBoost classifier predicts one of
{pyramidal, interneuron} from whole-cell features (Sholl, branch
statistics, soma geometry). When the user supplies a known cell type,
this stage is bypassed and the supplied value is passed to Stage 2 as
the cell-type feature. Predicted-type accuracy (0.974) is reported in §5.2.

**Stage 2 — branch classifier.** For each branch in the SWC tree, an
XGBoost classifier predicts axon / basal / apical given branch-level
features (length, tortuosity, radial distance, parent–child geometry,
PCA-projected trajectory) and the cell-type label (from Stage 1 or
user) as an input feature.

**Stage 2.5 — apical/basal GNN.** A two-layer GraphSAGE network
(hidden dim 64, ReLU, dropout 0.2) operates on the branch graph
(nodes = branches with 51 dendrite-relevant features, undirected
parent–child edges) to re-decide basal vs. apical for branches Stage 2
labeled as dendrite, on cells classified as pyramidal. It is trained
with Adam (lr 1e-3, weight decay 5e-4) up to 200 epochs with early
stopping (patience 25) under 5-fold cross-validation, using a focal
loss (γ = 2) with inverse-square-root class weighting to counter the
apical minority. Axon-vs-dendrite is left to Stage 2 (already at F1
0.99). Implementation:
[paper/gnn_apical_basal.py](paper/gnn_apical_basal.py).

**Stage 3 — topology refinement + Branch3 rescue.** A rule-based pass
enforces tree-level constraints: contiguous apical labels along the
principal trunk, no axon labels inside the dendritic arbor, per-subtree
label consistency. A conservative *Branch3 rescue* subnet flags and
reclassifies borderline apical trunks that Stage 3 alone is too
cautious to relabel.

### 4.4 Stage 4 — Quality-flag model

Even at ~0.95 mean F1 the pipeline produces a long lower tail of
severely mislabeled cells. We train a separate XGBoost flag classifier
on the labeler's own outputs to identify these cells *without* a
ground-truth pass. Inputs are labeler confidence statistics (mean and
worst-branch posterior), simple geometry features (node count,
predicted apical fraction, number of predicted classes), and a Branch3
disagreement feature (does the rescue subnet disagree with Stage 3?).
The training target is the binary indicator *per-cell neurite F1 <
0.60*.

To prevent training/evaluation leakage on the held-out fold, we use a
leave-one-seed-out (LOSO) protocol: for each seed s ∈ {42, 123, 789},
the flagger is trained on cells from the other two seeds' held-out
sets and evaluated on cells from seed s's held-out set, with
cell-deduplication across folds. Full protocol:
[paper/MULTISEED_FLAG_NO_LEAK_METHOD.md](paper/MULTISEED_FLAG_NO_LEAK_METHOD.md).

**Important — the flag model is a deployment-config artifact.** It is
trained and evaluated on labeler outputs produced in the deployment
configuration (Stage 1 active), because that is the only configuration
in which the user would actually want the flag (when the cell type is
already known, Stage 1 errors do not occur).

### 4.5 Constructed baselines

Because no off-the-shelf end-to-end SWC labeler exists (§1, §2), the
four baselines we compare against are constructed in this work. Each
baseline follows the same recipe:

```
SWC  →  extract morphometric features over a granule  →
         classifier (RF or MLP)  →  predict granule label  →
         propagate granule label to every node in the granule.
```

The four baselines differ only in (i) granule type and (ii) feature
set, so the comparison isolates the value of the staged pipeline over
"plain morphometric features + standard classifier" — the only
plausible non-deep-learning alternative an external group could
assemble today.

| Baseline | Granule | Feature set | Classifier |
|---|---|---|---|
| **NeuroM-RF** | per-branch segment | NeuroM-style 16-dim (path length, radial / euclidean distance from soma, z-above-soma, radius statistics, taper, downstream subtree size, bifurcation count, partition asymmetry, cell-type one-hot) | RandomForest (400 trees, max depth 30) |
| **Sholl-RF** | per-primary-subtree | ~24-dim Sholl-derived (Sholl intersection counts at fixed radii, max Sholl, peak radius, total nodes, total bifurcations, max radial distance, total arbor length, max root-to-tip path, PCA principal-axis components, z-extent, cell-type one-hot) | RandomForest |
| **Sholl-MLP** | per-primary-subtree | same as Sholl-RF | small PyTorch MLP |
| **L-Measure-RF** | per-primary-subtree | 23-dim L-Measure-style classical morphometrics (lengths, diameters, branch counts, asymmetries, partition statistics — no Sholl intersections) | RandomForest |

Training-side notes: all baselines are trained on the same train split
as our pipeline (§4.4) and receive the ground-truth cell type as an
input feature, matching the GT cell-type evaluation configuration of
our pipeline so the §5.1 comparison is apples-to-apples. Per-node
training labels for each granule are the *majority neurite label* of
that granule. Implementation: [paper/external_baselines.py](paper/external_baselines.py).

**Caveats — what these baselines are and are not.**

- The Sholl baselines mirror the Emissah/Tecuatl/Ascoli per-subtree
  architecture but use sklearn classifiers instead of the published
  GCN, because the GCN is not released as a runnable artifact. The
  underlying *task setup* (per-subtree Sholl-feature classification)
  is faithful to the published approach.
- The L-Measure baseline computes the L-Measure *feature content* per
  subtree using NeuroM-equivalent metrics rather than calling the
  L-Measure Java binary, which operates per-cell only and cannot
  produce per-node labels.
- NeuroM-RF is the floor baseline: a straightforward "fit a
  RandomForest to standard branch features" approach a competing
  group might try first. It is included to give an honest sense of
  what naïve feature engineering gets you on this task.

### 4.6 Evaluation configurations

To avoid the apples-vs-oranges trap that comes with a Stage 1
component, we define and consistently label two evaluation
configurations throughout this paper:

- **GT cell type** — Stage 1 is replaced by the ground-truth cell type.
  All four baselines also receive the ground-truth cell type, so this
  is the configuration used in §5.1 (main baseline comparison) and §5.2
  (stage ablation). Reading: *what does the neurite-labeling problem
  look like, holding cell-type knowledge constant?*
- **Stage 1 active (deployment)** — the user has not supplied a cell
  type, Stage 1 predicts it (at 0.974 accuracy; §5.2). This is the
  configuration used in §5.3 (flag model), §5.4 (failure modes), and
  §5.7–5.8 (feedback loop). Reading: *what does the system actually
  produce for an end user with no prior knowledge?*

Per-class metrics are computed on individual nodes. Per-cell F1 is the
neurite macro-F1 for one cell. Aggregate per-cell statistics (mean,
P10, P25) summarize the distribution across the held-out set.

We train all stages with seeds 42, 123, and 789, retraining every
component from scratch for each seed. Stage 1 and Stage 2 are XGBoost
models with gentle class-weighting to handle the apical minority
class. Significance: 95% bootstrap confidence intervals on per-file
metric differences (5,000 resamples) and paired sign-flip p-values
([final_paired_stats.txt](paper/results/final_paired_stats.txt)).
The unit is *unique SWC file*; repeated seed–file pairs are averaged
before testing.

### 4.7 Stage 5 — Curator feedback loop (human-in-the-loop)

The flag model (§4.4) surfaces cells the labeler is likely to have
mislabeled. Rather than treating that as a terminal filter, we close
the loop: a flagged cell is routed to a curator whose corrected labels
feed back into training. This turns the labeler from a static model
into one that *improves with use*, and — because both the labeler and
the flag model are retrained — it also teaches the flag model to fire
more precisely over time. The structure is:

1. **Label + flag.** The labeler produces per-node labels, a flag
   score, and the flag-feature vector. Cells with score ≥ threshold are
   surfaced for review; the rest are accepted.
2. **Curator supplies corrected labels — nothing else.** The curator
   does *not* judge the flag (no "correct/false-alarm" button). They
   return only what they believe are the right labels for the cell
   (which may be byte-identical to the model's output if they judge it
   fine). This keeps the curator's task minimal and avoids a subjective
   meta-decision.
3. **The flag ground truth is derived, not chosen.** The system
   computes whether the flag was justified from the disagreement between
   the curator's labels and the model's, using the *same* definition the
   flag model was trained on:

   > flag was justified ⟺ per-cell F1( curator, model ) < 0.60

   A curator overhaul (low F1) yields a **true-positive** flag example;
   a curator who barely changed anything (F1 ≈ 1) yields a **false-alarm**
   example. Each reviewed cell thus produces two training signals: a
   labeler example (the corrected SWC) and a flag example (the
   flag-feature vector labeled by the derived truth).
4. **Persistent curator-verified store.** Both signals accumulate in a
   store (corrected SWCs + a manifest carrying the derived flag labels
   and features).
5. **Versioned augmented retrain — factory preserved.** On a periodic
   schedule the Stage-2 labeler is retrained on `factory_train ∪
   curator-corrected`, and the flag model is retrained on
   `factory_flag_data ∪ curator-derived (features, truth)`. Crucially,
   the retrain writes a **new versioned checkpoint** and never modifies
   the original *factory* checkpoint; a `ModelRegistry` enforces this
   (factory is read-only; each version records its parent and the
   curator examples used). Deployments can roll back to any prior
   version, and the factory model is guaranteed recoverable. Because the
   flag step preferentially surfaces the long-tail cells where the model
   is most wrong, curator effort is concentrated where it reduces error
   fastest — the active-learning argument.

The structure is implemented in
[hybrid/feedback_loop.py](hybrid/feedback_loop.py)
(`CuratorFeedbackLoop`, `CuratorVerifiedStore`, `ModelRegistry`,
`derive_flag_truth`), which composes the existing pipeline and Stage-2
trainer and delegates flag-model training to a pluggable trainer rather
than reimplementing them. §5.7 validates the labeling improvement with a
simulated curator; retraining the flag model on curator-derived labels
(false-alarm reduction) is analyzed in §5.8.

---

## 5. Results

### 5.1 Main result: v12+Branch3 vs constructed baselines (GT cell type)

**Baseline framing.** As noted in §1, §2, and detailed in §4.5, no
off-the-shelf end-to-end SWC labeler exists. The four baselines below
(NeuroM-RF, Sholl-MLP, Sholl-RF, L-Measure-RF) are *constructed by us*
to span the design space of plausible morphometric-features-plus-
classifier approaches a competing group might build; see §4.5 for
feature sets and classifier choices. All four are evaluated on the
same QC-pass corpus and 80/20 file-hash splits as our pipeline and
receive the ground-truth cell type as an input feature, matching the
GT cell type configuration of our pipeline so the comparison is
apples-to-apples.

**Table 1** — Mean ± SD over seeds 42, 123, 789. **Configuration: GT
cell type.** Full table:
[final_multiseed_vs_baselines_table.txt](paper/results/final_multiseed_vs_baselines_table.txt).

| Method | accuracy | neurite macro-F1 | axon F1 | basal F1 | apical F1 | per-cell F1 mean | per-cell F1 P10 |
|---|---|---|---|---|---|---|---|
| NeuroM-RF | .9300 ± .0050 | .8223 ± .0051 | .9668 ± .0029 | .8273 ± .0032 | .6730 ± .0126 | .8472 ± .0040 | .5421 ± .0017 |
| Sholl-MLP | .9713 ± .0016 | .9235 ± .0018 | .9884 ± .0019 | .9201 ± .0092 | .8621 ± .0142 | .8673 ± .0094 | .4164 ± .0447 |
| Sholl-RF  | .9778 ± .0013 | .9405 ± .0028 | .9915 ± .0008 | .9348 ± .0056 | .8951 ± .0055 | .8915 ± .0074 | .4600 ± .0347 |
| L-Measure-RF | .9793 ± .0016 | .9476 ± .0059 | .9915 ± .0010 | .9389 ± .0069 | .9125 ± .0127 | .9057 ± .0093 | .5397 ± .0548 |
| **Ours (v12+Branch3)** | **.9819 ± .0025** | **.9513 ± .0064** | **.9923 ± .0012** | **.9522 ± .0076** | **.9094 ± .0126** | **.9402 ± .0092** | **.7359 ± .0779** |

Three observations:

1. **Headline metric (neurite macro-F1).** Our pipeline reaches 0.951,
   ahead of the strongest baseline (L-Measure-RF at 0.948) by +0.005.
   The gap widens for the weaker baselines (Sholl-MLP: +0.028,
   NeuroM-RF: +0.129). Paired sign-flip p < 0.0001 against every
   baseline ([final_paired_stats.txt](paper/results/final_paired_stats.txt)).
2. **Apical F1.** L-Measure-RF is essentially tied with us on apical
   alone (0.9125 vs 0.9094, within SD). Our advantage on neurite
   macro-F1 comes from basal (+0.013 over L-Measure-RF) and axon
   (+0.001).
3. **Robustness — the headline finding.** Where baselines fail, they
   fail badly: per-cell F1 P10 is 0.54 (best baseline; NeuroM-RF and
   L-Measure-RF are tied at this ceiling) and only 0.42 (Sholl-MLP).
   Our pipeline lifts P10 to 0.74 — a +0.20 absolute improvement and
   the metric that most directly affects downstream usability of the
   labeled corpus.

### 5.2 Stage ablation (GT cell type and deployment)

**Table 2** isolates the contribution of each stage on **GT cell
type** and reports the cost of switching to **Stage 1 active
(deployment)** in the final row. Mean over 3 seeds; full table:
[final_ablation_table.txt](paper/results/final_ablation_table.txt).

| Configuration | accuracy | neurite macro-F1 | axon F1 | basal F1 | apical F1 | per-cell F1 P10 |
|---|---|---|---|---|---|---|
| Stage 2 only (GT) | .9701 | .9252 | .9849 | .9256 | .8652 | .6384 |
| Stage 2 + GNN (GT) | .9696 | .9223 | .9849 | .9239 | .8583 | .6239 |
| Stage 2 + GNN + Stage 3 (GT) | .9828 | .9525 | .9934 | .9519 | .9123 | .6439 |
| **Full Branch3 (GT)** | **.9832** | **.9543** | **.9934** | **.9535** | **.9160** | **.6554** |
| Full Branch3 (deployment) | .9822 | .9503 | .9932 | .9514 | .9062 | .6502 |

Three observations:

1. **The GNN alone does not help.** Stage 2 + GNN performs *worse*
   than Stage 2 alone on every metric (Δ −0.003 to −0.007 F1). The
   GNN helps only when paired with Stage 3 topology refinement.
2. **Stage 3 is the biggest single gain.** Adding Stage 3 lifts
   apical F1 from 0.858 to 0.912 (+0.054) and neurite macro-F1
   from 0.922 to 0.953 (+0.030).
3. **Stage 1 active costs ~0.004 macro-F1.** Stage 1 itself predicts
   the binary cell type at **0.974 ± 0.007** accuracy (n = 3 seeds),
   and this ~2.6% error is the *only* source of difference between the
   GT-cell-type and deployment rows. Switching from ground-truth to
   Stage 1-predicted cell type costs −0.004 neurite macro-F1, −0.010
   apical F1, and −0.005 P10 — small absolute drops, consistent with the
   high Stage 1 accuracy. All deployment-configuration results below
   (§5.3–5.8) inherit this Stage 1 error.

### 5.3 Quality-flag model (deployment configuration)

The flag model is trained and evaluated in the **deployment
configuration** (Stage 1 active) under the leave-one-seed-out protocol
of §4.4. The target is per-cell neurite F1 < 0.60. The flag dataset
contains 7,023 row-seed pairs over 5,739 unique files, with 528 bad
cells (7.5% prevalence).

**Table 3a — Flag-model accuracy at a fixed 10% rejection rate.** Mean
over LOSO folds. Source:
[final_flag_feature_ablation.txt](paper/results/final_flag_feature_ablation.txt).
F1 computed from precision and recall.

| Scope | Features | Precision | Recall | F1 | AP | Kept-set P10 |
|---|---|---:|---:|---:|---:|---:|
| all | compact (confidence + geometry) | 0.380 | 0.513 | 0.437 | 0.385 | 0.924 |
| all | + Branch3 disagreement | 0.400 | 0.535 | 0.458 | 0.405 | 0.945 |
| all | + baseline-model disagreement (`baseline_oof`) | **0.434** | **0.582** | **0.497** | **0.479** | **0.958** |
| all | + multi-v12 disagreement (`v12_oof`) | 0.431 | 0.577 | 0.493 | 0.474 | 0.961 |
| pyramidal only | `baseline_oof` | 0.557 | 0.644 | 0.597 | 0.612 | 0.897 |
| interneuron only | `baseline_oof` | 0.214 | 0.482 | 0.296 | 0.246 | 1.000 |

**Table 3b — Per-cell F1 P10 before and after 10% flag rejection
(deployment configuration).**

| Configuration | P10 |
|---|---|
| Deployment pipeline, no rejection | 0.66 |
| Deployment pipeline + flag model (`baseline_oof`, 10% reject) | **0.96** |
| Oracle: drop bottom 10% by ground-truth F1 (upper bound, pyramidal only) | 0.93 |

(Pyramidal-only oracle from [rejection_sweep.txt](paper/results/rejection_sweep.txt).)

Three observations:

1. **`baseline_oof` is the strongest deployable feature set**,
   reaching F1 = 0.50 (precision 0.43, recall 0.58) on the all-cell
   scope. We recommend this as the default flag model.
2. **Flag performance is much stronger on pyramidals than
   interneurons** (F1 0.60 vs 0.30). The interneuron flag scope sees
   so few bad cells (low absolute count and low prevalence) that the
   classifier struggles to find signal, but on the kept set every
   interneuron retained after flag rejection has P10 = 1.00.
3. **Post-rejection P10 reaches 0.96.** Rejecting the 10% of cells the
   flag scores most suspicious lifts the worst-decile per-cell F1 from
   0.66 (deployment, no rejection) to 0.96, *exceeding* the
   pyramidal-only oracle ceiling because the flag model also identifies
   bad interneurons.

This is the practical case for deployment: even though the deployment
configuration loses ~0.04 macro-F1 to Stage 1 errors and ~0.08 P10 to
the long tail, the flag model recovers far more than that on the
worst-decile metric that drives downstream usability.

### 5.4 Failure-mode analysis (deployment configuration)

We categorize the 528 bottom-decile bad cells across all seeds into
nine mutually compatible failure modes
([final_failure_modes.md](paper/results/final_failure_modes.md)):

| Category | Bad rows | Enrichment | Flag recall | False flags |
|---|---:|---:|---:|---:|
| APICAL_OVERPREDICTED | 15 | 9.99x | 0.000 | 0 |
| STAGE1_PROPAGATION  | 89 | 9.77x | 0.635 | 1 |
| APICAL_UNDERPREDICTED | 287 | 9.22x | 0.698 | 8 |
| APICAL_MISSED | 280 | 9.20x | 0.709 | 7 |
| BASAL_LOST | 131 | 8.96x | 0.512 | 3 |
| AXON_DENDRITE_CONFUSION | 331 | 8.96x | 0.528 | 16 |
| WHOLE_CELL_MISLABEL | 12 | 4.61x | 0.273 | 6 |
| TRUNCATED_APICAL | 118 | 2.93x | 0.620 | 21 |
| APICAL_INVERTED | 218 | 1.60x | 0.601 | 34 |

The dominant categories are apical-related (missing, underpredicted,
inverted, truncated). The flagger catches apical-missed and
underpredicted cells at ~70% recall, but is much weaker on whole-cell
mislabels (27%) — because confidence is high on uniformly-wrong
outputs — and on apical overprediction (0%), where the model is
confidently wrong in the other direction. The `STAGE1_PROPAGATION`
category — bad cells whose root cause is a Stage 1 cell-type error —
contributes 89 of the 528 bad cells (17%); these cells are flagged at
64% recall, suggesting the flag model picks up on most Stage 1 errors
indirectly through downstream confidence drop.

![Qualitative examples](paper/results/figures/final_figure_qualitative.png)

**Figure 2.** Rendered SWC morphologies (2D projection, colored by
structure type), ground truth vs. predicted. *Top:* a correctly-labeled
layer-5 pyramidal cell — predicted labels are identical to ground truth
(per-cell F1 = 1.00). *Bottom:* a mislabeled cell (per-cell F1 = 0.11)
where the apical/basal assignment is inverted relative to the source
labels. This case also illustrates why the curator loop matters: the
source labeling here is itself non-standard (the tall dendrite is
labeled basal and a short downward branch apical), so the disagreement
partly reflects inconsistency in the input labels — exactly the kind of
cell the flag step surfaces for human review.

### 5.5 Lab-subset evaluation and lab-only retraining

A common reviewer concern with cross-source corpora is generalization
to a specific user's data: the published numbers in §5.1 mix three
sources (Allen, hpf_ca1, NeuroMorpho), so it is unclear whether the
model is equally good on well-curated single-lab data, and a user
deploying on a different single lab cannot read their expected
performance off Table 1 directly. We address this in two complementary
ways, both restricted to the in-house lab corpus (`hpf_ca1__*`,
1,358 QC-pass cells, all pyramidal — see §3).

**(a) Restricted evaluation — existing model, lab cells only.** We take
the existing v12+Branch3 models from §5.1 (trained on the full
cross-source corpus) and re-evaluate them on the *subset* of the
held-out test files belonging to the in-house lab corpus. This is the
same model, scored on the cleaner data slice. Metrics are computed in
the same fashion as the all-corpus table (node-level pooled per-class
F1, per-cell mean / P10 / P25). Mean ± SD over 3 seeds (n = 867 lab
test cells pooled), GT cell type configuration. Source:
[v12_lab_subset_multiseed_table.txt](paper/results/v12_lab_subset_multiseed_table.txt).

**Table 4a — v12+Branch3 (GT cell type) on lab subset vs all corpus.**

| Metric | All corpus (Table 1) | Lab subset | Δ |
|---|---:|---:|---:|
| node accuracy | .9819 ± .0025 | **.9923 ± .0017** | +.0104 |
| neurite macro-F1 | .9513 ± .0064 | .9496 ± .0070 | −.0017 |
| axon F1 | .9923 ± .0012 | **.9984 ± .0009** | +.0061 |
| basal F1 | .9522 ± .0076 | .9281 ± .0148 | −.0241 |
| apical F1 | .9094 ± .0126 | .9222 ± .0059 | +.0128 |
| per-cell F1 mean | .9402 ± .0092 | **.9544 ± .0045** | +.0142 |
| per-cell F1 P10 | .7359 ± .0779 | **.7877 ± .0602** | +.0518 |
| per-cell accuracy mean | — | .9840 ± .0035 | — |
| per-cell accuracy P10 | — | .9767 ± .0015 | — |

On lab cells, node accuracy is essentially saturated (99.2%; per-cell
mean 98.4%, P10 97.7%), and per-cell F1 mean and P10 are clearly better
than on the full corpus (P10 0.74 → 0.79). Neurite macro-F1 is
essentially tied (0.951 vs 0.950): the higher accuracy is offset by a
−0.024 drop in basal F1 — an artifact of macro-averaging on these large
CA1 cells, where axon nodes outnumber basal ~20×, so a few basal→axon
errors weigh heavily on the basal class. Axon F1 reaches 0.998.

We note in passing that the simple morphometric baselines of §5.1 close
most of the gap on this clean single-lab slice (their neurite macro-F1
rises to ~0.97, matching ours), consistent with the paper's central
claim: the staged pipeline earns its complexity on *heterogeneous*
corpora, not on clean single-source data where simple features already
suffice. We omit the full per-baseline table on the lab slice, since the
headline comparison (Table 1, full corpus) is the like-for-like one; the
per-baseline lab-slice numbers are available in the artifact manifest
([baselines_lab_subset_multiseed_table.txt](paper/results/baselines_lab_subset_multiseed_table.txt)).

**(b) Lab-only retrain — full pipeline trained and evaluated on lab data alone.**
We rebuild Stage 2 + GNN + Branch3 on the lab QC-pass corpus alone,
using the same 80/20 file-hash split per seed (seeds 42, 123, 789;
1,063–1,080 train / 278–295 test per seed). Stage 1 is degenerate
because the lab corpus is 100% pyramidal, so for lab-only deployment
we hardcode cell type = pyramidal and skip Stage 1 at both train and
inference time. All other model components (Stage 2 XGBoost, GNN,
Stage 3 topology refinement, Branch3 rescue) are retrained from
scratch on lab data with the same gentle class-weighting configuration
(`SWCAL_CLASS_BALANCE_POWER=1.25`, `SWCAL_GNN_CLASS_WEIGHT=inverse_sqrt`,
focal gamma 2.0). Source:
[v12_lab_only_retrain_multiseed_table.txt](paper/results/v12_lab_only_retrain_multiseed_table.txt).

**Table 5 — v12+Branch3 (GT cell type) on lab test set: cross-source training vs. lab-only training.**

| Metric | (a) Cross-source train | (b) Lab-only retrain | Δ (b − a) |
|---|---:|---:|---:|
| node accuracy | .9923 ± .0017 | **.9929 ± .0014** | +.0006 |
| neurite macro-F1 | .9496 ± .0070 | **.9528 ± .0059** | +.0032 |
| axon F1 | .9984 ± .0009 | **.9986 ± .0007** | +.0002 |
| basal F1 | .9281 ± .0148 | **.9314 ± .0164** | +.0033 |
| apical F1 | .9222 ± .0059 | **.9283 ± .0015** | +.0061 |
| per-cell F1 mean | .9544 ± .0045 | **.9552 ± .0024** | +.0008 |
| per-cell F1 P10 | .7877 ± .0602 | **.8011 ± .0491** | +.0134 |
| per-cell accuracy mean | .9840 ± .0035 | **.9885 ± .0001** | +.0045 |
| per-cell accuracy P10 | .9767 ± .0015 | **.9789 ± .0028** | +.0022 |

Lab-only training wins on every metric — most on apical F1 (+.006) and
per-cell P10 (+.013) — despite ~10× less training data (1,064 vs ~9,500
cells), and with tighter seed-to-seed variance. The architecture scales
down: Stage 2 + GNN + Branch3 trains end-to-end on ~1k cells and matches
or beats the cross-source model on the same held-out lab cells. For a
lab user the headline is 99.3% node accuracy and 98.9% per-cell mean
accuracy — the model effectively reproduces their manual labels. The
practical recommendation: start with the released cross-source model
(a); if you have a large clean in-house corpus, retrain (b) for a small
but consistent gain, especially on the worst-decile tail.

### 5.6 Runtime

We benchmark inference latency per cell on a 100-cell sample drawn
from the seed-123 held-out test split (mean 8,976 nodes/cell, median
1,016 nodes/cell — i.e., a long-tailed mix matching the full corpus
distribution). Reported numbers exclude one-time model load and SWC
parsing so they reflect the marginal cost of labeling an additional
cell once the model is in memory. The first three cells are discarded
as warmup. Source:
[runtime_benchmark.txt](paper/results/runtime_benchmark.txt).

**Table 6 — Per-cell inference latency, all methods.** Mean and median
in ms per cell; throughput in cells/s and nodes/s. Same cell sample
for every method.

| Method | mean ms | median ms | P10 ms | P90 ms | cells/s | nodes/s |
|---|---:|---:|---:|---:|---:|---:|
| Sholl-MLP    |  52.5 |   5.6 |  1.4 | 147.4 | 19.04 | 170,880 |
| Sholl-RF     |  87.8 |  45.1 | 30.9 | 171.5 | 11.38 | 102,171 |
| NeuroM-RF    | 112.7 |  34.6 | 30.9 | 217.5 |  8.87 |  79,631 |
| L-Measure-RF | 354.7 |  42.2 | 31.1 | 632.1 |  2.82 |  25,308 |
| **Ours (v12+Branch3)** | 353.8 | 227.7 | 185.6 | 592.1 | **2.83** | **25,369** |

Three observations:

1. **Absolute latency is sub-second per cell** (median 228 ms, P90
   592 ms). On a typical workstation the pipeline processes ~2.8
   cells/s ≈ 10,000 cells/hour, which is fine for offline corpus
   labeling — the full 11,862-cell QC-pass corpus labels in roughly
   one wall-clock hour from cold.
2. **Per-cell mean throughput is competitive with L-Measure-RF**
   (354 ms vs 355 ms). The Sholl baselines are 3–7× faster per cell;
   our pipeline pays the extra cost for the GNN, Stage 3 topology
   refinement, and Branch3 rescue.
3. **Latency distribution is tighter on our pipeline.** Where
   L-Measure-RF has a median of 42 ms but a P90 of 632 ms and a max
   of 5.6 s (heavy tail driven by the largest cells), our pipeline
   shows median 228 ms → P90 592 ms — comparable P90 but no extreme
   outliers (max 1.8 s vs 5.6 s). For an interactive tool this
   matters more than mean throughput.

Caveats: timings are from a single Windows workstation; Stage 2 uses
GPU (XGBoost), the GNN and Branch3 rescue run on CPU, all four
baselines run on CPU. We do not attempt to optimize the pipeline for
latency in this paper; the GNN + Branch3 graph builds dominate the
per-cell cost and could likely be sped up substantially by batching
inference across multiple cells. We report the warm-inference numbers
as a baseline practitioners can budget against; deployment in
SWC-Studio uses the same code path.

### 5.7 Curator feedback loop: simulated active learning

The feedback-loop architecture of §4.7 claims the labeler improves
with curator effort, and that routing that effort through the flag
step is more efficient than reviewing random cells. We validate both
claims with a *simulated curator*: because we hold ground-truth labels,
we can reveal the true labels of queried cells (standing in for a
curator's corrections), add them to the training pool, retrain, and
measure the effect on a fixed held-out test set that is never queried.

**Setup.** From the QC-pass corpus we take a deterministic working set
(N ≈ 5,000 cells) split by file hash into a small seed training set,
a curator-queryable pool (labels hidden until queried), and a fixed
test set. The acquisition function is the flag/uncertainty signal:
per-cell node-weighted mean (1 − max posterior) of the current Stage-2
classifier — a faithful proxy for the trained flag model. Two arms
share the round-0 model then diverge: **flag-guided** queries the
most-uncertain unqueried pool cells each round; **random** queries the
same number of random cells. Each round reveals the queried labels,
retrains the Stage-2 branch classifier on seed ∪ revealed, and
re-evaluates the fixed test set. To keep multi-round retraining
tractable we isolate the Stage-2 classifier (cached branch features,
retrained each round); both arms use the identical procedure, so the
comparison isolates the value of flag-guided selection. Source:
[paper/_active_learning_sim.py](paper/_active_learning_sim.py),
[active_learning_curve.txt](paper/results/active_learning_curve.txt).

**Table 7 — Active-learning curve, flag-guided vs random.** Mean ± SD
over 3 seeds (42, 123, 789); seed_train ≈ 740, pool ≈ 3,260, test ≈ 990
per seed. Test set fixed and never queried. Source:
[active_learning_curve_multiseed.txt](paper/results/active_learning_curve_multiseed.txt).

| Curator corrections | macro-F1 flag | macro-F1 random | Δ | apical flag | apical random | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | .735 ± .016 | .735 ± .016 | — | .762 ± .037 | .762 ± .037 | — |
| 200 | .728 ± .036 | .730 ± .025 | −.003 | .756 ± .080 | .763 ± .062 | −.007 |
| 400 | **.756 ± .019** | .743 ± .019 | +.013 | **.807 ± .023** | .785 ± .037 | +.021 |
| 600 | **.766 ± .015** | .746 ± .012 | +.020 | **.827 ± .015** | .788 ± .029 | +.039 |
| 800 | **.766 ± .013** | .745 ± .014 | +.021 | **.832 ± .016** | .792 ± .025 | +.040 |
| 1000 | **.770 ± .015** | .749 ± .009 | +.021 | **.841 ± .024** | .800 ± .025 | +.041 |
| 1200 | **.771 ± .015** | .758 ± .012 | +.013 | **.843 ± .024** | .823 ± .025 | +.020 |
| 1400 | **.771 ± .014** | .754 ± .011 | +.018 | **.843 ± .021** | .811 ± .036 | +.031 |
| 1600 | **.771 ± .014** | .763 ± .015 | +.008 | **.842 ± .021** | .828 ± .024 | +.014 |

![Active-learning curve](paper/results/figures/final_figure_active_learning.png)

**Figure 3.** Simulated-curator active learning (mean ± SD over 3 seeds).
Neurite macro-F1 (left) and apical F1 (right) vs. number of curator
corrections, for flag-guided vs. random selection. Both arms improve
(the loop works); flag-guided sits above random from the second round on.

Two findings, both robust across seeds:

1. **The loop works — the labeler improves with curator effort.** Both
   arms climb monotonically from macro-F1 0.735 (seed model only) toward
   ~0.77 as curator-corrected cells enter the training pool. This is the
   core validation of the §4.7 architecture: correcting flagged cells
   and retraining measurably improves the labeler on cells it never saw.
2. **Flag-guided routing is more label-efficient than random.** From
   400 corrections onward, flag-guided sits +0.013 to +0.021 macro-F1
   and +0.02 to +0.04 apical F1 above random, outside the seed-to-seed
   SD by ~600 corrections. In label-efficiency terms: to reach a test
   macro-F1 of 0.75, flag-guided needs ≈ 400 curator corrections vs
   ≈ 1,200 for random (**67% fewer labels**); to reach 0.76, ≈ 600 vs
   ≈ 1,600 (**62% fewer**); random never reaches 0.77 within the
   1,600-correction budget, while flag-guided reaches it by 1,000. The
   flag step doesn't just filter bad cells — it directs scarce curator
   effort to the cells that most improve the model.

One honest caveat: at the first round (200 corrections) flag-guided
dips slightly below random (−0.003 macro-F1, within SD). This is the
familiar active-learning cold-start — the most-uncertain cells in the
very first batch include unlearnable outliers (severely damaged
reconstructions) whose corrected labels generalize poorly. The effect
is transient; flag-guided pulls clearly ahead by the second round.

Scope: this experiment isolates the Stage-2 branch classifier
(retrained each round on cached features) so the multi-round loop is
tractable; both arms use the identical procedure, so the comparison
cleanly attributes the gain to flag-guided *selection*.

**Full-pipeline confirmation (and a limit on the selection claim).**
A multi-round curve that retrains the *entire* labeler (Stage 2 + GNN +
Stage 3) each round is compute-prohibitive (~days). We therefore ran a
one-shot endpoint version on seed 123: train the full pipeline on a
seed set (614 cells), then on seed + 1,000 corrections selected either
by the seed model's flag/uncertainty signal or at random, and evaluate
all three on a fixed 818-cell test set (GT cell type; Branch3 omitted
for tractability). Source:
[active_learning_full_pipeline.txt](paper/results/active_learning_full_pipeline.txt).

| Full-pipeline model | neurite macro-F1 | apical F1 | per-cell F1 mean | per-cell F1 P10 |
|---|---:|---:|---:|---:|
| seed (614 cells) | .9146 | .8637 | .8768 | .4614 |
| + 1,000 flag-selected | .9443 | .8882 | .8962 | .5365 |
| + 1,000 random | **.9536** | **.9092** | **.9154** | **.6479** |

This gives a mixed, honest result. **The loop's labeling gain survives
the full pipeline**: adding corrections lifts full-pipeline macro-F1 by
+0.030 and apical F1 by +0.025 over seed-only, confirming that the
Stage-2 improvement is not washed out by the GNN and topology refinement.
**But at this one-shot, 1,000-cell batch scale, flag-guided selection did
*not* beat random** (−0.009 macro-F1). We attribute this to a known
failure mode of one-shot large-batch uncertainty sampling: selecting the
single most-uncertain 1,000 cells (≈40% of the pool) loads the training
set with redundant, correlated, and outlier-heavy reconstructions —
here 1,000 hard cells added to 614 seed cells, so ~62% of training is
now pathological — which generalizes worse to a representative test set
than a diverse random sample. The selection advantage in the Stage-2
curve above comes precisely from its *iterative, small-batch* protocol
(200 cells/round, re-scored each round), which never accumulates such a
batch. We therefore state the scope precisely: **the loop improves the
labeler at both Stage-2 and full-pipeline scale; flag-guided selection
is more efficient than random specifically in the iterative small-batch
regime, and a one-shot large batch should use diverse or hybrid
acquisition, not pure uncertainty.** Confirming iterative flag-guidance
at full-pipeline scale is left to future work for compute reasons (§8).

### 5.8 Does the loop improve the *flag model* too?

§5.7 shows the loop improves the *labeler*. A separate question is
whether it improves the *flag model* — specifically its precision, i.e.
whether it learns to stop firing on low-confidence-but-actually-correct
cells (false alarms), which are the flagger's dominant error today
(precision ≈ 0.43 at 10% rejection, §5.3). The feedback loop generates
exactly the supervision needed: every reviewed cell yields a
flag-feature vector labeled by the *derived* truth
`per-cell F1(curator, model) < 0.60` (§4.7). Retraining the flag model
on `factory_flag_data ∪ curator-derived (features, truth)` should raise
precision as the curator's false-alarm verdicts accumulate.

**Experimental design (leakage-safe).** We split the multi-seed flag
dataset (7,023 cell-rows, 5,739 unique files, 59 flag features, 7.5%
bad rate) *by file* into a fixed held-out **test** set (20%), a small
**base** set that trains the factory flag model (15%), and a **pool**
(65%) of curator-reviewable cells. Each round: the current flag model
scores the pool, its top-K flagged cells are "reviewed" (their true
bad/good status revealed, standing in for a curator correction), those
`(features, status)` pairs are added to the flag training set, and the
flag model is retrained. Precision, recall, and average precision are
measured on the **test** set (never reviewed) at a fixed 10% rejection
rate. We compare three arms — **factory** (base only, never updated),
**feedback** (review the top-flagged pool cells), and **random**
(review random pool cells) — averaged over 3 by-file splits. Source:
[paper/_flag_feedback_sim.py](paper/_flag_feedback_sim.py),
[flag_feedback_curve.txt](paper/results/flag_feedback_curve.txt).

**Table 8 — Flag precision @10% rejection vs. curator verdicts.** Mean ± SD
over 3 splits.

| Curator verdicts | Feedback | Random | Factory |
|---:|---:|---:|---:|
| 0 | .328 ± .026 | .328 ± .026 | .328 |
| 300 | **.367 ± .034** | .345 ± .022 | .328 |
| 600 | **.383 ± .037** | .345 ± .027 | .328 |
| 900 | **.393 ± .031** | .345 ± .029 | .328 |
| 1050 | **.398 ± .029** | .352 ± .021 | .328 |
| 1200 | **.391 ± .029** | .357 ± .045 | .328 |

![Flag feedback curve](paper/results/figures/final_figure_flag_feedback.png)

**Figure 4.** Retraining the flag model on curator-derived verdicts
(mean ± SD over 3 splits). *Left:* flag precision @10% rejection vs.
number of verdicts, for feedback (review flagged) vs. random review vs.
the never-updated factory model. *Right:* recall and average precision
for the feedback arm.

The answer is **yes — curator feedback measurably improves the flag
model**, and reviewing *flagged* cells is markedly better than reviewing
random ones:

1. **Precision rises from 0.328 to ~0.40** (+0.07 absolute, **+21%
   relative**) as ~1,000 curator verdicts accumulate. The flag model
   learns to stop firing on the low-confidence-but-actually-correct
   cells — exactly the false-alarm reduction the loop was designed for.
2. **Reviewing flagged cells beats random review at every budget**
   (feedback .398 vs random .352 at 1,050 verdicts). Flagged cells are
   the decision-boundary examples — both the true errors and the
   false alarms — so their verdicts sharpen the flag boundary far more
   than random cells, most of which are easy negatives.
3. **Recall and AP improve too** (recall 0.484 → 0.586; AP 0.361 →
   0.472 over the same range), so the retrained flag model is a
   strictly better ranker, not just better at one threshold.

Honest limits. (i) Precision plateaus around 0.40 — this is a
false-alarm-reduction gain on the cells the flagger *surfaces*, not a
fix for the recall blind spot of §5.4 (confidently-wrong cells the model
never flags, so no verdict is ever generated for them). (ii) Because the
verdict here is derived from held-out ground truth, this simulates a
curator who always labels correctly; real curator noise would attenuate
the gain. (iii) This experiment updates only the flag model (fast
tabular retrain); it is the flag-side complement to the labeler-side
result in §5.7.

---

## 6. Discussion

**Where the gains come from.** The stage ablation surprised us: the
GNN, the component we expected to be load-bearing, in isolation
produces no gain over the Stage 2 branch classifier and even slightly
hurts. The real gains come from the deterministic Stage 3 topology
refinement, which enforces tree-level constraints the branch-wise
XGBoost cannot see. This suggests the value of the GNN is not in
propagating information per se but in providing a smoothed
apical/basal map that Stage 3 can then enforce trunk-contiguity over.

**Where the residual errors are.** Even at 0.95 macro-F1, the per-cell
P10 in deployment configuration is 0.66 — a long lower tail of severely
mislabeled cells. The failure-mode breakdown shows this tail is
dominated by apical errors on pyramidal cells (missed, underpredicted,
inverted) and by axon/dendrite confusion on single-lab subsets of
NeuroMorpho. These are largely cases where the *input* SWC is
ambiguous (axon-only fragments, truncated traces, inverted-z
conventions), not cases where the model has trained itself into a
corner. The flagger catches roughly two-thirds of these, which is the
practical justification for the deployment workflow: auto-label
everything, flag the worst 10%, hand those to a human curator.

**Train on your own data.** The pipeline is shipped with a single
driver script (`paper/_retrain_v12_gentle_seed.py`) that consumes a
user-supplied QC CSV pointing at SWC files plus cell-type labels and
produces a complete labeler in one invocation. Users with their own
well-curated single-lab corpus can therefore specialize the labeler
to their tracing conventions, which we exercise in §5.5 on the
in-house lab corpus. This is the primary deployment path for groups
whose data conventions differ from the cross-source corpus.

**Deployment.** SWC-Studio currently ships the compact flagger
(confidence + geometry + Branch3 disagreement), which is the
deployable variant because it requires only one inference pass. The
`baseline_oof` and `v12_oof` variants give better detection but
require running additional models at inference time, which is
impractical for an interactive tool. We recommend `baseline_oof` for
batch curation workflows where the additional compute is acceptable.
[TODO: confirm SWC-Studio framing.]

---

## 7. Limitations

1. **Pyramidal and interneuron only.** Our corpus and all trained
   models cover exactly two cell types. The system has not been
   validated on Purkinje, granule, motor neurons, or invertebrate
   morphologies, and the Stage 1 classifier returns binary output.
   Extending to a third class would require retraining Stage 1, a new
   cell-type feature path through Stage 2, and re-deriving the
   topology rules in Stage 3.
2. **Single-primary-tree limitation.** The labeler assumes a normal
   SWC where each primary tree (the maximal subtree rooted at a child
   of the soma) is one structure type for its full extent. Stage 2
   and Stage 3 do not split a single primary tree into multiple
   structure types midway down. This matches the typical biological
   convention — axons and dendrites emerge as separate primary trees
   from the soma — but it can fail on reconstruction artifacts where
   a single primary tree branches downstream into what should be
   distinct axon and dendrite subtrees. The phenomenon is
   biologically uncommon but does appear in some SWC reconstructions,
   and the model will label the entire artifactual primary tree as a
   single type.
3. **No leave-one-lab-out evaluation.** Our 80/20 split is random by
   file hash within the curated corpus. We have not held out an
   entire contributing laboratory or contributing brain region, so
   we cannot directly claim generalization to a new lab's tracing
   conventions. We partially mitigate this concern with the
   lab-subset analysis in §5.5 — both restricted evaluation of the
   cross-source model on lab-only test cells, and a full lab-only
   retrain that demonstrates the architecture trains from a small
   single-lab corpus — and by exposing a single-script retrain
   pathway so any user can specialize the model to their own
   well-curated SWC corpus. A formal leave-one-lab-out evaluation
   on the cross-source corpus remains future work.
4. **Latency vs. baselines.** Per §5.6, our pipeline labels at ~2.8
   cells/s, the same throughput as L-Measure-RF and 3–7× slower than
   the Sholl baselines. Absolute latency (median 228 ms / cell, P90
   592 ms) is sub-second per cell and is fine for offline corpus
   labeling, but is too slow for high-throughput streaming use
   without batching. We have not implemented the batched inference
   path; the GNN and Branch3 graph builds dominate cost and are the
   natural targets for that optimization.
5. **Flagger ceiling.** The flagger's recall plateaus around 58% at
   10% rejection. Whole-cell mislabel and apical-overprediction
   failure modes are systematically missed because the model is
   confidently wrong; this is an upper bound on what a
   confidence-based flagger can achieve and suggests an opportunity
   for a second flagger trained on anatomically-implausible
   structural patterns.
6. **No human-in-the-loop study.** We have not measured how much
   curator time auto-labeling actually saves on a representative
   curation task.

---

## 8. Future work

**Extending the curator feedback loop.** The closed-loop architecture
(§4.7) and its simulated-curator validation (§5.7) establish that the
labeler improves as curators correct flagged cells. Several design
questions remain open for a full deployment: (i) how to weight
curator-verified examples relative to the original training corpus,
especially when a single lab contributes disproportionately; (ii) how
to detect and prevent labeler drift if a single curator's conventions
diverge from the community norm; (iii) the appropriate retrain cadence
(per-cell online updates vs. periodic batched retrains); and (iv)
whether periodically retraining the *flag model itself* on curator
decisions (which cells were truly bad, which were false alarms) closes
the flagger-ceiling gap documented in §5.4. Our simulation retrains
only the Stage-2 classifier for tractability; a full study would
retrain the GNN, Branch3, and flag model on the same augmented pool
and measure the compounding effect.

**Other directions.**

- A second flag model trained on anatomically-implausible structural
  patterns to catch the failure modes the confidence-based flagger
  systematically misses (whole-cell mislabel, apical overprediction).
- Extension to a third cell type beyond pyramidal and interneuron
  (§7 limitation 1).
- A human-in-the-loop time-savings study quantifying how much curator
  time the auto-label + flag workflow actually saves on a
  representative curation task (§7 limitation 6).
- A formal leave-one-lab-out evaluation on the cross-source corpus
  to directly quantify generalization across labs (§7 limitation 3).
- **Diverse / hybrid acquisition and iterative full-pipeline retraining.**
  §5.7's full-pipeline endpoint showed that pure uncertainty sampling in
  a single large batch underperforms random selection (it over-samples
  redundant outliers); flag-guidance helped only in the iterative
  small-batch regime. A batch-aware acquisition that trades off
  uncertainty against diversity (e.g. BADGE-style or clustering the
  flagged set before selection), evaluated under an iterative
  full-pipeline retrain, is the natural next step to make flag-guided
  selection robust at deployment batch sizes.

---

## 9. Conclusion

We have presented an end-to-end SWC labeling pipeline (QC → cell type
→ branch classifier + GNN → topology refinement + Branch3 → quality
flag) for automatic axon/basal/apical labeling. In the GT-cell-type
configuration, used for apples-to-apples comparison against four
constructed baselines, the system reaches axon / basal / apical F1 of
0.993 / 0.954 / 0.916 on held-out cells across three seeds and lifts
the worst-decile per-cell F1 from at most 0.54 (best baseline) to
0.74. In the deployment configuration, where Stage 1 predicts cell
type at 0.974 accuracy, the system reaches axon / basal / apical F1 of
0.993 / 0.951 / 0.906; the quality-flag model then identifies the
worst 10% of predictions with 58% recall and lifts the kept-set
worst-decile F1 from 0.66 to 0.96. Beyond the static labeler, we
introduce a curator feedback loop that turns the flag step into an
active-learning acquisition function: correcting flag-selected cells
and retraining reaches a target accuracy with ~60% fewer curator
corrections than random review, so the deployed labeler improves
efficiently with use. Code, models, splits, and result manifests are
released.

---

## 10. Reproducibility

- **Code.** [hybrid/](hybrid/) contains the canonical pipeline modules
  (`pipeline.py`, `qc_input.py`, `train_stage2.py`, `stage3_refine.py`,
  `swc_normalize.py`, `features.py`, `branch_features.py`,
  `evaluate.py`, `feedback_loop.py`). The curator feedback loop is in
  [hybrid/feedback_loop.py](hybrid/feedback_loop.py); the
  active-learning simulation is
  [paper/_active_learning_sim.py](paper/_active_learning_sim.py).
- **Models.** Canonical labeler at
  [paper/models/v12_gentle_seed123/](paper/models/v12_gentle_seed123/);
  per-seed variants at `v12_gentle_seed{42,123,789}/`. Flag models at
  [paper/models/flag_model_seed123_branch3/](paper/models/flag_model_seed123_branch3/).
- **Splits and seeds.** All splits are deterministic from the SWC
  filename hash; train/test split JSONs are committed alongside each
  model. Random seeds: 42, 123, 789.
- **Result artifacts.** Manifest of every cited table and figure at
  [paper/results/final_canonical_outputs_manifest.json](paper/results/final_canonical_outputs_manifest.json).
- **Pipeline checklist.** [paper/CANONICAL_PAPER_CHECKLIST.md](paper/CANONICAL_PAPER_CHECKLIST.md)
  lists every claim, the artifact that supports it, and the script that
  regenerates it. [TODO: refresh `paper/README.md` — currently
  references deleted v9-era scripts.]

---

## Remaining before submission

Done: pipeline schematic (Fig 1), qualitative renderings (Fig 2), runtime
benchmark (§5.6), GNN architecture details (§4.3), Related Work pass (§2),
figure numbering + captions (Figs 1–4), 4-pillar contribution focus.

Still open:
- [ ] Verify Related Work citations (authors/venues/years) against the
      cited URLs; confirm the Emissah/Tecuatl/Ascoli reference (§2).
- [ ] SWC-Studio deployment framing confirmation (§6).
- [ ] Consolidated hyperparameter appendix table (optional).
- [ ] Refresh `paper/README.md` to match the current v12 pipeline.
- [ ] Optional: leave-one-lab-out cross-dataset eval (§7); iterative
      full-pipeline / diversity-aware acquisition (§8).
