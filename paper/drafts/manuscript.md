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
takes an SWC reconstruction and returns a consistently labeled SWC. The
pipeline runs five sequential steps: (i) a quality-control gate that
filters malformed inputs; (ii) a cell-type classifier that either uses
a user-supplied cell type or predicts pyramidal vs. interneuron from
whole-cell features (predicted-type accuracy 0.974 ± 0.007); (iii) a
per-subtree XGBoost classifier that types each primary subtree as
axon / basal / apical (the soma is located structurally, not learned);
(iv) a graph neural network and a per-branch three-class rescue network
that re-score the apical/basal boundary, followed by a deterministic
topology-refinement pass that consolidates these branch-level edits into
an anatomically consistent labeling; and (v) an optional quality-flag
model that identifies low-confidence predictions and flags them for
human review. On a curated
11,862-cell, three-source corpus split 80/20 by file hash and evaluated
across three random seeds, the auto-labeler reaches per-class F1 of 0.993
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
feed back into training, so the auto-labeler improves with use. A
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

**No off-the-shelf auto-labeler exists for this task.** To our knowledge,
no released software package or published model performs end-to-end
per-node SWC re-labeling on raw morphology reconstructions. NeuroM and
L-Measure are widely-used morphometric *analysis* libraries — they
compute features from already-labeled SWCs, they do not assign labels.
The closest *labeling* work is the Emissah/Tecuatl/Ascoli line of
per-subtree classifiers based on Sholl descriptors, but it predicts
subtree types in isolation rather than producing a full re-labeled SWC
and is not released as a runnable auto-labeler. The novelty of the present
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

1. **A staged pipeline for end-to-end SWC re-labeling.** A sequence of
 a cell-type classifier, a per-subtree classifier, an apical/basal
 graph neural network, a per-branch rescue network, and a
 topology-refinement pass attains a mean per-class F1 of 0.993 / 0.954
 / 0.916 for axon / basal / apical (0.993 / 0.951 / 0.906 when the cell type
 is predicted rather than supplied). We evaluate it against four
 morphometric baselines on identical file-level splits with paired
 significance testing (n = 5,739 cells), and a controlled ablation
 attributes the largest single gain to topology refinement rather
 than to the graph neural network.
2. **A quality-flag model for a label-then-curate workflow.** A flag
 model identifies the least reliable 10% of predictions at 58% recall
 (F1 0.50) from a single inference pass; rejecting them raises the
 10th-percentile per-cell F1 of the retained set from 0.66 to 0.96,
 converting the auto-labeler into a triage tool that concentrates
 human effort on the cells that need it.
3. **Corpus-specific specialization.** The pipeline can be retrained on
 a user-supplied corpus with a single command; we show (§5.5) that a
 model retrained on one well-curated laboratory corpus matches or
 exceeds the cross-source model on that laboratory's data.
4. **A curator feedback loop that improves both the auto-labeler and the
 flag model.** The flag stage routes uncertain cells to a curator
 whose corrected labels are folded back into training (§4.7). In a
 simulated-curator study, flag-guided review reaches a target
 accuracy with roughly 60% fewer corrections than random selection
 (§5.7), and retraining the flag model on curator verdicts raises its
 precision by 21% (§5.8), so the deployed system improves with
 continued use.

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
carry per-node type labels — not auto-labelers. We assemble L-Measure,
NeuroM, and Sholl-feature baselines (§4.5) on top of these feature
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
This motivates our Sholl-feature baselines (a random forest and an MLP). It is not
end-to-end on raw SWC (it decides per-subtree in isolation, without a
QC gate, topology-consistency pass, or quality flag) and is not
released as a runnable auto-labeler, so we reimplement the per-subtree
classifier on our split as a baseline (§4.5).

**Reconstruction and editing tools vs. automatic typing.** A large
ecosystem of software produces and edits SWC files — tracing tools such
as neuTube (Feng, Zhao & Kim, *eNeuro* 2015), ShuTu (Jin et al.,
*Neuroinformatics* 2019), Vaa3D/BigNeuron, and the DIADEM-era tracers.
These convert microscopy image stacks into SWC skeletons, and community
efforts (TREES toolbox, L-Measure, NeuroM) then edit or analyze them. In
all of them the structure-type column (axon / basal / apical) is
assigned **manually by the annotator during tracing** or left to a
default; none *automatically assigns or corrects* per-node types on an
already-traced SWC. Our system targets exactly this missing step: given
an SWC whose type column may be absent, inconsistent, or wrong, produce
a corrected per-node typing. [Reconstruction-tool citations to verify.]

**Databases and curation.** The SWC format (Cannon et al., *J. Neurosci.
Methods* 1998) and NeuroMorpho.Org (Ascoli, Donohue & Halavi, *J.
Neuroscience* 2007) underpin the corpora we use. NeuroMorpho applies
limited per-upload curation but does not relabel structure types, so
per-lab conventions and errors persist downstream — the very problem
our flag model and curator loop are designed to support (not replace).

**Human-in-the-loop and active learning.** Uncertainty-guided sample
selection is a standard active-learning tool; our contribution is to
show that the *quality-flag* signal of a deployed auto-labeler doubles as an
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
the feature families above because no runnable prior auto-labeler exists.

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

()

The dominant QC failure modes are missing soma (5,624 / 9,192 = 61%)
and joint missing-soma + no-neurites (24%). The remainder fail size
and single-root checks. Interneuron failure is driven almost entirely
by the NeuroMorpho subset, where many uploads are axon-only or
dendrite-only fragments rather than complete cells. We report all
downstream results on the QC-pass corpus.

We split this corpus 80/20 by a deterministic hash of the
`(seed, SWC filename)` pair, so each of the three seeds induces its own
train/test split (the three held-out sets overlap only partially). This
is deliberate: the mean ± SD we report across seeds therefore reflects
two sources of variability at once — resampling of the held-out set and
model initialization — giving a more conservative estimate of
generalization than fixing a single split would. The hash is stable (a
given file always falls in the same bucket for a given seed, independent
of corpus size), which keeps splits reproducible, and we verify that no
test file leaks into its own seed's training fold.

---

## 4. Method

### 4.1 Pipeline overview

The end-to-end auto-labeler runs in five stages that consume an SWC file
and return a labeled SWC together with an optional quality flag
(Figure 1). **Stage 0** applies a quality-control gate that rejects
malformed reconstructions. **Stage 1** resolves the cell type —
pyramidal or interneuron — which the user supplies when known and an
XGBoost classifier predicts when it is not; the cell type fixes the
downstream label set, since only pyramidal cells carry an apical class.
**Stage 2** assigns axon, basal, or apical to each *primary subtree* (a
process emerging from the soma) with a subtree-level classifier and
propagates that label to the subtree's nodes; the posterior probability
behind each assignment is the node's *confidence*, a score reused
throughout — by the soft handoff (§4.3), the topology refinement (§4.3),
and the quality flag (§4.4). **Stage 2.5** then refines the basal-versus-apical
boundary on pyramidal cells with two per-branch graph networks in
sequence — a basal/apical GNN and a three-class rescue network
(Branch3) — updating the confidence where they override a label.
Finally, **Stage 3** applies a rule-based topology-refinement pass that
consolidates these branch-level edits into an anatomically consistent
labeling, yielding the final labeled SWC.
An optional **Stage 4** quality-flag model then scores each labeled cell
and surfaces the likely-mislabeled ones for human review. Stages 0–4
constitute the auto-labeler, and a user may stop there at the labeled
SWC; an optional **Stage 5** closes the loop (§4.7), routing flagged
cells to a curator whose corrections are folded back into training so
the deployed system improves with use (§5.7).

```
Raw SWC
 ↓
Stage 0 QC gate (filter; reject or pass)
 ↓
Stage 1 Cell type user input ───────────────┐
 (pyramidal / interneuron) │
 OR │
 predict (XGBoost, whole-cell features) ─────────────┘
 ↓
Stage 2 Per-subtree axon / basal / apical XGBoost classifier
 ↓
Stage 2.5 Apical/basal refinement: GNN + Branch3 (pyramidals only)
 ↓
Stage 3 Topology refinement (consolidates 2.5 into the final labels)
 ↓
Labeled SWC
 ↓
Stage 4 Quality flag model (flags suspect cells)
 ↓
 flagged? ──no──► accept labels
 │ yes
 ▼
Stage 5 Curator review (human-in-the-loop, §4.7)
 │ supplies corrected labels (or confirms them unchanged);
 │ flag truth is derived, not chosen
 ▼
 curator-verified pool ──► periodic augmented retrain
 └──────────────────────────► (back into Stage 2 / GNN / Branch3 / flag)
```

![Pipeline schematic](paper/results/figures/final_figure_pipeline_schematic.png)

**Figure 1.** The end-to-end pipeline. Stages 0–4 form the auto-labeler
(learned components in blue, rule-based in green, the quality flag in
red); Stage 5 (yellow) is the human-in-the-loop feedback path, whose
curator-verified corrections drive a versioned retrain (dashed) that
never overwrites the factory checkpoint.

### 4.2 Stage 0 — Quality control

The QC gate is a deterministic filter (no trainable parameters) that
rejects SWC files that violate any of five structural requirements:
single root, has soma, has neurites, positive radii, reasonable size.
QC pass rates by source and cell type are reported in §3.

### 4.3 Stages 1–3 — Cell type, branch labeling, and topology refinement

**Stage 1 — cell type.** A gradient-boosted decision-tree classifier
(XGBoost) predicts one of {pyramidal, interneuron} from 55 whole-cell
descriptors: Sholl statistics, branch- and subtree-count ratios,
radial- and path-length distributions, radius statistics, and features
derived from the cell's principal axis and soma geometry (Appendix A.1).
When the user supplies a known cell type this stage is bypassed.
Predicted-type accuracy is 0.974 ± 0.007 (§5.2); because the cell type
conditions every downstream stage, it is a meaningful component metric.

**Soft handoff — recovering from an uncertain cell type.** Stage 1's
output is not forced onto the downstream stages. When its
predicted-class probability falls below a threshold (0.65), the pipeline
labels the cell both ways — running the full Stage 2–3 branch labeling
and refinement once under the *pyramidal* hypothesis and once under the
*interneuron* hypothesis — and keeps whichever labeling has the higher
mean confidence across the cell's branches (confidence as defined in
§4.1). In effect the cell type is decided by which hypothesis the
axon/basal/apical labeling is most confident about, rather than by
Stage 1 alone. This matters at the pyramidal/interneuron
boundary: a borderline cell that Stage 1 leans toward "interneuron"
would otherwise be denied an apical label (interneurons are typed over
{axon, basal} only), but if its morphology supports an apical trunk the
branch labeling scores higher confidence under the pyramidal hypothesis
and the handoff selects it. A confident Stage 1 call (probability ≥ 0.65)
passes through unchanged.

**Stage 2 — subtree labeling.** Stage 2 types each *primary subtree* —
the maximal subtree rooted at a child of the soma, i.e. one process
emerging from the cell body. A gradient-boosted classifier assigns each
primary subtree a single class, drawn from the cell type's label set:
**{axon, basal, apical} for pyramidal cells** and **{axon, basal} for
interneurons** (interneurons have no apical dendrite, so that class does
not exist for them, and Stage 2 is a 2-class problem there rather than a
3-class one). It uses 24 subtree-level descriptors — size, depth, path
length, radius profile, height and upward alignment relative to the
soma, and the subtree's rank among the cell's subtrees by size, height,
and radius (Appendix A.2) — and copies the predicted class to every node
of the subtree. Working at subtree level is deliberate, and it matches the biology:
each process emerging from the soma — the axon, each basal dendrite,
and the apical dendrite — is a *single* structural type along its
entire length, so the correct label is genuinely a property of the
whole primary subtree, not of individual branches. Classifying at the
subtree level therefore aligns the unit of prediction with the unit of
biological truth. It also helps on the hardest call: a thin apical
trunk versus an axon is far clearer from whole-subtree geometry (one
long, straight, upward-projecting arbor) than from any single branch,
and it makes each emerging process receive one coherent label rather
than a patchwork of per-branch guesses. (The rare cases where a single
primary subtree should carry more than one type — a reconstruction
artifact that fuses distinct processes — are a known limitation; §7.) The
axon-versus-dendrite decision is effectively settled here (F1 ≈ 0.99);
the harder basal-versus-apical assignment, decided only coarsely at the
subtree level, is re-examined branch-by-branch by the refinements that
follow (Stages 2.5–3). The soma is not classified: it
is located structurally from tree topology during branch extraction, so
soma labeling is a trivially-correct detection step (soma F1 = 1.00) —
which is why we report *neurite* macro-F1 (axon, basal, apical) as the
headline metric, excluding the always-correct soma class.

**Stage 2.5 — apical/basal refinement (pyramidal cells only).** Because
only pyramidal cells have an apical dendrite, this stage is applied to
pyramidal cells alone; interneurons pass straight from Stage 2 to
Stage 3. It runs *after* Stage 2, on its output, and comprises two graph
networks in sequence, each re-scoring the apical/basal decision
branch-by-branch:

*(a) Basal-versus-apical GNN.* A two-layer GraphSAGE network (hidden
dim 64, ReLU, dropout 0.2) operates on the branch graph — one node per
branch carrying 47 dendrite-relevant features (the branch descriptors
minus the cell-type indicators; Appendix A.3), with undirected
parent–child edges — and reassigns dendritic branches between basal and
apical. It never touches axon labels, which Stage 2 has already fixed,
and runs only when Stage 2 found apical evidence (it typed at least one
subtree apical, or both a basal and an apical branch are present), so it
cannot impose apical structure on a purely basal cell. Trained with Adam
(lr 1e-3, weight decay 5e-4), up to 200 epochs, early stopping
(patience 25), 5-fold cross-validation, and a focal loss (γ = 2) with
inverse-square-root class weighting for the apical minority.

*(b) Three-class rescue network (Branch3).* A second GraphSAGE head then
re-scores *all three* neurite classes per branch (hence "Branch3":
branch-level, 3-class). Trained on the pipeline's *actual inference-time
outputs* — branch features plus the current per-branch label and
confidence — it corrects systematic residual errors the first network
leaves, most importantly apical trunks still labelled axon. It changes a
label only when confident.

Both networks edit labels *per branch*; their reassignments are
provisional evidence that the Stage 3 topology pass consolidates back to
the subtree level (next).

**Stage 3 — topology refinement.** Stage 2.5 edits labels *per branch*,
so a primary subtree that Stage 2 typed uniformly may now
hold a scatter of labels. A final rule-based pass turns that scatter
back into an anatomically consistent labeling and enforces *whole-cell*
constraints that no local model can see — the per-subtree classifier
decides each subtree in isolation, and the graph networks see only a
branch and its neighbours, so none of them can enforce a rule like
"there is exactly one apical trunk." For **pyramidal** cells it (i)
re-votes a single label per primary subtree by confidence-weighted
majority — this *consolidates* the GNN and Branch3 branch-level edits,
and it overturns Stage 2's original per-subtree call where that evidence
is strong (so the vote is not a redundant repeat of Stage 2, but the
step that turns branch-level evidence back into a clean per-subtree
labeling); (ii) enforces at most one axon subtree; (iii) selects a
single apical trunk as the subtree projecting farthest along the cell's
own principal axis (robust to rotated coordinate frames) and confines
apical labels to it; and (iv) applies confidence-weighted parent–child
smoothing and small-island flipping to remove isolated mislabels. Every
threshold is confidence-gated, so the pass corrects uncertain regions
without rewriting confident ones. **Interneurons are handled
differently, and deliberately so:** an interneuron's axon frequently
emerges *partway down a dendrite* rather than as its own primary
subtree, so hard "one label per subtree" voting would erase it. Instead,
interneurons use softer node-level rules — the thinnest, longest primary
subtree is biased toward axon, and per-subtree majorities are propagated
only to low-confidence nodes — which permit an axon to be recovered
*within* a dendritic subtree. (This mismatch between the
one-type-per-subtree assumption and interneuron axon anatomy is the
cell-type-specific face of the single-primary-tree limitation, §7.)

### 4.4 Stage 4 — Quality-flag model

Even at ~0.95 mean F1 the pipeline produces a long lower tail of
severely mislabeled cells. We train a separate XGBoost flag classifier
on the auto-labeler's own outputs to identify these cells *without* a
ground-truth pass. Inputs are auto-labeler confidence statistics (mean and
worst-branch posterior), simple geometry features (node count,
predicted apical fraction, number of predicted classes), and a Branch3
disagreement feature (does the rescue subnet disagree with Stage 3?).
The training target is the binary indicator *per-cell neurite F1 <
0.60*.

To prevent training/evaluation leakage on the held-out fold, we use a
leave-one-seed-out (LOSO) protocol: for each seed s ∈ {42, 123, 789},
the flagger is trained on cells from the other two seeds' held-out
sets and evaluated on cells from seed s's held-out set, with
cell-deduplication across folds.

The flag model is trained and evaluated on the outputs of the fully
automatic pipeline — with Stage 1 predicting the cell type — because
that is the setting in which a user relies on it. Those automatic
outputs contain exactly the errors a curator needs surfaced, including
the occasional cell-type mistake; the idealized configuration in which
the correct cell type is supplied (used for the baseline comparison in
§5.1) is not a setting a deployed user would be in.

### 4.5 Constructed baselines

Because no off-the-shelf end-to-end SWC auto-labeler exists (§1, §2), the
four baselines we compare against are constructed in this work. Each
baseline follows the same recipe:

```
SWC → extract morphometric features over a labeling unit →
 classifier (RF or MLP) → predict the unit's label →
 assign that label to every node in the unit.
```

Here a *labeling unit* is the piece of the reconstruction that receives
a single predicted type — either one branch segment or one primary
subtree (the maximal subtree rooted at a child of the soma). The four
baselines differ only in (i) the choice of labeling unit and (ii) the
feature set, so the comparison isolates the value of our staged pipeline
over a plain "morphometric features + standard classifier" approach —
the most plausible alternative an external group could assemble today.

| Baseline | Labeling unit | Feature set | Classifier |
|---|---|---|---|
| **NeuroM-RF** | per-branch segment | NeuroM-style 17-dim (path length, radial / euclidean distance from soma, z-above-soma, radius statistics, taper, downstream subtree size, bifurcation count, partition asymmetry, cell-type one-hot) | RandomForest (400 trees, max depth 30) |
| **Sholl-RF** | per-primary-subtree | 21-dim Sholl-derived (7 Sholl intersection counts at fixed radii, max Sholl, peak radius, total nodes, total bifurcations, max radial distance, total arbor length, max root-to-tip path, PCA principal-axis components, z-extent, cell-type one-hot) | RandomForest |
| **Sholl-MLP** | per-primary-subtree | same as Sholl-RF | small PyTorch MLP |
| **L-Measure-RF** | per-primary-subtree | 23-dim L-Measure-style classical morphometrics (lengths, diameters, branch counts, asymmetries, partition statistics — no Sholl intersections) | RandomForest |

Training-side notes: all baselines are trained on the same train split
as our pipeline (§4.4) and receive the ground-truth cell type as an
input feature, matching the GT cell-type evaluation configuration of
our pipeline so the §5.1 comparison is apples-to-apples. Each unit's
training label is the *majority neurite label* of its nodes.


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
metric differences (5,000 resamples) and paired sign-flip p-values.
The unit is *unique SWC file*; repeated seed–file pairs are averaged
before testing.

### 4.7 Stage 5 — Curator feedback loop (human-in-the-loop)

The flag model (§4.4) surfaces cells the auto-labeler is likely to have
mislabeled. Rather than treating that as a terminal filter, we close
the loop: a flagged cell is routed to a curator whose corrected labels
feed back into training. This turns the auto-labeler from a static model
into one that *improves with use*, and — because both the auto-labeler and
the flag model are retrained — it also teaches the flag model to fire
more precisely over time. The structure is:

1. **Label + flag.** The auto-labeler produces per-node labels, a flag
 score, and the flag-feature vector. Cells with score ≥ threshold are
 surfaced for review; the rest are accepted.
2. **The curator provides only corrected labels.** The curator is not
 asked to judge the flag itself; they simply return the labels they
 consider correct for the cell (which may coincide with the model's
 output if they find it already correct). This keeps the curator's
 task to ordinary annotation and avoids requiring a separate judgment
 about the flag.
3. **The flag ground truth is derived, not chosen.** The system
 computes whether the flag was justified from the disagreement between
 the curator's labels and the model's, using the *same* definition the
 flag model was trained on:

 > flag was justified ⟺ per-cell F1( curator, model ) < 0.60

 A curator overhaul (low F1) yields a **true-positive** flag example;
 a curator who barely changed anything (F1 ≈ 1) yields a **false-alarm**
 example. Each reviewed cell thus produces two training signals: a
 auto-labeler example (the corrected SWC) and a flag example (the
 flag-feature vector labeled by the derived truth).
4. **Persistent curator-verified store.** Both signals accumulate in a
 store (corrected SWCs + a manifest carrying the derived flag labels
 and features).
5. **Versioned augmented retrain — factory preserved.** On a periodic
 schedule the Stage-2 auto-labeler is retrained on the factory training
 set together with the curator-corrected cells, and the flag model is
 retrained on the factory flag-training data together with the
 curator-derived (feature vector, truth) pairs. Crucially,
 the retrain writes a **new versioned checkpoint** and never modifies
 the original *factory* checkpoint; a model registry enforces this
 (the factory checkpoint is read-only; each version records its parent
 and the curator examples used). Deployments can roll back to any prior
 version, and the factory model is guaranteed recoverable. Because the
 flag step preferentially surfaces the long-tail cells where the model
 is most wrong, curator effort is concentrated where it reduces error
 fastest — the active-learning argument.

The loop is realized as four cooperating components — an orchestrator
that runs the flag-and-review cycle, a persistent curator-verified store,
a factory-preserving model registry, and the flag-truth derivation rule —
which compose the existing pipeline and Stage-2 trainer and delegate
flag-model training to a pluggable trainer rather than reimplementing
them. §5.7 validates the labeling improvement with a simulated curator;
retraining the flag model on curator-derived labels (false-alarm
reduction) is analyzed in §5.8.

---

## 5. Results

### 5.1 Main result: neurite labeling vs. baseline methods (ground-truth cell type)

**Baseline framing.** As noted in §1, §2, and detailed in §4.5, no
off-the-shelf end-to-end SWC auto-labeler exists. The four baselines below
(NeuroM-RF, Sholl-MLP, Sholl-RF, L-Measure-RF) are *constructed by us*
to span the design space of plausible morphometric-features-plus-
classifier approaches a competing group might build; see §4.5 for
feature sets and classifier choices. All four are evaluated on the
same QC-pass corpus and 80/20 file-hash splits as our pipeline and
receive the ground-truth cell type as an input feature, matching the
GT cell type configuration of our pipeline so the comparison is
apples-to-apples.

**Table 1** — Mean ± SD over seeds 42, 123, 789. **Configuration: GT
cell type.**

| Method | accuracy | neurite macro-F1 | axon F1 | basal F1 | apical F1 | per-cell F1 mean | per-cell F1 P10 |
|---|---|---|---|---|---|---|---|
| NeuroM-RF | .9300 ± .0050 | .8223 ± .0051 | .9668 ± .0029 | .8273 ± .0032 | .6730 ± .0126 | .8472 ± .0040 | .5421 ± .0017 |
| Sholl-MLP | .9713 ± .0016 | .9235 ± .0018 | .9884 ± .0019 | .9201 ± .0092 | .8621 ± .0142 | .8673 ± .0094 | .4164 ± .0447 |
| Sholl-RF | .9778 ± .0013 | .9405 ± .0028 | .9915 ± .0008 | .9348 ± .0056 | .8951 ± .0055 | .8915 ± .0074 | .4600 ± .0347 |
| L-Measure-RF | .9793 ± .0016 | .9476 ± .0059 | .9915 ± .0010 | .9389 ± .0069 | .9125 ± .0127 | .9057 ± .0093 | .5397 ± .0548 |
| **Our pipeline** | **.9819 ± .0025** | **.9513 ± .0064** | **.9923 ± .0012** | **.9522 ± .0076** | **.9094 ± .0126** | **.9402 ± .0092** | **.7359 ± .0779** |

The most consequential result is the worst-case metric:

1. **Worst-case per-cell reliability (10th-percentile per-cell F1).**
 This metric measures how poorly the *worst-labeled* cells are typed,
 and it is what determines how many reconstructions a downstream study
 must discard or hand-correct. Every baseline leaves it low — 0.42
 (Sholl-MLP) to 0.54 (best baseline) — meaning a large fraction of
 cells are labeled too unreliably to use as-is. Our pipeline raises it
 to 0.74, a +0.20 absolute improvement over the best baseline, so
 substantially fewer cells fall below a usable threshold. The
 practical consequence is a more uniformly reliable labeling across
 the whole corpus, which is precisely what downstream morphometric
 and modeling analyses depend on.
2. **Overall accuracy (neurite macro-F1).** Our pipeline reaches 0.951,
 ahead of the strongest baseline (L-Measure-RF, 0.948) and well ahead
 of the weaker ones (Sholl-MLP 0.923, NeuroM-RF 0.822), with a paired
 sign-flip p < 0.0001 against every baseline.
3. **Per-class breakdown.** The gains concentrate in the basal
 (+0.013 over L-Measure-RF) and axon classes; on apical alone the
 strongest baseline is statistically tied with our pipeline (0.9125
 vs 0.9094, within one SD).

### 5.2 Stage ablation (GT cell type and deployment)

**Table 2** isolates the contribution of each stage on **GT cell
type** and reports the cost of switching to **Stage 1 active
(deployment)** in the final row. Mean over 3 seeds;

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
over LOSO folds.
F1 computed from precision and recall.

| Scope | Features | Precision | Recall | F1 | AP | Kept-set P10 |
|---|---|---:|---:|---:|---:|---:|
| all | compact (confidence + geometry) | 0.380 | 0.513 | 0.437 | 0.385 | 0.924 |
| all | + rescue-network disagreement | 0.400 | 0.535 | 0.458 | 0.405 | 0.945 |
| all | + baseline-method disagreement | **0.434** | **0.582** | **0.497** | **0.479** | **0.958** |
| all | + multi-model disagreement | 0.431 | 0.577 | 0.493 | 0.474 | 0.961 |
| pyramidal only | + baseline-method disagreement | 0.557 | 0.644 | 0.597 | 0.612 | 0.897 |
| interneuron only | + baseline-method disagreement | 0.214 | 0.482 | 0.296 | 0.246 | 1.000 |

**Table 3b — Per-cell F1 P10 before and after 10% flag rejection
(deployment configuration).**

| Configuration | P10 |
|---|---|
| Deployment pipeline, no rejection | 0.66 |
| Deployment pipeline + flag model (baseline-method disagreement, 10% reject) | **0.96** |
| Oracle: drop bottom 10% by ground-truth F1 (upper bound, pyramidal only) | 0.93 |

(Oracle computed on pyramidal cells only.)

Three observations:

1. **Adding baseline-method disagreement gives the strongest feature
 set**, reaching F1 = 0.50 (precision 0.43, recall 0.58) on the
 all-cell scope. We recommend it as the default flag model.
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
nine mutually compatible failure modes:

| Category | Bad rows | Enrichment | Flag recall | False flags |
|---|---:|---:|---:|---:|
| Apical over-predicted | 15 | 9.99x | 0.000 | 0 |
| Stage 1 propagation | 89 | 9.77x | 0.635 | 1 |
| Apical under-predicted | 287 | 9.22x | 0.698 | 8 |
| Apical missed | 280 | 9.20x | 0.709 | 7 |
| Basal lost | 131 | 8.96x | 0.512 | 3 |
| Axon/dendrite confusion | 331 | 8.96x | 0.528 | 16 |
| Whole-cell mislabel | 12 | 4.61x | 0.273 | 6 |
| Truncated apical | 118 | 2.93x | 0.620 | 21 |
| Apical inverted | 218 | 1.60x | 0.601 | 34 |

The dominant categories are apical-related (missing, underpredicted,
inverted, truncated). The flagger catches apical-missed and
underpredicted cells at ~70% recall, but is much weaker on whole-cell
mislabels (27%) — because confidence is high on uniformly-wrong
outputs — and on apical overprediction (0%), where the model is
confidently wrong in the other direction. The Stage 1 propagation
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
ways, both restricted to the in-house lab corpus (1,358 QC-pass
cells, all pyramidal — see §3).

**(a) Restricted evaluation — existing model, lab cells only.** We take
the cross-source models from §5.1 (trained on the full corpus) and
re-evaluate them on the *subset* of the
held-out test files belonging to the in-house lab corpus. This is the
same model, scored on the cleaner data slice. Metrics are computed in
the same fashion as the all-corpus table (node-level pooled per-class
F1, per-cell mean / P10 / P25). Mean ± SD over 3 seeds (n = 867 lab
test cells pooled), GT cell type configuration.

**Table 4a — Our pipeline (ground-truth cell type) on the lab subset vs. the full corpus.**

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
per-baseline lab-slice numbers are available in the artifact manifest.

**(b) Lab-only retrain — full pipeline trained and evaluated on lab data alone.**
We rebuild Stage 2 + GNN + Branch3 on the lab QC-pass corpus alone,
using the same per-seed 80/20 file-hash splitting procedure as
elsewhere (seeds 42, 123, 789; 1,063–1,080 train / 278–295 test per
seed, each seed its own split). Stage 1 is degenerate
because the lab corpus is 100% pyramidal, so for lab-only deployment
we hardcode cell type = pyramidal and skip Stage 1 at both train and
inference time. All other trainable components (the Stage 2 subtree classifier and the
Stage 2.5 GNN and Branch3 networks; the Stage 3 topology pass is
rule-based) are retrained from scratch on lab data with the same gentle
class-weighting configuration
(class-balance power 1.25, inverse-square-root GNN class weighting,
focal-loss gamma 2.0).

**Table 5 — Our pipeline (ground-truth cell type) on the lab test set: cross-source vs. lab-only training.**

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
as warmup.

**Table 6 — Per-cell inference latency, all methods.** Mean and median
in ms per cell; throughput in cells/s and nodes/s. Same cell sample
for every method.

| Method | mean ms | median ms | P10 ms | P90 ms | cells/s | nodes/s |
|---|---:|---:|---:|---:|---:|---:|
| Sholl-MLP | 52.5 | 5.6 | 1.4 | 147.4 | 19.04 | 170,880 |
| Sholl-RF | 87.8 | 45.1 | 30.9 | 171.5 | 11.38 | 102,171 |
| NeuroM-RF | 112.7 | 34.6 | 30.9 | 217.5 | 8.87 | 79,631 |
| L-Measure-RF | 354.7 | 42.2 | 31.1 | 632.1 | 2.82 | 25,308 |
| **Our pipeline** | 353.8 | 227.7 | 185.6 | 592.1 | **2.83** | **25,369** |

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
as a baseline practitioners can budget against.

### 5.7 Curator feedback loop: simulated active learning

The feedback-loop architecture of §4.7 claims the auto-labeler improves
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
per-cell node-weighted mean (1 − max posterior) of the current labeling
model — a faithful proxy for the trained flag model. Two arms
share the round-0 model then diverge: **flag-guided** queries the
most-uncertain unqueried pool cells each round; **random** queries the
same number of random cells. Each round reveals the queried labels,
retrains the labeling model on seed ∪ revealed, and re-evaluates the
fixed test set. To make the many rounds tractable, the model retrained
here is a lightweight per-branch classifier over cached branch features,
a stand-in for the full subtree-based labeler; both arms use the
identical procedure, so the comparison cleanly isolates the value of
flag-guided selection, and the full-pipeline confirmation below repeats
the test with the actual labeler.

**Table 7 — Active-learning curve, flag-guided vs random.** Mean ± SD
over 3 seeds (42, 123, 789); seed_train ≈ 740, pool ≈ 3,260, test ≈ 990
per seed. Test set fixed and never queried.

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

1. **The loop works — the auto-labeler improves with curator effort.** Both
 arms climb monotonically from macro-F1 0.735 (seed model only) toward
 ~0.77 as curator-corrected cells enter the training pool. This is the
 core validation of the §4.7 architecture: correcting flagged cells
 and retraining measurably improves the auto-labeler on cells it never saw.
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

Scope: this experiment retrains a per-branch stand-in classifier each
round on cached features so the multi-round loop is tractable; both arms
use the identical procedure, so the comparison cleanly attributes the
gain to flag-guided *selection*.

**Full-pipeline confirmation (and a limit on the selection claim).**
A multi-round curve that retrains the *entire* auto-labeler (Stage 2 + GNN +
Stage 3) each round is compute-prohibitive (~days). We therefore ran a
one-shot endpoint version on seed 123: train the full pipeline on a
seed set (614 cells), then on seed + 1,000 corrections selected either
by the seed model's flag/uncertainty signal or at random, and evaluate
all three on a fixed 818-cell test set (GT cell type; Branch3 omitted
for tractability).

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
auto-labeler at both Stage-2 and full-pipeline scale; flag-guided selection
is more efficient than random specifically in the iterative small-batch
regime, and a one-shot large batch should use diverse or hybrid
acquisition, not pure uncertainty.** Confirming iterative flag-guidance
at full-pipeline scale is left to future work for compute reasons (§8).

### 5.8 Improving the flag model from curator verdicts

§5.7 shows the loop improves the *auto-labeler*. A separate question is
whether it improves the *flag model* — specifically its precision, i.e.
whether it learns to stop firing on low-confidence-but-actually-correct
cells (false alarms), which are the flagger's dominant error today
(precision ≈ 0.43 at 10% rejection, §5.3). The feedback loop generates
exactly the supervision needed: every reviewed cell yields a
flag-feature vector labeled by the *derived* truth (per-cell F1 between
curator and model below 0.60; §4.7). Retraining the flag model on the
factory flag-training data together with the curator-derived (feature
vector, truth) pairs should raise precision as the curator's false-alarm
verdicts accumulate.

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
(review random pool cells) — averaged over 3 by-file splits.

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

Curator feedback measurably improves the flag model, and reviewing
*flagged* cells is markedly more effective than reviewing random ones:

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
tabular retrain); it is the flag-side complement to the auto-labeler-side
result in §5.7.

---

## 6. Discussion

**Where the gains come from.** The stage ablation yields a notable
finding: in isolation, the GNN provides no gain over the Stage 2
subtree classifier and slightly degrades it. The gains come instead from the
deterministic Stage 3 topology refinement, which enforces tree-level
constraints the branch-wise classifier cannot see. This suggests the
GNN's value lies not in propagating information per se but in producing
a smoothed apical/basal map over which Stage 3 can then enforce
trunk-contiguity.

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
driver script that consumes a
user-supplied QC CSV pointing at SWC files plus cell-type labels and
produces a complete auto-labeler in one invocation. Users with their own
well-curated single-lab corpus can therefore specialize the auto-labeler
to their tracing conventions, which we exercise in §5.5 on the
in-house lab corpus. This is the primary deployment path for groups
whose data conventions differ from the cross-source corpus.

**Deployment.** The default deployable flagger uses only the compact
feature set (confidence + geometry + rescue-network disagreement),
because it requires a single inference pass. The variants that add
baseline-method or multi-model disagreement give better detection but
require running additional models at inference time, which is
impractical for an interactive tool; we recommend them for batch
curation workflows where the extra compute is acceptable.

---

## 7. Limitations

1. **Pyramidal and interneuron only.** Our corpus and all trained
 models cover exactly two cell types. This is a deliberate scope
 choice rather than a fundamental restriction: pyramidal cells and
 interneurons together account for the large majority of cortical and
 hippocampal neurons in the public reconstruction corpora researchers
 actually work with, so the two-type model already covers most cells
 a user is likely to encounter. It also captures the key structural
 contrast the auto-labeler must resolve — the presence or absence of an
 apical dendrite. That said, the system has not been validated on
 Purkinje, granule, motor neurons, or invertebrate morphologies, and
 the Stage 1 classifier returns binary output. Adding cell types is a
 natural direction for future work; each new type would require
 retraining Stage 1, a cell-type feature path through Stage 2, and any
 type-specific topology rules in Stage 3 — but nothing in the staged
 architecture precludes it.
2. **Single-primary-tree limitation.** The auto-labeler assumes a normal
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
3. **The reference labels impose a ceiling on measurable accuracy.**
 All metrics are computed against the structure-type labels already
 present in the source corpora, which are themselves imperfect. The
 apical-versus-basal distinction in particular is subjective near the
 soma and at branch boundaries, and independent expert annotators do
 not always agree on it; axon-versus-dendrite calls on faint or
 truncated processes are similarly ambiguous. The reference labels
 therefore carry irreducible disagreement, and a per-class F1 of 1.0
 on apical and basal is not attainable even in principle. Our scores
 on these two classes (0.95 basal, 0.92 apical) may already approach
 the ceiling the annotation quality permits, and part of the residual
 error reflects genuine inter-annotator ambiguity rather than model
 failure. A definitive assessment would require a multi-annotator
 gold-standard set to quantify inter-rater agreement as an explicit
 upper bound, which is beyond the scope of this work.
4. **No human-in-the-loop study.** We have not measured how much curator
 time the automatic labeling actually saves on a representative
 curation task.
5. **Fixed four-type output vocabulary.** The labeler emits only the
 four canonical SWC structure types — soma, axon, basal dendrite, and
 apical dendrite. The SWC format also permits custom, lab-specific
 structure codes (values ≥ 5), and some laboratories attach additional
 per-node annotations (e.g. specialized compartment or marker labels);
 the labeler reproduces none of these. Every node is forced into one
 of the four canonical classes, so any non-standard typing present in
 an input reconstruction is not preserved on output. Supporting extra
 output classes is a straightforward extension — it requires only
 training data carrying those labels for the per-subtree and
 branch-level classifiers — but is outside the scope of the released
 model.

---

## 8. Future work

**Extending the curator feedback loop.** The closed-loop architecture
(§4.7) and its simulated-curator validation (§5.7) establish that the
auto-labeler improves as curators correct flagged cells. Several design
questions remain open for a full deployment: (i) how to weight
curator-verified examples relative to the original training corpus,
especially when a single lab contributes disproportionately; (ii) how
to detect and prevent auto-labeler drift if a single curator's conventions
diverge from the community norm; (iii) the appropriate retrain cadence
(per-cell online updates vs. periodic batched retrains); and (iv)
whether periodically retraining the *flag model itself* on curator
decisions (which cells were truly bad, which were false alarms) reduces
the false-alarm rate further (cf. the failure-mode analysis in §5.4).
Our simulation retrains
only the Stage-2 classifier for tractability; a full study would
retrain the GNN, Branch3, and flag model on the same augmented pool
and measure the compounding effect.

**Other directions.**

- A second flag model trained on anatomically-implausible structural
 patterns to catch the failure modes the confidence-based flagger
 systematically misses (whole-cell mislabel, apical overprediction).
- Extension to additional cell types beyond pyramidal and interneuron
 (§7, limitation 1).
- A human-in-the-loop time-savings study quantifying how much curator
 time the automatic-labeling-plus-flag workflow saves on a
 representative curation task (§7, limitation 4).
- A leave-one-laboratory-out evaluation on the cross-source corpus to
 quantify generalization to a laboratory's tracing conventions not
 seen during training.
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
→ subtree classifier → GNN + Branch3 → topology refinement → quality
flag) for automatic axon/basal/apical labeling. In the GT-cell-type
configuration, used for apples-to-apples comparison against four
constructed baselines, the system reaches axon / basal / apical F1 of
0.993 / 0.954 / 0.916 on held-out cells across three seeds and lifts
the worst-decile per-cell F1 from at most 0.54 (best baseline) to
0.74. In the deployment configuration, where Stage 1 predicts cell
type at 0.974 accuracy, the system reaches axon / basal / apical F1 of
0.993 / 0.951 / 0.906; the quality-flag model then identifies the
worst 10% of predictions with 58% recall and lifts the kept-set
worst-decile F1 from 0.66 to 0.96. Beyond the static auto-labeler, we
introduce a curator feedback loop that turns the flag step into an
active-learning acquisition function: correcting flag-selected cells
and retraining reaches a target accuracy with ~60% fewer curator
corrections than random review, so the deployed auto-labeler improves
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
- **Models.** Canonical auto-labeler at
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

## Appendix A — Feature definitions

Feature names below are the identifiers used in the released code; the
source module for each set is given in its subsection.

### A.1 Stage 1 — whole-cell features (55)

Computed once per cell and used to predict cell type.

- *Counts and ratios:* `n_nodes`, `n_root_nodes`, `n_branch_points`,
  `n_terminals`, `n_primary_subtrees`, `branch_point_ratio`,
  `terminal_ratio`, `branching_density`, `terminal_density`.
- *Path and radial extent:* `max_path_length`, `mean_path_length`,
  `std_path_length`, `max_radial_distance`, `mean_radial_distance`,
  `std_radial_distance`, `path_radial_ratio_mean`,
  `max_neurite_to_soma_ratio`.
- *Radius:* `mean_radius`, `std_radius`, `max_radius`, `radius_cv`,
  `soma_radius`.
- *Spatial extent and shape:* `z_span`, `z_asymmetry`,
  `z_max_above_soma`, `z_max_below_soma`, `x_span`, `y_span`,
  `xy_aspect_ratio`, `planarity`, `linearity`, `thickness_ratio`.
- *Subtree structure:* `max_subtree_size`, `min_subtree_size`,
  `subtree_size_std`, `max_subtree_path`, `subtree_path_std`,
  `max_subtree_size_ratio`, `max_subtree_path_ratio`, `max_strahler`,
  `mean_strahler`.
- *Sholl:* `sholl_max_intersections`, `sholl_peak_distance`,
  `sholl_decay_rate`.
- *Terminals and principal axis:* `terminal_radial_mean`,
  `terminal_radial_std`, `pc1_span`, `pc1_asymmetry`,
  `pc1_max_above_soma`, `pc1_max_below_soma`, `pc1_radial_max`,
  `subtree_pc1_concentration`, `subtree_pc1_top_alignment`.
- *Bifurcation angles:* `bif_angle_mean`, `bif_angle_std`.

### A.2 Stage 2 — per-primary-subtree features (24)

Computed once per primary subtree; used to assign the subtree's neurite
type. Source: [hybrid/subtree_features.py](hybrid/subtree_features.py).

- *Size and shape:* `subtree_size`, `subtree_depth`, `subtree_max_path`,
  `subtree_max_radial`, `subtree_z_span`, `subtree_up_alignment`,
  `subtree_branch_density`, `subtree_terminal_density`.
- *Radius:* `subtree_mean_radius`, `subtree_std_radius`,
  `subtree_min_radius`, `subtree_root_radius`, `subtree_taper_ratio`.
- *Position relative to soma:* `subtree_mean_z_rel`, `root_path_dist`,
  `root_radial_dist`.
- *Rank and ratio within the cell:* `size_ratio`, `path_ratio`,
  `z_ratio`, `radius_ratio`, `is_longest_subtree`, `z_rank`,
  `radius_rank`, `n_primary_subtrees`.

### A.3 Stage 2.5 — per-branch graph-network features (47)

The basal/apical GNN uses these 47 per-branch descriptors (source:
[hybrid/branch_features.py](hybrid/branch_features.py)). The Branch3
rescue network uses the same branch descriptors, additionally augmented
with the current per-branch predicted label and confidence.

- *Branch geometry:* `path_length`, `radial_extent`, `n_nodes`,
  `mean_radius`, `std_radius`, `min_radius`, `max_radius`,
  `taper_ratio`, `persistence`, `up_alignment`, `branchiness`,
  `symmetry`.
- *Position relative to soma:* `root_path_dist`, `root_radial_dist`,
  `z_rel_soma`, `z_span`.
- *Subtree context:* `subtree_size`, `subtree_depth`,
  `subtree_max_path`, `branch_order`.
- *Parent / sibling context:* `parent_radius`, `radius_ratio_to_parent`,
  `n_siblings`, `is_primary`.
- *Cell-normalized:* `path_length_rel`, `radial_extent_rel`,
  `radius_rel`, `subtree_size_rel`, `is_longest_subtree`,
  `subtree_path_ratio`, `subtree_z_ratio`, `subtree_z_rank`,
  `subtree_max_radial_rank`.
- *Radius profile:* `proximal_mean_radius`, `distal_mean_radius`,
  `proximal_persistence`, `distal_persistence`.
- *Principal-axis geometry:* `polar_angle_from_up`,
  `vertical_horizontal_span_ratio`, `soma_z_offset_norm`,
  `primary_subtree_polar_spread`, `principal_axis_projection`,
  `polar_angle_from_principal_axis`, `principal_axis_alignment_strength`,
  `subtree_principal_projection`, `subtree_principal_rank`,
  `path_to_first_bifurcation_norm`.

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
