"""Build a Word doc of speaker notes for the SWC-Studio deck."""
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

NAVY = RGBColor(0x1B, 0x2A, 0x4E)
CORAL = RGBColor(0xE0, 0x78, 0x56)
MUTED = RGBColor(0x64, 0x74, 0x8B)

doc = Document()

# Page margins
for section in doc.sections:
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)

# Default body style
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)


def add_title(text, size=22, color=NAVY, bold=True, after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    r = p.add_run(text)
    r.font.name = "Georgia"
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    return p


def add_subtitle(text, size=12, color=CORAL):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text.upper())
    r.font.name = "Calibri"
    r.font.size = Pt(size)
    r.font.bold = True
    r.font.color.rgb = color
    return p


def add_body(text, size=11, after=10):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.35
    r = p.add_run(text)
    r.font.name = "Calibri"
    r.font.size = Pt(size)
    return p


def add_section_break(label_text=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(8)
    if label_text:
        r = p.add_run("— " + label_text + " —")
        r.font.size = Pt(9)
        r.font.color.rgb = MUTED
        r.italic = True
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER


# --- Title page --------------------------------------------------------------
add_title("SWC-Studio  ·  Speaker Notes", size=26, after=4)
p = doc.add_paragraph()
r = p.add_run("Read each block as you advance through results_summary.pptx. "
              "Each section is ~30–45 seconds out loud, ~8–10 minutes total.")
r.font.size = Pt(11); r.italic = True; r.font.color.rgb = MUTED
p.paragraph_format.space_after = Pt(18)


# --- Slide-by-slide notes ----------------------------------------------------
slides = [
    ("Slide 1", "Title",
     "Welcome. Today I'll walk you through SWC-Studio — a tool we've built to "
     "automatically label the parts of a neuron from its 3D reconstruction. "
     "We'll cover the software, the data, the algorithm, how we measure success, "
     "and where we are now. The full pipeline runs in three stages, and the "
     "latest version reaches a pooled F1 score of 0.98 across 1,737 cells."),

    ("Slide 2", "Introducing SWC-Studio",
     "SWC-Studio is a toolkit for working with neuron morphology files. It "
     "comes in three forms that all share the same backend: a desktop GUI for "
     "interactive inspection and repair, a command-line tool for batch "
     "processing, and a Python library you can import into your own "
     "pipelines. Its core capabilities include validation and repair of SWC "
     "files, radii cleaning, manual editing, batch processing, and "
     "rule-based auto-labeling with automatic apical detection. Today we'll "
     "focus on that last piece — the auto-labeling engine. It's a three-"
     "stage hybrid pipeline that combines machine learning with biology "
     "rules to label every point in a neuron as soma, axon, basal, or "
     "apical."),

    ("Slide 3", "The problem",
     "There are over a hundred thousand 3D neuron reconstructions in public "
     "databases like NeuroMorpho and the Allen Institute. The trouble is, the "
     "labels are inconsistent — some files only label the cell body, others "
     "label axon vs. dendrite but disagree on apical, and there's no single "
     "standard. An expert can re-label one cell in 10 to 30 minutes, but "
     "doing a thousand by hand is a project in itself. So we built a fully "
     "automatic pipeline that takes any SWC file and assigns a label to every "
     "point."),

    ("Slide 4", "The dataset",
     "We trained and tested on 1,737 labeled cells — 932 pyramidals and 805 "
     "interneurons — totaling about 5.4 million individual 3D points. The "
     "data comes from three sources: NeuroMorpho.org, the Allen Cell Types "
     "Database, and a handful of custom lab reconstructions for benchmarking. "
     "Each cell is stored as an SWC file — essentially a table where every "
     "row is one 3D point with an ID, a type code, coordinates, a radius, "
     "and a pointer to its parent point. The type codes are: 1 for soma, "
     "2 for axon, 3 for basal dendrite, and 4 for apical dendrite."),

    ("Slide 5", "The pipeline",
     "The pipeline has three stages. Stage 1 decides what kind of cell we're "
     "looking at — pyramidal or interneuron. This matters because "
     "interneurons don't have an apical dendrite at all. Stage 2 is the "
     "per-branch machine-learning classifier — for each segment of the cell, "
     "it predicts axon, basal, or apical. Stage 3 is a rule-based refinement "
     "layer that applies known biology — like “a pyramidal cell has at most "
     "one apical.” Input is a raw SWC file; output is the same file with "
     "every point labeled."),

    ("Slide 6", "Stage 1 overview",
     "Stage 1 is a binary classifier — pyramidal versus interneuron — that "
     "looks at the whole cell's shape. It computes 49 shape descriptors and "
     "feeds them into an ensemble model. Categories include size and counts, "
     "spatial extent, branching shape, and a new set of principal-axis "
     "features that capture the cell's intrinsic up–down direction. The big "
     "new feature here is the soft handoff — when the model isn't confident, "
     "instead of forcing a decision, we run the rest of the pipeline both "
     "ways and keep whichever produces sharper predictions. This rescues "
     "borderline cells that used to be locked into the wrong path. Current "
     "accuracy is 98.2%."),

    ("Slide 7", "Stage 1 in detail",
     "Here's the full feature breakdown. We have basic counts — branch "
     "points, terminals, subtrees. Path and distance statistics. Radius "
     "statistics. Spatial-extent measures including z-axis features. "
     "Subtree-level descriptors and Strahler-order topology. And the new "
     "principal-axis (PCA) features — these compute the cell's own intrinsic "
     "axis, so the model isn't fooled when a cell has been rotated or when "
     "the slice was cut thin. The PCA features were added in April and "
     "noticeably improved the model's robustness to coordinate-frame quirks "
     "in the data."),

    ("Slide 8", "Stage 2 overview",
     "Stage 2 takes the cell, splits it into branches between branch points, "
     "and predicts a label for each branch. It uses 57 features per branch — "
     "geometry, position relative to the soma, parent and sibling context, "
     "and apical-versus-basal-specific cues. Crucially, we use two different "
     "models: one trained only on pyramidals, one only on interneurons. "
     "Stage 1 picks which model to apply. Each prediction comes with a "
     "confidence score that Stage 3 uses later. The latest pooled F1 scores "
     "are 0.997 for axon, 0.97 for basal, and 0.94 for apical."),

    ("Slide 9", "Stage 2 in detail",
     "Ten feature categories per branch. Branch geometry covers length, "
     "straightness, radius statistics. Position-vs-soma features capture how "
     "far this branch is from the cell body. Subtree and order features "
     "capture how deep this branch sits in the tree. Cell-normalized "
     "features divide local measurements by cell-level maxima — so a long "
     "branch in a small cell looks different from a long branch in a big "
     "cell. The apical-vs-basal cues specifically target the hardest "
     "distinction: dendrite versus dendrite. And again the new PCA and "
     "branching-rate features were added recently to address specific "
     "failure modes — apicals project along the principal axis, axons "
     "branch sparsely while apicals branch densely."),

    ("Slide 10", "Stage 3 overview",
     "Stage 3 applies biology rules that are easy to state but hard for the "
     "ML to learn from data alone. The four big ones: at most one apical per "
     "cell — if multiple are predicted, demote all but the strongest. "
     "Branches inside the same subtree should agree on a label. When picking "
     "the apical winner, prefer the subtree aligned with the cell's "
     "principal axis. And confident predictions can override less confident "
     "neighbors."),

    ("Slide 11", "Stage 3 in detail",
     "The full rule list, in execution order. We start with subtree majority "
     "voting — collapse the per-branch predictions inside each subtree into "
     "a single subtree-level label. Then the PCA-based apical picker — for "
     "pyramidals only. Then the single-apical enforcement. Then we constrain "
     "stray apical labels to actually live inside the apical-owner subtree. "
     "Parent-child smoothing fixes flip-flops along long branches. Island "
     "flipping catches isolated wrong-label runs. We finish with a soft "
     "majority pass that weights by confidence, then two cell-type-specific "
     "cleanups: an axon bias for thin descending interneuron branches, and "
     "removing stray single-node soma labels left over from tracing artifacts."),

    ("Slide 12", "How we measure success",
     "We use a confusion matrix. The rows are the ground truth — what an "
     "expert labeled. The columns are what our model predicted. Cells on the "
     "diagonal are correct predictions; cells off the diagonal are mistakes. "
     "So in this example: 230,000 apical points were correctly labeled "
     "apical. About 5,000 true apical points got called basal, and 14,000 "
     "true basal points got called apical. Those two off-diagonal numbers "
     "are our remaining bottleneck — the dendrite-versus-dendrite confusion."),

    ("Slide 13", "Key terms",
     "Three numbers we use to score the model. Precision asks: when we say "
     "apical, are we right? Recall asks: did we find all the real apicals? "
     "Either one alone can be gamed — you can have perfect precision by only "
     "predicting “apical” when you're absolutely sure, but miss most of "
     "them. F1 combines the two into a single score from 0 to 1, where 1 is "
     "perfect. We also report results two ways — pooled, where every point "
     "counts equally, and per-file, where every cell counts equally "
     "regardless of size. Both views matter because they tell us different "
     "things about the failure modes."),

    ("Slide 14", "Latest results",
     "Where we are today. Overall accuracy is 99.2%. Pooled F1 is 0.976, "
     "per-file mean F1 is 0.963. Soma is fully solved at 1.000. Axon "
     "is solved at 0.997. Basal dendrite is at 0.97, and apical dendrite at "
     "0.94 — that's the bottleneck. The remaining work is almost entirely "
     "about better separating the two dendrite classes. Notably, the bottom "
     "tenth percentile of files now scores 0.90 — a 22-point jump from the "
     "previous version, driven by the principal-axis features and the soft "
     "Stage-1 handoff."),

    ("Slide 15", "What's next",
     "Three priorities for closing the apical gap. First — trunk-detection "
     "features. Apicals have one dominant trunk before branching; basals "
     "are bushy from the start. We're going to encode that directly. "
     "Second — a dedicated apical-vs-basal head. Right now the model decides "
     "between four classes at once; specializing on just the dendrite-vs-"
     "dendrite decision should help. Third — handle remaining edge cases. "
     "After dataset cleanup, the worst-scoring files are now atypical-but-"
     "valid morphology rather than label errors, plus a few rare cases like "
     "z-axis-inverted reconstructions worth flagging at load time. Goal is "
     "apical F1 above 0.95 and a tool the field can rely on. Thank you — "
     "happy to take questions."),
]

for tag, title, body in slides:
    add_subtitle(tag)
    add_title(title, size=18, after=4)
    add_body(body, after=14)

out = "/Users/tuo/Desktop/SWC-Studio/paper/speaker_notes.docx"
doc.save(out)
print("WROTE", out)
