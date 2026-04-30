"""SWC-Studio overview deck — medical-audience version.

15 slides:
1.  Title
2.  Biology in 60 seconds
3.  The problem
4.  The dataset
5.  The pipeline at a glance
6.  Stage 1 overview (cell-type detection)
7.  Stage 1 features in detail            [NEW]
8.  Stage 2 overview (per-branch labeling)
9.  Stage 2 features in detail            [NEW]
10. Stage 3 overview (topology rules)
11. Stage 3 rules in detail               [NEW]
12. How we measure success (confusion matrix explained)
13. Key terms (precision / recall / F1)
14. Latest results
15. What's next
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# ---------- Palette ----------
NAVY     = RGBColor(0x1B, 0x2A, 0x4E)
CREAM    = RGBColor(0xF4, 0xEF, 0xE6)
INK      = RGBColor(0x14, 0x1B, 0x2D)
MUTED    = RGBColor(0x64, 0x74, 0x8B)
CORAL    = RGBColor(0xE0, 0x78, 0x56)
SAGE     = RGBColor(0x4A, 0x7C, 0x7E)
GOLD     = RGBColor(0xC9, 0x9A, 0x3E)
HAIRLINE = RGBColor(0xD9, 0xD0, 0xC1)
PALE_BG  = RGBColor(0xFA, 0xF6, 0xEE)
WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
SOFT_GREEN = RGBColor(0x7B, 0xA3, 0x82)
SOFT_RED   = RGBColor(0xC9, 0x6F, 0x6A)

H_FONT = "Georgia"
B_FONT = "Calibri"

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height

BLANK = prs.slide_layouts[6]


# ---------- helpers ----------
def add_rect(slide, x, y, w, h, fill, line=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(0.75)
    return shp


def add_oval(slide, x, y, w, h, fill, line=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.OVAL, x, y, w, h)
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(0.75)
    return shp


def add_arrow(slide, x, y, w, h, fill):
    shp = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x, y, w, h)
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    return shp


def add_text(slide, x, y, w, h, text, *, font=B_FONT, size=14, bold=False,
             color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
             italic=False):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    if isinstance(text, str):
        runs = [(text, {})]
    else:
        runs = text
    for i, (t, opts) in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = opts.get("align", align)
        r = p.add_run()
        r.text = t
        r.font.name = opts.get("font", font)
        r.font.size = Pt(opts.get("size", size))
        r.font.bold = opts.get("bold", bold)
        r.font.italic = opts.get("italic", italic)
        r.font.color.rgb = opts.get("color", color)
    return tb


def set_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def content_header(slide, eyebrow, title):
    set_bg(slide, CREAM)
    add_rect(slide, Inches(0), Inches(0), Inches(0.18), SH, NAVY)
    add_text(slide, Inches(0.7), Inches(0.45), Inches(11), Inches(0.35),
             eyebrow, font=B_FONT, size=11, bold=True, color=CORAL)
    add_text(slide, Inches(0.7), Inches(0.78), Inches(12), Inches(0.8),
             title, font=H_FONT, size=30, bold=True, color=NAVY)


# =============================================================================
# SLIDE 1 — Title
# =============================================================================
s = prs.slides.add_slide(BLANK)
set_bg(s, NAVY)
add_rect(s, Inches(0), Inches(0), Inches(0.18), SH, CORAL)

add_text(s, Inches(0.9), Inches(1.1), Inches(11), Inches(0.4),
         "SWC-STUDIO", font=B_FONT, size=12, bold=True, color=CREAM)
add_text(s, Inches(0.9), Inches(1.7), Inches(11), Inches(2.2),
         "Auto-Labeling Neurons",
         font=H_FONT, size=64, bold=True, color=WHITE)
add_text(s, Inches(0.9), Inches(3.2), Inches(11), Inches(1.0),
         "A 3-stage pipeline for parsing neuron morphology",
         font=H_FONT, size=26, italic=True, color=CORAL)

add_rect(s, Inches(0.9), Inches(4.4), Inches(1.5), Inches(0.04), CREAM)

add_text(s, Inches(0.9), Inches(4.6), Inches(11), Inches(0.5),
         "Translating raw 3D neuron reconstructions into labeled morphological parts",
         font=B_FONT, size=16, color=CREAM)

add_text(s, Inches(0.9), Inches(6.6), Inches(11), Inches(0.4),
         "Latest evaluation  ·  1,737 cells  ·  pooled F1 0.98",
         font=B_FONT, size=13, color=MUTED)


# =============================================================================
# SLIDE 2 — Software & algorithm intro
# =============================================================================
s = prs.slides.add_slide(BLANK)
content_header(s, "INTRODUCING SWC-STUDIO", "A toolkit for working with neuron morphology files")

# Top blurb
add_text(s, Inches(0.7), Inches(2.0), Inches(12), Inches(0.9),
         [
             ("SWC-Studio", {"size":15, "bold":True, "color":NAVY}),
             (" is a desktop, CLI, and Python toolkit for inspecting neuron reconstructions, finding structural or annotation problems, repairing them, and running repeatable morphology workflows from one shared backend.",
              {"size":15, "color":INK}),
         ])

# Three form-factor cards
y0 = Inches(3.1)
form_h = Inches(1.85)
forms = [
    ("Desktop GUI", "swcstudio-gui",
     "Open a file, inspect issues visually, apply repairs, save the result.",
     CORAL),
    ("CLI", "swcstudio",
     "Batch-process whole folders. Same logic as the GUI, scriptable.",
     SAGE),
    ("Python library", "swcstudio",
     "Import the same backend in your own analysis pipelines.",
     GOLD),
]
cw = Inches(4.0)
gap = Inches(0.15)
for i, (name, cmd, body, color) in enumerate(forms):
    x = Inches(0.7) + i * (cw + gap)
    add_rect(s, x, y0, cw, form_h, WHITE)
    add_rect(s, x, y0, cw, Inches(0.10), color)
    add_text(s, x + Inches(0.25), y0 + Inches(0.2),
             cw - Inches(0.5), Inches(0.45),
             name, font=H_FONT, size=18, bold=True, color=NAVY)
    add_text(s, x + Inches(0.25), y0 + Inches(0.7),
             cw - Inches(0.5), Inches(0.35),
             cmd, font="Consolas", size=11, italic=True, color=color)
    add_text(s, x + Inches(0.25), y0 + Inches(1.1),
             cw - Inches(0.5), Inches(0.7),
             body, font=B_FONT, size=12, color=INK)

# Bottom: capabilities list (left) + "today's focus" callout (right)
y1 = Inches(5.2)
# Capabilities card
add_rect(s, Inches(0.7), y1, Inches(7.4), Inches(2.0), WHITE)
add_rect(s, Inches(0.7), y1, Inches(0.10), Inches(2.0), NAVY)
add_text(s, Inches(0.95), y1 + Inches(0.15),
         Inches(7.0), Inches(0.4),
         "CORE CAPABILITIES",
         font=B_FONT, size=11, bold=True, color=CORAL)
add_text(s, Inches(0.95), y1 + Inches(0.55),
         Inches(7.0), Inches(1.5),
         [
             ("•  Issue-driven SWC validation and repair",   {"size":12, "color":INK}),
             ("•  Rule-based auto-labeling with automatic apical detection",
                                                              {"size":12, "color":INK, "bold":True}),
             ("•  Radii cleaning and manual geometry editing",{"size":12, "color":INK}),
             ("•  Batch processing across folders",           {"size":12, "color":INK}),
             ("•  Shared GUI, CLI, and Python surface",       {"size":12, "color":INK}),
         ])

# Today's focus callout
add_rect(s, Inches(8.3), y1, Inches(4.4), Inches(2.0), NAVY)
add_text(s, Inches(8.55), y1 + Inches(0.15),
         Inches(4.0), Inches(0.4),
         "TODAY'S FOCUS",
         font=B_FONT, size=11, bold=True, color=CORAL)
add_text(s, Inches(8.55), y1 + Inches(0.55),
         Inches(4.0), Inches(0.6),
         "The auto-labeling engine",
         font=H_FONT, size=18, bold=True, color=WHITE)
add_text(s, Inches(8.55), y1 + Inches(1.15),
         Inches(4.0), Inches(0.85),
         "A 3-stage hybrid pipeline that combines machine learning with biology rules to label every point in a neuron — soma, axon, basal, or apical.",
         font=B_FONT, size=12, italic=True, color=CREAM)


# =============================================================================
# SLIDE 3 — The problem
# =============================================================================
s = prs.slides.add_slide(BLANK)
content_header(s, "THE PROBLEM", "Why we built this")

col_w = Inches(4.0)
col_h = Inches(4.6)
gap = Inches(0.2)
y0 = Inches(2.0)
start = Inches(0.7)
cols = [
    ("100,000+", "3D neuron reconstructions in public databases",
     "Neuroscientists trace neurons under the microscope. Each trace becomes a 3D point cloud — an SWC file.",
     CORAL),
    ("Many missing labels", "or labels that disagree between labs",
     "Some files only label the cell body. Others mark axon vs dendrite but disagree on apical. No single standard.",
     GOLD),
    ("Manual relabeling = months", "per dataset, not scalable",
     "An expert can re-label one cell in 10–30 minutes. Doing 1,000 cells by hand is a full project on its own.",
     SAGE),
]
for i, (big, sub, body, color) in enumerate(cols):
    x = start + (col_w + gap) * i
    add_rect(s, x, y0, col_w, col_h, WHITE)
    add_rect(s, x, y0, col_w, Inches(0.10), color)
    add_text(s, x + Inches(0.3), y0 + Inches(0.4), col_w - Inches(0.6), Inches(1.0),
             big, font=H_FONT, size=30, bold=True, color=NAVY)
    add_text(s, x + Inches(0.3), y0 + Inches(1.4), col_w - Inches(0.6), Inches(0.8),
             sub, font=B_FONT, size=14, bold=True, italic=True, color=color)
    add_text(s, x + Inches(0.3), y0 + Inches(2.2), col_w - Inches(0.6), Inches(2.2),
             body, font=B_FONT, size=12, color=INK)

add_rect(s, Inches(0.7), Inches(6.85), Inches(12), Inches(0.06), HAIRLINE)
add_text(s, Inches(0.7), Inches(6.95), Inches(12), Inches(0.45),
         [
             ("Our solution: ", {"size":14, "bold":True, "color":CORAL}),
             ("a fully automatic 3-stage pipeline that reads any SWC file and outputs a per-point label — soma, axon, dendrite, or apical.",
              {"size":14, "color":NAVY, "italic":True}),
         ])


# =============================================================================
# SLIDE 4 — Dataset
# =============================================================================
s = prs.slides.add_slide(BLANK)
content_header(s, "THE DATASET", "What we trained and tested on")

y = Inches(2.0)
stat_h = Inches(1.5)
def big_stat(x, val, label, color):
    add_rect(s, x, y, Inches(2.6), stat_h, WHITE)
    add_rect(s, x, y, Inches(2.6), Inches(0.10), color)
    add_text(s, x + Inches(0.2), y + Inches(0.3),
             Inches(2.2), Inches(0.7),
             val, font=H_FONT, size=36, bold=True, color=NAVY)
    add_text(s, x + Inches(0.2), y + Inches(1.0),
             Inches(2.2), Inches(0.4),
             label, font=B_FONT, size=12, color=INK)

big_stat(Inches(0.7), "1,737",   "labeled cells",      CORAL)
big_stat(Inches(3.5), "932",     "pyramidal cells",    SAGE)
big_stat(Inches(6.3), "805",     "interneurons",       SAGE)
big_stat(Inches(9.1), "5.4 M",   "individual points",  NAVY)

# Sources block
src_y = Inches(4.0)
add_rect(s, Inches(0.7), src_y, Inches(7.0), Inches(3.0), WHITE)
add_rect(s, Inches(0.7), src_y, Inches(0.10), Inches(3.0), CORAL)
add_text(s, Inches(0.95), src_y + Inches(0.2), Inches(6.5), Inches(0.4),
         "WHERE THE DATA COMES FROM",
         font=B_FONT, size=11, bold=True, color=CORAL)
add_text(s, Inches(0.95), src_y + Inches(0.65), Inches(6.5), Inches(2.4),
         [
             ("NeuroMorpho.org", {"size":13, "bold":True, "color":NAVY}),
             ("    Public archive — > 250,000 reconstructions across labs", {"size":12, "color":INK}),
             ("Allen Institute", {"size":13, "bold":True, "color":NAVY}),
             ("    Allen Cell Types Database (mouse cortex)", {"size":12, "color":INK}),
             ("Custom lab reconstructions", {"size":13, "bold":True, "color":NAVY}),
             ("    Internal benchmark cells with curated labels", {"size":12, "color":INK}),
         ])

# SWC file format box (made taller and more readable)
swc_y = Inches(4.0)
swc_h = Inches(3.0)
add_rect(s, Inches(7.9), swc_y, Inches(4.8), swc_h, WHITE)
add_rect(s, Inches(7.9), swc_y, Inches(0.10), swc_h, GOLD)
add_text(s, Inches(8.15), swc_y + Inches(0.2), Inches(4.4), Inches(0.4),
         "WHAT AN SWC FILE LOOKS LIKE",
         font=B_FONT, size=11, bold=True, color=GOLD)
add_text(s, Inches(8.15), swc_y + Inches(0.7), Inches(4.4), Inches(0.4),
         "Each row = one 3D point on the neuron",
         font=B_FONT, size=12, italic=True, color=INK)
# Code-like table area
add_rect(s, Inches(8.15), swc_y + Inches(1.15), Inches(4.4), Inches(1.2), PALE_BG)
add_text(s, Inches(8.3), swc_y + Inches(1.20), Inches(4.2), Inches(0.35),
         "id   type    x     y     z    r   parent",
         font="Consolas", size=11, bold=True, color=NAVY)
add_text(s, Inches(8.3), swc_y + Inches(1.55), Inches(4.2), Inches(0.35),
         "1     1    0.0   0.0   0.0  4.5    -1",
         font="Consolas", size=11, color=MUTED)
add_text(s, Inches(8.3), swc_y + Inches(1.85), Inches(4.2), Inches(0.35),
         "2     3    1.2   0.0  -0.5  0.8     1",
         font="Consolas", size=11, color=MUTED)
add_text(s, Inches(8.15), swc_y + Inches(2.45), Inches(4.4), Inches(0.4),
         "type:  1=soma   2=axon   3=basal   4=apical",
         font=B_FONT, size=12, italic=True, color=INK)


# =============================================================================
# SLIDE 5 — Pipeline at a glance
# =============================================================================
s = prs.slides.add_slide(BLANK)
content_header(s, "THE PIPELINE", "How a raw cell becomes a labeled cell")

stages = [
    ("01", "STAGE 1",
     "What kind of cell?",
     "Decide whether the whole cell is pyramidal or interneuron.",
     CORAL),
    ("02", "STAGE 2",
     "What is each branch?",
     "Classify every branch as axon, basal dendrite, or apical dendrite.",
     SAGE),
    ("03", "STAGE 3",
     "Apply biology rules",
     "Topology refinement — enforce one apical, consistent subtrees, etc.",
     GOLD),
]
box_w = Inches(3.5)
box_h = Inches(3.4)
gap = Inches(0.5)
y0 = Inches(2.2)
start_x = Inches(0.7)
for i, (num, tag, headline, body, color) in enumerate(stages):
    x = start_x + (box_w + gap) * i
    add_rect(s, x, y0, box_w, box_h, WHITE)
    add_rect(s, x, y0, Inches(0.10), box_h, color)
    add_text(s, x + Inches(0.3), y0 + Inches(0.25),
             Inches(2.0), Inches(0.6),
             num, font=H_FONT, size=36, bold=True, color=color)
    add_text(s, x + Inches(0.3), y0 + Inches(1.0),
             box_w - Inches(0.6), Inches(0.4),
             tag, font=B_FONT, size=11, bold=True, color=MUTED)
    add_text(s, x + Inches(0.3), y0 + Inches(1.4),
             box_w - Inches(0.6), Inches(0.8),
             headline, font=H_FONT, size=22, bold=True, color=NAVY)
    add_text(s, x + Inches(0.3), y0 + Inches(2.3),
             box_w - Inches(0.6), Inches(1.0),
             body, font=B_FONT, size=12, color=INK)

for i in range(len(stages) - 1):
    ax = start_x + (box_w + gap) * (i + 1) - gap + Inches(0.05)
    ay = y0 + box_h / 2 - Inches(0.18)
    add_arrow(s, ax, ay, Inches(0.4), Inches(0.36), MUTED)

add_text(s, Inches(0.7), Inches(6.0), Inches(12), Inches(0.4),
         "Pipeline as a whole",
         font=B_FONT, size=11, bold=True, color=CORAL)
add_text(s, Inches(0.7), Inches(6.4), Inches(12), Inches(0.7),
         [
             ("Input: ", {"size":14, "bold":True, "color":NAVY}),
             ("an unlabeled SWC file (just 3D coordinates and parent pointers).      ",
              {"size":14, "color":INK}),
             ("Output: ", {"size":14, "bold":True, "color":NAVY}),
             ("the same file, with every point labeled as soma, axon, basal, or apical.",
              {"size":14, "color":INK}),
         ])


# =============================================================================
# SLIDE 6 — Stage 1 overview
# =============================================================================
s = prs.slides.add_slide(BLANK)
content_header(s, "STAGE 1  ·  CELL-TYPE DETECTION", "What kind of cell are we looking at?")

add_text(s, Inches(0.7), Inches(2.0), Inches(7.0), Inches(0.5),
         "What it does",
         font=B_FONT, size=11, bold=True, color=CORAL)
add_text(s, Inches(0.7), Inches(2.4), Inches(7.0), Inches(2.0),
         [
             ("Reads the whole-cell shape of the neuron and predicts: is this a ",
              {"size":14, "color":INK}),
             ("pyramidal", {"size":14, "color":NAVY, "bold":True}),
             (" or an ",   {"size":14, "color":INK}),
             ("interneuron",{"size":14, "color":NAVY, "bold":True}),
             ("?  This decision controls which labels Stage 2 is allowed to use — interneurons have no apical dendrite.",
              {"size":14, "color":INK}),
         ])

add_text(s, Inches(0.7), Inches(4.0), Inches(7.0), Inches(0.5),
         "How it decides",
         font=B_FONT, size=11, bold=True, color=CORAL)
add_text(s, Inches(0.7), Inches(4.4), Inches(7.0), Inches(2.5),
         [
             ("Computes 49 shape descriptors of the whole cell. Categories:",
              {"size":13, "color":INK}),
             ("•  Size and counts  (branch points, terminals, subtrees)",
              {"size":12, "color":INK}),
             ("•  Spatial extent  (how far the cell reaches in x, y, z)",
              {"size":12, "color":INK}),
             ("•  Principal-axis (PCA) features  — the cell's intrinsic up/down direction",
              {"size":12, "color":INK, "bold":True}),
             ("•  Branching shape  (Strahler order, Sholl analysis, subtree ratios)",
              {"size":12, "color":INK}),
         ])

# NEW soft handoff box
right_x = Inches(8.2)
add_rect(s, right_x, Inches(2.0), Inches(4.5), Inches(4.7), NAVY)
add_text(s, right_x + Inches(0.3), Inches(2.2),
         Inches(4.0), Inches(0.4),
         "NEW  ·  SOFT HANDOFF",
         font=B_FONT, size=11, bold=True, color=CORAL)
add_text(s, right_x + Inches(0.3), Inches(2.6),
         Inches(4.0), Inches(0.8),
         "When Stage 1 isn't sure",
         font=H_FONT, size=22, bold=True, color=WHITE)
add_text(s, right_x + Inches(0.3), Inches(3.55),
         Inches(4.0), Inches(3.0),
         [
             ("If the model's confidence is below 65%, we don't commit to one cell type.",
              {"size":12, "color":CREAM}),
             ("Instead, the rest of the pipeline runs twice — once treating the cell as pyramidal, once as interneuron — and we keep whichever produces sharper, more confident predictions.",
              {"size":12, "color":CREAM}),
             ("This rescues borderline cells (e.g. flat slice-prep pyramidals) that would otherwise be locked into the wrong path.",
              {"size":12, "color":WHITE, "italic":True, "bold":True}),
         ])

add_text(s, Inches(0.7), Inches(7.0), Inches(12), Inches(0.4),
         [
             ("Latest accuracy: ", {"size":12, "bold":True, "color":NAVY}),
             ("98.2%", {"size":14, "bold":True, "color":CORAL}),
             ("   ·   trained with 5-fold cross-validation on 1,737 cells",
              {"size":12, "color":MUTED, "italic":True}),
         ])


# =============================================================================
# SLIDE 7 — Stage 1 features in detail [NEW]
# =============================================================================
s = prs.slides.add_slide(BLANK)
content_header(s, "STAGE 1  ·  IN DETAIL", "All 49 whole-cell features, by category")

# Six grouped lists in a 2x3 grid
groups = [
    ("Size & counts (7)", CORAL,
     ["n_nodes", "n_root_nodes", "n_branch_points",
      "n_terminals", "n_primary_subtrees",
      "branch_point_ratio", "terminal_ratio"]),
    ("Path & distance (7)", SAGE,
     ["max / mean / std path length",
      "max / mean / std radial distance",
      "path / radial ratio (tortuosity)"]),
    ("Radius (4)", GOLD,
     ["mean_radius", "std_radius",
      "max_radius", "radius_cv  (coefficient of variation)"]),
    ("Spatial extent (10)", NAVY,
     ["z_span, z_asymmetry, z_max above/below soma",
      "x_span, y_span, xy_aspect_ratio",
      "planarity, linearity, thickness_ratio"]),
    ("Subtree & topology (12)", SAGE,
     ["max / min / std primary subtree size",
      "max / std primary subtree path",
      "max subtree size & path ratios",
      "max & mean Strahler order",
      "Sholl: max intersections, peak, decay"]),
    ("Principal-axis (PCA) — NEW (7)", CORAL,
     ["pc1_span", "pc1_asymmetry",
      "pc1_max above / below soma",
      "pc1_radial_max",
      "subtree_pc1_concentration",
      "subtree_pc1_top_alignment"]),
]

cw = Inches(4.05)
ch = Inches(2.2)
gx = Inches(0.15)
gy = Inches(0.2)
y0 = Inches(2.0)
sx = Inches(0.7)
for i, (title, color, items) in enumerate(groups):
    row = i // 3
    col = i % 3
    x = sx + col * (cw + gx)
    y = y0 + row * (ch + gy)
    add_rect(s, x, y, cw, ch, WHITE)
    add_rect(s, x, y, Inches(0.08), ch, color)
    add_text(s, x + Inches(0.2), y + Inches(0.15),
             cw - Inches(0.3), Inches(0.35),
             title, font=B_FONT, size=12, bold=True, color=NAVY)
    body = [(itm, {"size":11, "color":INK}) for itm in items]
    add_text(s, x + Inches(0.2), y + Inches(0.55),
             cw - Inches(0.3), ch - Inches(0.6),
             body)

add_text(s, Inches(0.7), Inches(6.85), Inches(12), Inches(0.4),
         [
             ("Total: 49 file-level features.   ", {"size":12, "bold":True, "color":NAVY}),
             ("PCA features were added April 2026 to make the model robust to slice-prep distortion and rotated coordinate frames.",
              {"size":12, "italic":True, "color":INK}),
         ])


# =============================================================================
# SLIDE 8 — Stage 2 overview
# =============================================================================
s = prs.slides.add_slide(BLANK)
content_header(s, "STAGE 2  ·  PER-BRANCH LABELING", "What is each branch of this cell?")

add_text(s, Inches(0.7), Inches(2.0), Inches(12), Inches(0.5),
         "What it does",
         font=B_FONT, size=11, bold=True, color=CORAL)
add_text(s, Inches(0.7), Inches(2.4), Inches(12), Inches(1.6),
         [
             ("The cell is split into ", {"size":14, "color":INK}),
             ("branches", {"size":14, "color":NAVY, "bold":True}),
             (" — segments between branch points. For each branch, a machine-learning model predicts a label: ",
              {"size":14, "color":INK}),
             ("axon, basal dendrite, ", {"size":14, "color":SAGE, "bold":True}),
             ("or ", {"size":14, "color":INK}),
             ("apical dendrite", {"size":14, "color":CORAL, "bold":True}),
             (".  Each prediction comes with a confidence score (0–1).",
              {"size":14, "color":INK}),
         ])

y = Inches(4.4)
cards = [
    ("57 features per branch",
     "Length, tortuosity, depth from soma, neighbors' radii, principal-axis alignment, branching rate, etc.",
     SAGE),
    ("Cell-type-conditioned model",
     "Pyramidal cells use a model trained only on pyramidals. Interneurons use a separate model. Avoids confusion.",
     CORAL),
    ("Confidence per branch",
     "Each label comes with a probability. Stage 3 uses these to resolve conflicts and override low-confidence calls.",
     GOLD),
]
cw = Inches(4.0)
gap = Inches(0.15)
sx = Inches(0.7)
for i, (head, body, color) in enumerate(cards):
    x = sx + (cw + gap) * i
    add_rect(s, x, y, cw, Inches(2.4), WHITE)
    add_rect(s, x, y, Inches(0.10), Inches(2.4), color)
    add_text(s, x + Inches(0.25), y + Inches(0.2),
             cw - Inches(0.5), Inches(0.6),
             head, font=H_FONT, size=16, bold=True, color=NAVY)
    add_text(s, x + Inches(0.25), y + Inches(0.95),
             cw - Inches(0.5), Inches(1.4),
             body, font=B_FONT, size=12, color=INK)

add_text(s, Inches(0.7), Inches(7.0), Inches(12), Inches(0.4),
         [
             ("Latest pooled F1: ", {"size":12, "bold":True, "color":NAVY}),
             ("axon 0.997  ·  basal 0.97  ·  apical 0.94",
              {"size":12, "bold":True, "color":CORAL}),
         ])


# =============================================================================
# SLIDE 9 — Stage 2 features in detail [NEW]
# =============================================================================
s = prs.slides.add_slide(BLANK)
content_header(s, "STAGE 2  ·  IN DETAIL", "What the model sees about each branch")

groups = [
    ("Branch geometry (12)", CORAL,
     "Length, straightness (persistence), node count, radius statistics, taper, branchiness."),
    ("Position vs. soma (4)", SAGE,
     "Path & Euclidean distance to soma, height above soma, vertical span within the branch."),
    ("Subtree & order (4)", GOLD,
     "Subtree size, depth, longest path, branch order (depth from soma)."),
    ("Parent / sibling (4)", NAVY,
     "Parent radius, radius ratio, sibling count, whether the branch starts at the soma."),
    ("Cell-normalized (7)", SAGE,
     "Branch features divided by cell-level maxima, plus cell-type one-hot flags."),
    ("Apical-vs-basal (10)", CORAL,
     "Subtree z-rank, polar angles, trunk-primary flag, vertical/horizontal span ratio, longest-subtree indicator."),
    ("Axon signatures (4)", GOLD,
     "Starts-at-soma flag, radius drop at anchor, thin-segment fraction, minimum radius downstream."),
    ("Multi-scale local (4)", SAGE,
     "Mean radius and straightness over the proximal vs. distal third of the branch."),
    ("Principal-axis (PCA) — NEW (5)", CORAL,
     "Cell's intrinsic up direction. Projection, polar angle, alignment strength, subtree projection & rank."),
    ("Branching-rate — NEW (3)", GOLD,
     "Bifurcations per micron, mean internode distance, subtree-level branching density."),
]

# 4 cols x 3 rows so all 10 cards fit on the slide
cw = Inches(2.95)
ch = Inches(1.55)
gx = Inches(0.13)
gy = Inches(0.13)
y0 = Inches(2.0)
sx = Inches(0.7)
for i, (title, color, body) in enumerate(groups):
    row = i // 4
    col = i % 4
    x = sx + col * (cw + gx)
    y = y0 + row * (ch + gy)
    add_rect(s, x, y, cw, ch, WHITE)
    add_rect(s, x, y, Inches(0.08), ch, color)
    add_text(s, x + Inches(0.2), y + Inches(0.12),
             cw - Inches(0.3), Inches(0.7),
             title, font=B_FONT, size=11, bold=True, color=NAVY)
    add_text(s, x + Inches(0.2), y + Inches(0.55),
             cw - Inches(0.3), ch - Inches(0.6),
             body, font=B_FONT, size=10, color=INK)

add_text(s, Inches(0.7), Inches(7.0), Inches(12), Inches(0.4),
         [
             ("Total: 57 raw features per branch + 7 augmented from the subtree-owner model = 64 inputs.",
              {"size":12, "bold":True, "color":NAVY}),
         ])


# =============================================================================
# SLIDE 10 — Stage 3 overview
# =============================================================================
s = prs.slides.add_slide(BLANK)
content_header(s, "STAGE 3  ·  TOPOLOGY REFINEMENT", "Apply biology rules the ML can't learn alone")

add_text(s, Inches(0.7), Inches(2.0), Inches(12), Inches(0.5),
         "Why a final rule layer?",
         font=B_FONT, size=11, bold=True, color=CORAL)
add_text(s, Inches(0.7), Inches(2.4), Inches(12), Inches(1.0),
         "Stage 2 looks at one branch at a time and can flip-flop within a single dendrite. Stage 3 enforces simple biological constraints across the whole cell — fixing many of these errors.",
         font=B_FONT, size=14, italic=True, color=INK)

rules = [
    ("At most one apical",
     "A pyramidal cell has exactly one apical dendrite. If multiple subtrees are predicted apical, keep the strongest and demote the others.",
     CORAL),
    ("Subtree consistency",
     "All branches in the same primary subtree should agree on a label. Use majority voting weighted by confidence.",
     SAGE),
    ("PCA tie-break",
     "When picking the apical winner, prefer the subtree that aligns with the cell's principal axis (the natural up direction).",
     GOLD),
    ("Confidence overrides",
     "Low-confidence Stage 2 predictions can be overridden by stronger neighbors. Confident labels propagate downstream.",
     NAVY),
]
y0 = Inches(3.5)
cw = Inches(6.0)
ch = Inches(1.65)
gap_x = Inches(0.2)
gap_y = Inches(0.2)
for i, (title, body, color) in enumerate(rules):
    row = i // 2
    col = i % 2
    x = Inches(0.7) + col * (cw + gap_x)
    y = y0 + row * (ch + gap_y)
    add_rect(s, x, y, cw, ch, WHITE)
    add_rect(s, x, y, Inches(0.10), ch, color)
    add_text(s, x + Inches(0.25), y + Inches(0.15),
             cw - Inches(0.5), Inches(0.45),
             title, font=H_FONT, size=16, bold=True, color=NAVY)
    add_text(s, x + Inches(0.25), y + Inches(0.65),
             cw - Inches(0.5), Inches(1.0),
             body, font=B_FONT, size=12, color=INK)


# =============================================================================
# SLIDE 11 — Stage 3 rules in detail [NEW]
# =============================================================================
s = prs.slides.add_slide(BLANK)
content_header(s, "STAGE 3  ·  IN DETAIL", "Every rule the refinement layer applies, in order")

# Numbered list of rules
rules = [
    ("Subtree majority voting",
     "Within each primary subtree (a child branch of the soma), aggregate Stage 2's per-branch predictions into one label per subtree, weighted by confidence."),
    ("PCA-based apical picker (pyramidal only)",
     "If multiple primary subtrees are tagged apical, the winner is the one whose mean projection on the cell's principal axis is largest. Losers are demoted to basal."),
    ("Single-apical enforcement",
     "After PCA picking, only one primary subtree can carry the apical label. Remaining apical-tagged subtrees are reassigned to basal."),
    ("Apical-owner constraint",
     "Apical labels are only kept on branches inside the chosen apical-owner subtree. Stray apical predictions elsewhere are reset to basal."),
    ("Parent–child smoothing",
     "Walk along the tree and resolve label disagreements between a node and its parent using confidence weighting — this catches small flip-flops within long branches."),
    ("Island flipping",
     "Find isolated runs of one label surrounded by a different label and flip them, if the surrounding label is more confident."),
    ("Soft subtree majority (final pass)",
     "After all the above, run one more subtree-level vote with confidences smoothed across siblings. Stabilizes the final output."),
    ("Interneuron axon bias",
     "For interneurons (no apical possible), bias borderline thin descending dendrite branches toward axon, which matches the morphology."),
    ("Spurious soma cleanup",
     "Strip any single-node soma labels that are not connected to the proxy soma, removing tracing artefacts."),
]

y0 = Inches(2.0)
left_w = Inches(5.7)
right_w = Inches(6.9)
total_h = Inches(4.85)
add_rect(s, Inches(0.7), y0, Inches(12.0), total_h, WHITE)
# Two-column rule list
gap = Inches(0.1)
n = len(rules)
mid = (n + 1) // 2

def draw_rule(x, y, w, idx, title, body):
    # number badge
    add_oval(s, x + Inches(0.15), y + Inches(0.08),
             Inches(0.36), Inches(0.36), CORAL)
    add_text(s, x + Inches(0.15), y + Inches(0.10),
             Inches(0.36), Inches(0.32),
             str(idx), font=H_FONT, size=14, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER)
    add_text(s, x + Inches(0.65), y + Inches(0.08),
             w - Inches(0.7), Inches(0.32),
             title, font=B_FONT, size=12, bold=True, color=NAVY)
    add_text(s, x + Inches(0.65), y + Inches(0.40),
             w - Inches(0.7), Inches(0.6),
             body, font=B_FONT, size=10, color=INK)

row_h = Inches(1.0)
left_col_x = Inches(0.85)
right_col_x = Inches(7.05)
col_w = Inches(5.95)
inner_y = y0 + Inches(0.1)
for i, (title, body) in enumerate(rules[:mid]):
    draw_rule(left_col_x, inner_y + i * row_h, col_w, i + 1, title, body)
for j, (title, body) in enumerate(rules[mid:]):
    draw_rule(right_col_x, inner_y + j * row_h, col_w, mid + j + 1, title, body)

add_text(s, Inches(0.7), Inches(7.0), Inches(12), Inches(0.4),
         [
             ("Rules apply in order, top to bottom. ",
              {"size":12, "bold":True, "color":NAVY}),
             ("Each rule only changes labels when its evidence is stronger than what's already there.",
              {"size":12, "italic":True, "color":INK}),
         ])


# =============================================================================
# SLIDE 12 — Confusion matrix explained
# =============================================================================
s = prs.slides.add_slide(BLANK)
content_header(s, "HOW WE MEASURE SUCCESS", "Reading a confusion matrix")

# Left explanation: shrunk to width 5.6 to clear the matrix at x=7.4
add_text(s, Inches(0.7), Inches(2.0), Inches(5.6), Inches(0.5),
         "What is it?",
         font=B_FONT, size=11, bold=True, color=CORAL)
add_text(s, Inches(0.7), Inches(2.4), Inches(5.6), Inches(2.4),
         [
             ("A table that compares ", {"size":13, "color":INK}),
             ("ground truth", {"size":13, "color":NAVY, "bold":True}),
             (" (what an expert labeled) against ", {"size":13, "color":INK}),
             ("model predictions", {"size":13, "color":NAVY, "bold":True}),
             (".  Rows = the truth.  Columns = what the model said.",
              {"size":13, "color":INK}),
             ("",                                                  {"size":13, "color":INK}),
             ("Diagonal cells = correct. Off-diagonal cells = mistakes — the column tells you ",
              {"size":12, "color":INK}),
             ("which kind of mistake.", {"size":12, "color":INK, "bold":True}),
         ])

add_text(s, Inches(0.7), Inches(5.4), Inches(5.6), Inches(0.5),
         "Reading the example →",
         font=B_FONT, size=11, bold=True, color=CORAL)
add_text(s, Inches(0.7), Inches(5.8), Inches(5.6), Inches(1.6),
         [
             ("230,241 apical points correctly labeled apical.",
              {"size":12, "color":INK}),
             ("5,102 true apical points were called basal.",
              {"size":12, "color":INK}),
             ("14,199 true basal points were called apical.",
              {"size":12, "color":INK}),
             ("These two off-diagonals are our remaining bottleneck.",
              {"size":12, "color":CORAL, "italic":True, "bold":True}),
         ])

# Matrix shifted right (cm_x = 8.6 with row labels at x=7.65)
cm_x = Inches(8.6)
cm_y = Inches(2.3)
cell_w = Inches(0.95)
cell_h = Inches(0.85)
n_classes = 4
labels = ["soma", "axon", "basal", "apical"]
matrix = [
    [328,    0,        0,       0],
    [0, 4360649,    2550,    1134],
    [0,    9575,  543075,   14199],
    [0,   10765,    5102,  230241],
]

# Column header
add_text(s, cm_x, cm_y - Inches(0.7),
         cell_w * n_classes, Inches(0.4),
         "MODEL PREDICTED →", font=B_FONT, size=10, bold=True,
         color=MUTED, align=PP_ALIGN.CENTER)
for j, lbl in enumerate(labels):
    add_text(s, cm_x + cell_w * j, cm_y - Inches(0.3),
             cell_w, Inches(0.3),
             lbl, font=B_FONT, size=11, bold=True, color=NAVY,
             align=PP_ALIGN.CENTER)

# Side label rotated visually with text only
add_text(s, cm_x - Inches(2.0), cm_y + Inches(1.5),
         Inches(1.1), Inches(0.5),
         "TRUTH ↓", font=B_FONT, size=10, bold=True, color=MUTED,
         align=PP_ALIGN.RIGHT)

def fmt(v):
    if v >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v/1000:.0f}k"
    return str(v)

for i, row in enumerate(matrix):
    add_text(s, cm_x - Inches(0.95), cm_y + cell_h * i + Inches(0.25),
             Inches(0.85), Inches(0.4),
             labels[i], font=B_FONT, size=11, bold=True, color=NAVY,
             align=PP_ALIGN.RIGHT)
    for j, val in enumerate(row):
        is_diag = (i == j)
        if val == 0:
            fill = PALE_BG
            txt_color = MUTED
        elif is_diag:
            fill = SOFT_GREEN
            txt_color = WHITE
        else:
            fill = SOFT_RED
            txt_color = WHITE
        add_rect(s, cm_x + cell_w * j, cm_y + cell_h * i,
                 cell_w, cell_h, fill, line=HAIRLINE)
        add_text(s, cm_x + cell_w * j, cm_y + cell_h * i + Inches(0.18),
                 cell_w, Inches(0.5),
                 fmt(val), font=B_FONT, size=12, bold=True,
                 color=txt_color, align=PP_ALIGN.CENTER)

# Legend
leg_y = cm_y + cell_h * n_classes + Inches(0.3)
add_rect(s, cm_x, leg_y, Inches(0.25), Inches(0.25), SOFT_GREEN)
add_text(s, cm_x + Inches(0.35), leg_y, Inches(2.0), Inches(0.3),
         "correct", font=B_FONT, size=11, color=INK)
add_rect(s, cm_x + Inches(1.7), leg_y, Inches(0.25), Inches(0.25), SOFT_RED)
add_text(s, cm_x + Inches(2.05), leg_y, Inches(2.0), Inches(0.3),
         "mistake", font=B_FONT, size=11, color=INK)


# =============================================================================
# SLIDE 13 — Key terms
# =============================================================================
s = prs.slides.add_slide(BLANK)
content_header(s, "KEY TERMS", "Three numbers we use to score the model")

defs = [
    ("Precision",
     "When we say apical, are we right?",
     "Of all the points the model labeled apical, what fraction are actually apical?",
     "High precision → few false alarms",
     CORAL),
    ("Recall",
     "Did we catch all the real apicals?",
     "Of all the points that are truly apical, what fraction did the model find?",
     "High recall → few missed apicals",
     SAGE),
    ("F1 score",
     "A balance of the two",
     "A single score (0 to 1) that combines precision and recall. We use F1 because either alone can be gamed.",
     "F1 = 1 means perfect labeling",
     GOLD),
]
cw = Inches(4.0)
ch = Inches(4.4)
gap = Inches(0.15)
y0 = Inches(2.0)
sx = Inches(0.7)
for i, (term, q, definition, takeaway, color) in enumerate(defs):
    x = sx + (cw + gap) * i
    add_rect(s, x, y0, cw, ch, WHITE)
    add_rect(s, x, y0, cw, Inches(0.10), color)
    add_text(s, x + Inches(0.3), y0 + Inches(0.3),
             cw - Inches(0.6), Inches(0.7),
             term, font=H_FONT, size=28, bold=True, color=NAVY)
    add_text(s, x + Inches(0.3), y0 + Inches(1.1),
             cw - Inches(0.6), Inches(0.6),
             q, font=H_FONT, size=15, italic=True, color=color)
    add_text(s, x + Inches(0.3), y0 + Inches(1.9),
             cw - Inches(0.6), Inches(2.0),
             definition, font=B_FONT, size=12, color=INK)
    add_rect(s, x + Inches(0.3), y0 + Inches(3.6),
             cw - Inches(0.6), Inches(0.02), HAIRLINE)
    add_text(s, x + Inches(0.3), y0 + Inches(3.7),
             cw - Inches(0.6), Inches(0.6),
             takeaway, font=B_FONT, size=11, italic=True, bold=True, color=color)

add_text(s, Inches(0.7), Inches(6.7), Inches(12), Inches(0.4),
         "TWO WAYS WE AVERAGE",
         font=B_FONT, size=11, bold=True, color=CORAL)
add_text(s, Inches(0.7), Inches(7.05), Inches(12), Inches(0.4),
         [
             ("Pooled F1: ",     {"size":12, "bold":True, "color":NAVY}),
             ("treats every point in the dataset equally — bigger cells contribute more.   ",
              {"size":12, "color":INK}),
             ("Per-file F1: ",   {"size":12, "bold":True, "color":NAVY}),
             ("scores each cell separately, then averages — every cell counts the same.",
              {"size":12, "color":INK}),
         ])


# =============================================================================
# SLIDE 14 — Latest results
# =============================================================================
s = prs.slides.add_slide(BLANK)
content_header(s, "LATEST RESULTS", "How the pipeline performs today")

y = Inches(2.0)
stat_h = Inches(1.55)
stats = [
    ("0.992", "overall accuracy",   "fraction of all points correct",  NAVY),
    ("0.976", "F1 (pooled)",        "weighted across all points",      NAVY),
    ("0.963", "F1 (per-file mean)", "averaged across cells equally",   NAVY),
    ("0.94",  "F1 (apical)",        "the bottleneck class",            CORAL),
]
for i, (val, label, sub, color) in enumerate(stats):
    x = Inches(0.7) + i * Inches(3.05)
    add_rect(s, x, y, Inches(2.85), stat_h, WHITE)
    add_rect(s, x, y, Inches(2.85), Inches(0.10), color)
    add_text(s, x + Inches(0.25), y + Inches(0.25),
             Inches(2.5), Inches(0.7),
             val, font=H_FONT, size=36, bold=True, color=NAVY)
    add_text(s, x + Inches(0.25), y + Inches(0.95),
             Inches(2.5), Inches(0.4),
             label, font=B_FONT, size=12, bold=True, color=INK)
    add_text(s, x + Inches(0.25), y + Inches(1.25),
             Inches(2.5), Inches(0.3),
             sub, font=B_FONT, size=10, italic=True, color=MUTED)

add_text(s, Inches(0.7), Inches(3.9), Inches(12), Inches(0.4),
         "PER-CLASS F1", font=B_FONT, size=11, bold=True, color=CORAL)

bar_x = Inches(2.6)
bar_max = Inches(8.4)
row_h = Inches(0.55)
top = Inches(4.3)

def scale(v, lo=0.85, hi=1.00):
    frac = max(0.0, (v - lo) / (hi - lo))
    return Emu(int(bar_max * frac))

tick_y = top + row_h * 4 + Inches(0.05)
add_rect(s, bar_x, tick_y, bar_max, Inches(0.02), HAIRLINE)
for v in [0.85, 0.90, 0.95, 1.00]:
    tx = bar_x + scale(v)
    add_rect(s, tx, tick_y, Inches(0.02), Inches(0.10), MUTED)
    add_text(s, tx - Inches(0.3), tick_y + Inches(0.13),
             Inches(0.6), Inches(0.3),
             f"{v:.2f}", font=B_FONT, size=10, color=MUTED, align=PP_ALIGN.CENTER)

classes = [
    ("Soma",         1.000, SAGE),
    ("Axon",         0.997, SAGE),
    ("Basal",        0.972, SAGE),
    ("Apical",       0.937, CORAL),
]
for i, (name, f1, color) in enumerate(classes):
    yi = top + row_h * i
    add_text(s, Inches(0.7), yi + Inches(0.08),
             Inches(1.85), Inches(0.45),
             name, font=B_FONT, size=14, bold=True, color=INK,
             align=PP_ALIGN.RIGHT)
    add_rect(s, bar_x, yi + Inches(0.13), bar_max, Inches(0.28), HAIRLINE)
    w = scale(f1)
    if w > 0:
        add_rect(s, bar_x, yi + Inches(0.13), w, Inches(0.28), color)
    add_text(s, bar_x + w + Inches(0.1), yi + Inches(0.08),
             Inches(1.0), Inches(0.45),
             f"{f1:.3f}", font=B_FONT, size=13, bold=True, color=INK)

add_rect(s, Inches(0.7), Inches(6.95), Inches(12), Inches(0.04), HAIRLINE)
add_text(s, Inches(0.7), Inches(7.05), Inches(12), Inches(0.4),
         [
             ("Takeaway: ", {"size":13, "bold":True, "color":CORAL}),
             ("Soma and axon are essentially solved. Apical and basal — both dendrites — are the remaining work.",
              {"size":13, "italic":True, "color":NAVY}),
         ])


# =============================================================================
# SLIDE 15 — What's next
# =============================================================================
s = prs.slides.add_slide(BLANK)
set_bg(s, NAVY)
add_rect(s, Inches(0), Inches(0), Inches(0.18), SH, CORAL)

add_text(s, Inches(0.9), Inches(1.2), Inches(11), Inches(0.4),
         "WHAT'S NEXT", font=B_FONT, size=12, bold=True, color=CORAL)
add_text(s, Inches(0.9), Inches(1.7), Inches(11), Inches(1.3),
         "Closing the apical gap",
         font=H_FONT, size=44, bold=True, color=WHITE)

items = [
    ("Trunk-detection features",
     "Apical = one dominant trunk before branching. Basal = bushy from the start. Add features that capture trunk length and dominance directly."),
    ("Dedicated apical-vs-basal head",
     "Specialize a smaller model on just the dendrite-vs-dendrite decision, with features that target that specific contrast."),
    ("Handle remaining edge cases",
     "After dataset cleanup, the worst-scoring files are atypical-but-valid morphology, plus rare cases like z-axis-inverted reconstructions worth flagging at load time."),
]
y0 = Inches(3.5)
ih = Inches(0.95)
for i, (title, body) in enumerate(items):
    yi = y0 + i * ih
    add_oval(s, Inches(0.9), yi + Inches(0.1),
             Inches(0.45), Inches(0.45), CORAL)
    add_text(s, Inches(0.9), yi + Inches(0.13),
             Inches(0.45), Inches(0.4),
             str(i + 1), font=H_FONT, size=20, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER)
    add_text(s, Inches(1.6), yi, Inches(11), Inches(0.45),
             title, font=H_FONT, size=18, bold=True, color=WHITE)
    add_text(s, Inches(1.6), yi + Inches(0.4), Inches(11), Inches(0.5),
             body, font=B_FONT, size=12, color=CREAM, italic=True)

add_rect(s, Inches(0.9), Inches(6.9), Inches(11.5), Inches(0.04), CREAM)
add_text(s, Inches(0.9), Inches(7.0), Inches(11.5), Inches(0.4),
         "Goal: lift apical F1 from 0.94 to ≥ 0.95 — and produce a labeling tool the field can rely on.",
         font=H_FONT, size=14, italic=True, color=CORAL)


# =============================================================================
out = "/Users/tuo/Desktop/SWC-Studio/paper/results_summary.pptx"
prs.save(out)
print("WROTE", out)
