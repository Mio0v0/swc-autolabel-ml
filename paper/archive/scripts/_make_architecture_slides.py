#!/usr/bin/env python3
"""Generate a 3-slide deck explaining the current pipeline architecture
and dataset.

Slide 1: Pipeline architecture (4-stage hybrid + ensemble flag layer)
Slide 2: Dataset (v12_uncurated corpus + QC + train/test split)
Slide 3: Training configuration (class weighting + seed ensemble)

Output: D:/Desktop/auto_typing_architecture.pptx
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor


OUT = Path("D:/Desktop/auto_typing_architecture.pptx")

# Palette
NAVY   = RGBColor(0x1F, 0x3A, 0x68)
ACCENT = RGBColor(0x2E, 0x86, 0xAB)
GREEN  = RGBColor(0x2E, 0x7D, 0x32)
ORANGE = RGBColor(0xE6, 0x7E, 0x22)
RED    = RGBColor(0xC0, 0x39, 0x2B)
GREY   = RGBColor(0x55, 0x55, 0x55)
LIGHT  = RGBColor(0xF4, 0xF6, 0xF8)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)

S1_FILL = RGBColor(0x2E, 0x86, 0xAB)   # blue
S2_FILL = RGBColor(0x2E, 0x7D, 0x32)   # green
S3_FILL = RGBColor(0x8E, 0x44, 0xAD)   # purple
S4_FILL = RGBColor(0xE6, 0x7E, 0x22)   # orange
FLAG_FILL = RGBColor(0xC0, 0x39, 0x2B) # red


def _style(run, *, size=14, bold=False, color=None):
    run.font.size = Pt(size)
    run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color


def _title(slide, text):
    tf = slide.shapes.title.text_frame
    tf.text = text
    _style(tf.paragraphs[0].runs[0], size=30, bold=True, color=NAVY)


def _text(slide, left, top, width, height, lines):
    box = slide.shapes.add_textbox(Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(lines):
        if isinstance(item, str):
            text, level, bold, color, size = item, 0, False, GREY, 14
        else:
            text, level, bold, color, size = item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = level
        if not text:
            continue
        run = p.add_run()
        run.text = text
        _style(run, size=size, bold=bold, color=color)
    return box


def _box(slide, left, top, width, height, label, *,
         fill=ACCENT, text_color=WHITE, title_size=14,
         body_lines=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    shp = slide.shapes.add_shape(shape, Inches(left), Inches(top),
                                  Inches(width), Inches(height))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = fill
    tf = shp.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.08); tf.margin_right = Inches(0.08)
    tf.margin_top = Inches(0.05);  tf.margin_bottom = Inches(0.05)
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.text = label
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    _style(tf.paragraphs[0].runs[0], size=title_size, bold=True, color=text_color)
    if body_lines:
        for ln in body_lines:
            p = tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = ln
            _style(r, size=10, color=text_color)
    return shp


def _arrow(slide, x1, y1, x2, y2, *, color=NAVY, weight=2.0):
    line = slide.shapes.add_connector(1, Inches(x1), Inches(y1),
                                       Inches(x2), Inches(y2))
    line.line.color.rgb = color
    line.line.width = Pt(weight)
    # Add arrowhead
    from pptx.oxml.ns import qn
    ln = line.line._get_or_add_ln()
    tailEnd = ln.find(qn('a:tailEnd'))
    if tailEnd is None:
        from lxml import etree
        tailEnd = etree.SubElement(ln, qn('a:tailEnd'))
    tailEnd.set('type', 'triangle')
    tailEnd.set('w', 'med')
    tailEnd.set('h', 'med')
    return line


def _table(slide, left, top, width, height, headers, rows,
           *, header_fill=NAVY, header_text=WHITE,
           col_widths_in=None, first_col_bold=True, body_size=11):
    n_cols = len(headers); n_rows = len(rows) + 1
    shp = slide.shapes.add_table(n_rows, n_cols,
                                  Inches(left), Inches(top),
                                  Inches(width), Inches(height))
    tbl = shp.table
    if col_widths_in is not None:
        for j, w in enumerate(col_widths_in):
            tbl.columns[j].width = Inches(w)
    for j, h in enumerate(headers):
        c = tbl.cell(0, j); c.text = h
        c.fill.solid(); c.fill.fore_color.rgb = header_fill
        for p in c.text_frame.paragraphs:
            for r in p.runs:
                _style(r, size=11, bold=True, color=header_text)
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            c = tbl.cell(i, j); c.text = str(val)
            for p in c.text_frame.paragraphs:
                for r in p.runs:
                    _style(r, size=body_size,
                           bold=(j == 0 and first_col_bold),
                           color=GREY)
    return tbl


# ----------------------------------------------------------------------
def build():
    prs = Presentation()
    prs.slide_width  = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # =============================================================
    # SLIDE 1 — Full architecture overview (QC + 4-stage + ensemble flag)
    # =============================================================
    s = prs.slides.add_slide(prs.slide_layouts[5])
    _title(s, "End-to-end architecture — QC gate + 4-stage pipeline + ensemble flag")

    # ---- Row A: Front-end (input → normalize → QC, with reject branch) ----
    # Input
    _box(s, 0.2, 1.3, 1.4, 0.9, "Input SWC",
         fill=GREY, title_size=12,
         body_lines=["(id, type, x, y,", "z, r, parent)"])
    # Parse + normalize
    _box(s, 1.9, 1.3, 2.0, 0.9, "Parse + normalize",
         fill=GREY, title_size=12,
         body_lines=["custom types 5+ →",
                     "branch-dominant {1,2,3,4}"])
    # QC gate
    _box(s, 4.2, 1.3, 2.4, 0.9, "QC GATE",
         fill=RED, title_size=13,
         body_lines=["A. structural checks",
                     "B. OOD (Mahalanobis on S1 feats)"])
    # Reject branch
    _box(s, 4.85, 2.45, 1.1, 0.5, "REJECT",
         fill=GREY, title_size=11)
    _arrow(s, 5.4, 2.2, 5.4, 2.45, color=RED, weight=1.5)

    # PASS arrow into Stage 1
    _arrow(s, 1.6, 1.75, 1.9, 1.75)
    _arrow(s, 3.9, 1.75, 4.2, 1.75)
    _arrow(s, 6.6, 1.75, 6.95, 1.75)

    # ---- Row B: Core 4-stage pipeline ----
    _box(s, 6.95, 1.3, 1.55, 0.9, "STAGE 1\nCell-type RF",
         fill=S1_FILL, title_size=12,
         body_lines=["XGBoost (GPU)", "→ pyr | inter"])
    _box(s, 8.65, 1.3, 1.55, 0.9, "STAGE 2\nBranch RF",
         fill=S2_FILL, title_size=12,
         body_lines=["XGBoost (GPU)", "soma/ax/dend/api"])
    _box(s, 10.35, 1.3, 1.55, 0.9, "STAGE 3\nGNN",
         fill=S3_FILL, title_size=12,
         body_lines=["GraphSAGE", "apical-vs-basal"])
    _box(s, 12.05, 1.3, 1.15, 0.9, "STAGE 4\nTopology",
         fill=S4_FILL, title_size=12,
         body_lines=["rule cleanup"])
    _arrow(s, 8.5, 1.75, 8.65, 1.75)
    _arrow(s, 10.2, 1.75, 10.35, 1.75)
    _arrow(s, 11.9, 1.75, 12.05, 1.75)

    # "Per-cell labels (from one model)" output
    _text(s, 6.95, 2.25, 6.25, 0.4, [
        ("↓ per-node labels (one model)", 0, True, GREY, 11),
    ])

    # ---- Row C: Ensemble (the 4-stage pipeline runs 5× in parallel) ----
    _box(s, 0.2, 3.15, 13.0, 0.5,
         "ENSEMBLE — the QC gate + 4-stage pipeline above runs in parallel for 5 seed-different models",
         fill=NAVY, title_size=13)

    seeds = [
        ("seed=42",  "v12_gentle_seed42"),
        ("seed=123", "v12_gentle_seed123"),
        ("seed=456", "v12_gentle_seed456"),
        ("seed=789", "v12_gentle_seed789"),
    ]
    n_seeds = len(seeds)
    seed_w  = 12.6 / n_seeds  # spread across the row
    for i, (tag, sub) in enumerate(seeds):
        _box(s, 0.3 + i * seed_w, 3.8, seed_w - 0.2, 0.85,
             f"{tag}\n{sub}", fill=ACCENT, title_size=11)

    # Arrows from each model down to aggregation
    for i in range(n_seeds):
        _arrow(s, 0.3 + (i + 0.5) * seed_w, 4.65, 6.7, 5.05, color=GREY, weight=1)

    # ---- Row D: Aggregation + flag ----
    _box(s, 0.2, 5.05, 6.3, 1.15, "Per-node majority vote",
         fill=NAVY, title_size=13,
         body_lines=[f"disagreement = 1 − (top-class votes / {n_seeds})",
                     "Spearman(disagree, wrong) ≈ 0.4 (vs 0.04 for softmax)"])
    _box(s, 6.7, 5.05, 6.5, 1.15, "Output — labels + per-branch flag",
         fill=GREEN, title_size=13,
         body_lines=["GREEN  unanimous → accept label",
                     "YELLOW mild disagree → review",
                     "RED    heavy disagree (≥0.5) → reject  (precision 90%)"])
    _arrow(s, 6.5, 5.6, 6.7, 5.6)

    # Bottom caption
    _text(s, 0.2, 6.35, 13.0, 1.0, [
        ("Two QC layers exist: (1) corpus-build QC filters training data offline; "
         "(2) inference-time QC gate above rejects unlabelable input SWCs.",
         0, False, GREY, 12),
        ("OOD detector = Mahalanobis distance on Stage 1 feature vector, threshold = "
         "99th percentile of training-set distances.",
         0, False, GREY, 12),
        ("Inference: ~2.7 s/cell per model on RTX 4080. All 5 models are independent "
         "and run embarrassingly parallel.",
         0, False, GREY, 12),
    ])

    # =============================================================
    # SLIDE 2 — Dataset
    # =============================================================
    s = prs.slides.add_slide(prs.slide_layouts[5])
    _title(s, "Dataset — v12_uncurated corpus")

    # Source breakdown table (cleaner numbers)
    _text(s, 0.3, 1.3, 12.7, 0.5, [
        ("Three data sources, two cell types — every cell labeled by humans before us.",
         0, True, NAVY, 16),
    ])
    _table(s, 0.3, 1.85, 7.6, 2.2,
           ["Source", "Pyramidal scanned → QC-pass",
            "Interneuron scanned → QC-pass", "Total pass"],
           [
               ["NeuroMorpho.Org", "9,998 → 7,623", "9,999 → 3,303", "10,926"],
               ["Allen Brain Atlas", "104 → 79",     "198 → 127",     "206"],
               ["hpf_ca1 (lab)",    "1,377 → 1,358", "—",             "1,358"],
               ["TOTAL",            "11,479 → 9,060", "10,197 → 3,430", "12,490"],
           ],
           col_widths_in=[2.0, 2.2, 2.2, 1.2], body_size=11)

    # QC criteria
    _text(s, 8.1, 1.85, 5.0, 4.0, [
        ("QC gates (strict)", 0, True, NAVY, 16),
        ("structural integrity", 1, False, GREY, 13),
        ("≥30 nodes total", 1, False, GREY, 13),
        ("connectivity (single soma root)", 1, False, GREY, 13),
        ("max branch order ≤30", 1, False, GREY, 13),
        ("plausible coordinate range", 1, False, GREY, 13),
        ("", 0, False, None, 6),
        ("SWC type normalization", 0, True, NAVY, 16),
        ("non-standard types (5+) absorbed into", 1, False, GREY, 13),
        ("branch-dominant {1,2,3,4} type",        1, False, GREY, 13),
        ("recovers 534 lab files (60%→99% pass)", 1, True, GREEN, 13),
        ("", 0, False, None, 6),
        ("Soma proxy detection is label-free", 0, True, NAVY, 16),
        ("biggest-radius root, NOT type==1",  1, False, GREY, 13),
        ("(prevents leaking GT into features)", 1, False, GREY, 13),
    ])

    # Split panel
    _text(s, 0.3, 4.2, 7.6, 0.4, [
        ("Hash-bucket train/test split (deterministic by md5(seed:filename)):",
         0, True, NAVY, 15),
    ])
    _table(s, 0.3, 4.65, 7.6, 1.6,
           ["Split", "Pyramidal", "Interneuron", "Total"],
           [
               ["Train (80%)", "7,284", "2,738", "10,022"],
               ["Test  (20%)", "1,776",   "692", "2,468"],
               ["TOTAL",       "9,060", "3,430", "12,490"],
           ],
           col_widths_in=[2.0, 1.8, 1.8, 2.0], body_size=12)

    _text(s, 0.3, 6.4, 13.0, 1.0, [
        ("Class imbalance is severe — ~30% of all neurite nodes are apical/dendrite, "
         "~10% are axon, ~60% basal-dendrite.",
         0, False, GREY, 13),
        ("Handled by class-weighted training (see next slide), not by resampling.",
         0, False, GREY, 13),
        ("Test set is locked & never seen during model selection.",
         0, True, GREEN, 13),
    ])

    # =============================================================
    # SLIDE 3 — Training configuration
    # =============================================================
    s = prs.slides.add_slide(prs.slide_layouts[5])
    _title(s, "Training configuration — class weighting + 5-seed ensemble")

    _text(s, 0.3, 1.25, 13.0, 0.5, [
        ("Stage 2 (per-branch RF) and Stage 3 (GNN) both use class-weighted "
         "cross-entropy. Weights are computed per-fold from the training set.",
         0, False, GREY, 14),
    ])

    # Hyperparameter table
    _table(s, 0.3, 1.85, 12.7, 2.0,
           ["Stage", "Loss / weighting", "Variant chosen", "Env var"],
           [
               ["Stage 1 — Cell-type RF",
                "balanced (XGBoost sample_weight)",
                "default (no env)",
                "—"],
               ["Stage 2 — Branch RF",
                "node-balanced  w_c ∝ N_c^(−power)",
                "power = 1.25  (gentle)",
                "SWCAL_CLASS_BALANCE_POWER=1.25"],
               ["Stage 3 — GraphSAGE GNN",
                "weighted CE  w_c ∝ N_c^(−½)",
                "inverse_sqrt",
                "SWCAL_GNN_CLASS_WEIGHT=inverse_sqrt"],
           ],
           col_widths_in=[3.0, 4.0, 2.7, 3.0], body_size=12)

    # Why these variants
    _text(s, 0.3, 4.0, 6.3, 3.3, [
        ("Why 'gentle' (power=1.25) not aggressive (1.5)?", 0, True, NAVY, 15),
        ("Aggressive (power=1.5) boosts P10 (worst-decile)", 1, False, GREY, 12),
        ("by +2.4 pp, but apical F1 drops by 3.1 pp",         1, False, GREY, 12),
        ("(over-predicts apical in interneurons).",            1, False, GREY, 12),
        ("Gentle is the dominant choice on every metric.",     1, True,  GREEN, 12),
        ("", 0, False, None, 8),
        ("Why ensemble over single best model?", 0, True, NAVY, 15),
        ("Per-node confidence is a poor flag signal",       1, False, GREY, 12),
        ("(Stage 3 overrides Stage 2 labels but not confidence)", 1, False, GREY, 12),
        ("Disagreement between seeds is a post-refinement",  1, False, GREY, 12),
        ("confidence signal: Spearman 0.40 vs wrongness,",   1, False, GREY, 12),
        ("vs 0.04 for single-model softmax.",                 1, True, GREEN, 12),
    ])

    # Ensemble composition
    _text(s, 6.8, 4.0, 6.3, 3.3, [
        ("5-model seed ensemble", 0, True, NAVY, 15),
        ("All 5 models use IDENTICAL config:",  1, False, GREY, 12),
        ("  power=1.25 + inverse_sqrt",         1, False, GREY, 12),
        ("Only random seed differs.",            1, False, GREY, 12),
        ("", 0, False, None, 6),
        ("Seeds 0/123/456/789 — vanilla random split", 1, False, GREY, 12),
        ("Seed 'clean_holdout' — explicit test set =",  1, False, GREY, 12),
        ("  24 cells that were in TEST for all other 4", 1, False, GREY, 12),
        ("  models. Guarantees zero contamination on",  1, False, GREY, 12),
        ("  the methodologically clean holdout eval.",  1, True, GREEN, 12),
        ("", 0, False, None, 6),
        ("Wall-clock: ~90 min per model on RTX 4080,",  1, False, GREY, 12),
        ("trained sequentially. Inference parallelizable.", 1, False, GREY, 12),
    ])

    # ---- Save
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"Wrote {OUT}")
    print(f"  size: {OUT.stat().st_size / 1024:.1f} KB")
    print(f"  slides: {len(prs.slides)}")


if __name__ == "__main__":
    build()
