#!/usr/bin/env python3
"""Compile every result snapshot into one paper-ready comparison table.

Reads the existing per-row JSON snapshots and the external-baselines /
significance-tests outputs, then writes:

    paper/results/paper_table.txt           plain-text Markdown-like table
    paper/results/paper_table_full.json     machine-readable union

Sections:

    1. Floor baselines               (random / majority / heuristic from
                                      paper/results/baselines_results.json)

    2. External baselines            (NeuroM-RF, Sholl-RF, Sholl-MLP from
                                      paper/results/external_baselines_results.json
                                      — Emissah/Ascoli-style)

    3. Architecture progression      (v6 / v7 / v8 / v9_no_gnn / v9 from
                                      paper/results/snapshots/v*.json)

    4. Statistical tests             (paired Wilcoxon vs v9 final from
                                      paper/results/snapshots/significance_tests.json)

The table is the row reviewers actually want — every method on the same
test split, with the same headline metrics: accuracy, macro-F1, neurite-
macro-F1, per-class F1, per-file mean / P10. Add a star next to each
method whose paired-Wilcoxon test against v9-final has p_bonf < 0.05.

Usage
-----
    python -m paper.compile_paper_table
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "paper" / "results"
SNAPSHOTS = RESULTS / "snapshots"


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  warning: failed to read {path}: {exc}", file=sys.stderr)
        return None


def _floor_rows() -> list[dict]:
    """Floor baselines from `paper.baselines floor` (paper/results/baselines_results.json)."""
    payload = _load_json(RESULTS / "baselines_results.json")
    if not payload:
        return []
    rows = payload if isinstance(payload, list) else payload.get("results", [])
    out = []
    for r in rows:
        out.append(_normalize_row(r, source="floor"))
    return out


def _external_rows() -> list[dict]:
    payload = _load_json(RESULTS / "external_baselines_results.json")
    if not payload:
        return []
    out = []
    for method_name, r in payload.items():
        r = dict(r)
        r["method"] = method_name
        out.append(_normalize_row(r, source="external"))
    return out


def _ablation_rows() -> list[dict]:
    """Pull metrics from ablation eval snapshots. Two schemas land in
    snapshots/ depending on which evaluator wrote them:

    * ``paper.eval_engine_on_test`` (inference-only) writes
      ``{"summary": {"overall": {...}, "per_file": {...}}}``.
    * ``hybrid.evaluate`` (full retrain + eval) writes the v9-style
      schema ``{"overall_stage23": {...}, "overall_per_file_stage23": {...}}``.

    We accept either shape so the no-soft-handoff (cheap), no_pca,
    no_trunk, and multi_seed_* rows all merge cleanly."""
    candidates = [
        # v9 (leaked dataset) ablations — kept for historical comparison
        "no_soft_handoff",
        "no_pca",
        "no_trunk",
        "no_trunk_plus_no_soft_handoff",
        "multi_seed_123",
        "multi_seed_456",
        # v10 (de-duplicated dataset) ablations — current paper-relevant rows
        "v10_no_pca",
        "v10_no_trunk",
        "v10_no_soft_handoff",
    ]
    out: list[dict] = []
    for tag in candidates:
        payload = _load_json(SNAPSHOTS / f"eval_{tag}.json")
        if not payload:
            continue

        # Try v9-style schema first (hybrid.evaluate output), then fall
        # back to eval_engine_on_test's nested shape.
        if "overall_stage23" in payload or "overall_metrics" in payload:
            row = _v9_snapshot_to_row(tag, payload)
            out.append(_normalize_row(row, source="ablation"))
            continue

        s = payload.get("summary", {})
        overall = s.get("overall", {})
        per_label = overall.get("per_label", {}) or {}
        per_file = s.get("per_file", {}) or {}
        out.append(_normalize_row({
            "method": tag,
            "accuracy": overall.get("accuracy"),
            "macro_f1": overall.get("macro_f1"),
            "neurite_macro_f1": overall.get("neurite_macro_f1"),
            "per_class_f1": {
                "soma": (per_label.get("soma") or {}).get("f1"),
                "axon": (per_label.get("axon") or {}).get("f1"),
                "basal/dendrite": (per_label.get("basal") or {}).get("f1"),
                "apical": (per_label.get("apical") or {}).get("f1"),
            },
            "per_file": per_file,
        }, source="ablation"))
    return out


def _v6789_rows() -> list[dict]:
    """Pull headline metrics from the v6/v7/v8/v9 snapshot JSONs.
    The v7 / v9_no_gnn rows double as the no-subtree-stage2 / no-gnn
    ablation rows respectively — they have the same architecture
    minus that one component."""
    files = [
        ("v6", SNAPSHOTS / "v6_full_pipeline.json"),
        ("v7_gnn_branch (= no-subtree-stage2)", SNAPSHOTS / "v7_gnn_branch.json"),
        ("v8_subtree_gnn", SNAPSHOTS / "v8_subtree_gnn.json"),
        ("v9_no_gnn (= no-gnn ablation)", SNAPSHOTS / "v9_baseline_no_gnn.json"),
        ("v9_final (leaked split)", SNAPSHOTS / "v9_final_subtree_gnn.json"),
        ("v10_final (dedup split)", SNAPSHOTS / "eval_v10_final.json"),
    ]
    out = []
    for label, path in files:
        payload = _load_json(path)
        if not payload:
            continue
        r = _v9_snapshot_to_row(label, payload)
        out.append(_normalize_row(r, source="pipeline"))
    return out


def _v9_snapshot_to_row(label: str, payload: dict) -> dict:
    """Re-shape an evaluate-style snapshot into the same row schema as
    baselines_results.json so they stack cleanly. The evaluate.py output
    uses ``overall_stage23`` (after Stage 3 refinement) for the headline
    numbers and ``overall_per_file_stage23`` for the per-file summary."""
    overall = (
        payload.get("overall_stage23")
        or payload.get("overall_metrics")
        or payload.get("overall")
        or {}
    )
    per_label = overall.get("per_label") or payload.get("per_class_metrics") or {}
    file_summary = (
        payload.get("overall_per_file_stage23")
        or payload.get("file_summary")
        or payload.get("per_file_summary")
        or {}
    )

    # The evaluate snapshot uses 'basal' for what the rest of the pipeline
    # calls 'basal/dendrite'. Try both keys.
    basal_key = "basal/dendrite" if "basal/dendrite" in per_label else "basal"

    return {
        "method": label,
        "accuracy": overall.get("accuracy"),
        "macro_f1": overall.get("macro_f1"),
        "neurite_macro_f1": overall.get("neurite_macro_f1"),
        "per_class_f1": {
            "soma": (per_label.get("soma") or {}).get("f1"),
            "axon": (per_label.get("axon") or {}).get("f1"),
            "basal/dendrite": (per_label.get(basal_key) or {}).get("f1"),
            "apical": (per_label.get("apical") or {}).get("f1"),
        },
        "per_file": {
            "mean": file_summary.get("mean"),
            "median": file_summary.get("median"),
            "p10": file_summary.get("p10"),
            "n_files": file_summary.get("n_files"),
        },
    }


def _normalize_row(r: dict, source: str) -> dict:
    return {
        "method": r.get("method"),
        "source": source,
        "accuracy": _f(r.get("accuracy")),
        "macro_f1": _f(r.get("macro_f1")),
        "neurite_macro_f1": _f(r.get("neurite_macro_f1")),
        "soma_f1": _f((r.get("per_class_f1") or {}).get("soma")),
        "axon_f1": _f((r.get("per_class_f1") or {}).get("axon")),
        "basal_f1": _f((r.get("per_class_f1") or {}).get("basal/dendrite")),
        "apical_f1": _f((r.get("per_class_f1") or {}).get("apical")),
        "per_file_mean": _f((r.get("per_file") or {}).get("mean")),
        "per_file_p10": _f((r.get("per_file") or {}).get("p10")),
        "n_files": int((r.get("per_file") or {}).get("n_files") or 0),
    }


def _f(x):
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _wilcoxon_lookup() -> dict[str, dict]:
    """For each method, pull its paired-Wilcoxon test against v9_final."""
    payload = _load_json(SNAPSHOTS / "significance_tests.json")
    if not payload:
        return {}
    out: dict[str, dict] = {}
    for r in payload.get("pairwise_tests", []):
        a, b = r.get("method_a"), r.get("method_b")
        target = "v9_final"
        if a == target:
            other, sign = b, -1.0
        elif b == target:
            other, sign = a, +1.0
        else:
            continue
        out[other] = {
            "n_pairs": r.get("n_pairs"),
            "mean_diff_vs_v9": sign * float(r.get("mean_diff", 0.0)),
            "p_value": r.get("p_value"),
            "p_value_bonferroni": r.get("p_value_bonferroni"),
            "significant": r.get("significant_alpha_corrected"),
        }
    return out


def _format_table(rows: list[dict], wilcoxon: dict[str, dict]) -> str:
    cols = [
        ("method", 24),
        ("source", 9),
        ("acc", 6),
        ("macro_f1", 8),
        ("neurite_f1", 10),
        ("soma", 6),
        ("axon", 6),
        ("basal", 6),
        ("apical", 6),
        ("pf_mean", 8),
        ("pf_p10", 7),
        ("vs v9 Δ", 9),
        ("p_bonf", 9),
    ]
    header = "  ".join(f"{name:>{w}}" if name in {"acc","macro_f1","neurite_f1","soma","axon","basal","apical","pf_mean","pf_p10","vs v9 Δ","p_bonf"} else f"{name:<{w}}" for name, w in cols)
    sep = "─" * len(header)
    lines = [header, sep]

    def _fmt(v, places=4):
        return f"{v:.{places}f}" if isinstance(v, float) else "  -  "

    for row in rows:
        m = row["method"]
        wlk = wilcoxon.get(m, {})
        diff = wlk.get("mean_diff_vs_v9")
        pbonf = wlk.get("p_value_bonferroni")
        sig = "*" if wlk.get("significant") else " "
        diff_str = f"{diff:+.4f}" if isinstance(diff, float) else "  -  "
        pbonf_str = f"{pbonf:.2e}" if isinstance(pbonf, float) else "  -  "
        lines.append(
            f"{m:<24}  {row['source']:<9}  "
            f"{_fmt(row['accuracy']):>6}  {_fmt(row['macro_f1']):>8}  "
            f"{_fmt(row['neurite_macro_f1']):>10}  "
            f"{_fmt(row['soma_f1']):>6}  {_fmt(row['axon_f1']):>6}  "
            f"{_fmt(row['basal_f1']):>6}  {_fmt(row['apical_f1']):>6}  "
            f"{_fmt(row['per_file_mean']):>8}  {_fmt(row['per_file_p10']):>7}  "
            f"{diff_str:>9}  {pbonf_str:>9}{sig}"
        )
    lines.append("")
    lines.append("vs v9 Δ = mean (method - v9_final) per-file F1; positive = method beats v9_final.")
    lines.append("p_bonf = paired-Wilcoxon p-value, Bonferroni-corrected over all pairwise tests.")
    lines.append("* = significant at alpha=0.05 after Bonferroni correction.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-text", type=Path, default=RESULTS / "paper_table.txt",
    )
    parser.add_argument(
        "--out-json", type=Path, default=RESULTS / "paper_table_full.json",
    )
    args = parser.parse_args()

    rows = []
    rows.extend(_floor_rows())
    rows.extend(_external_rows())
    rows.extend(_v6789_rows())
    rows.extend(_ablation_rows())

    wilcoxon = _wilcoxon_lookup()
    text = _format_table(rows, wilcoxon)

    args.out_text.write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"Wrote {args.out_text}")
    args.out_json.write_text(json.dumps({
        "rows": rows, "wilcoxon": wilcoxon,
    }, indent=2))
    print(f"Wrote {args.out_json}")
    print()
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
