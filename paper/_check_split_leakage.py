#!/usr/bin/env python3
"""Verify seed split consistency and obvious train/test leakage risks."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.evaluate import _file_in_test_bucket  # noqa: E402


MODEL_DIR_SUFFIX = os.environ.get("SWCAL_MODEL_DIR_SUFFIX", "")
QC_CSV = Path(os.environ.get(
    "SWCAL_QC_CSV",
    str(ROOT / "paper" / "results" / "corpus_qc_v12_uncurated.csv"),
))


def _load_qc_pairs(qc_csv: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"pyramidal": [], "interneuron": []}
    with qc_csv.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("qc_pass", "").strip().lower() not in ("true", "1"):
                continue
            path = Path(row["path"])
            if not path.is_file():
                continue
            out.setdefault(row["cell_type"], []).append(path.name)
    for ct in out:
        out[ct] = sorted(out[ct])
    return out


def _pairs(split_block: dict[str, list[str]]) -> set[tuple[str, str]]:
    return {
        (ct, name)
        for ct, names in split_block.items()
        for name in names
    }


def _duplicates(split_block: dict[str, list[str]]) -> list[tuple[str, str, int]]:
    dupes: list[tuple[str, str, int]] = []
    for ct, names in split_block.items():
        for name, n in Counter(names).items():
            if n > 1:
                dupes.append((ct, name, n))
    return sorted(dupes)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--qc-csv", type=Path, default=QC_CSV)
    ap.add_argument("--baseline-json", type=Path, default=None)
    ap.add_argument("--out-stem", type=Path, default=None)
    args = ap.parse_args()

    model_dir = args.model_dir or (
        ROOT / "paper" / "models" / f"v12_gentle_seed{args.seed}{MODEL_DIR_SUFFIX}"
    )
    baseline_json = args.baseline_json
    if baseline_json is None:
        candidate = ROOT / "paper" / "results" / f"baselines_on_v12{MODEL_DIR_SUFFIX}.json"
        baseline_json = candidate if candidate.is_file() else None
    elif not baseline_json.is_absolute():
        baseline_json = (ROOT / baseline_json).resolve()
    out_stem = args.out_stem or (
        ROOT / "paper" / "results" / f"leakage_check_seed{args.seed}{MODEL_DIR_SUFFIX}"
    )
    if not args.qc_csv.is_absolute():
        args.qc_csv = (ROOT / args.qc_csv).resolve()

    split_path = model_dir / "train_test_split.json"
    if not split_path.is_file():
        raise SystemExit(f"MISSING: {split_path}")
    if not args.qc_csv.is_file():
        raise SystemExit(f"MISSING: {args.qc_csv}")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    test_size = float(split.get("test_size", 0.20))
    qc = _load_qc_pairs(args.qc_csv)

    expected = {"train": {"pyramidal": [], "interneuron": []},
                "test": {"pyramidal": [], "interneuron": []}}
    for ct, names in qc.items():
        for name in names:
            bucket = "test" if _file_in_test_bucket(name, args.seed, test_size) else "train"
            expected[bucket].setdefault(ct, []).append(name)
    for bucket in ("train", "test"):
        for ct in expected[bucket]:
            expected[bucket][ct] = sorted(expected[bucket][ct])

    train_pairs = _pairs(split["train"])
    test_pairs = _pairs(split["test"])
    expected_train_pairs = _pairs(expected["train"])
    expected_test_pairs = _pairs(expected["test"])
    overlap = sorted(train_pairs & test_pairs)
    missing_from_model = sorted((expected_train_pairs | expected_test_pairs) - (train_pairs | test_pairs))
    extra_in_model = sorted((train_pairs | test_pairs) - (expected_train_pairs | expected_test_pairs))
    wrong_train = sorted(train_pairs ^ expected_train_pairs)
    wrong_test = sorted(test_pairs ^ expected_test_pairs)

    branch3_path = model_dir / "gnn_branch3_rescue.pt"
    branch3 = {"present": branch3_path.is_file()}
    if branch3_path.is_file():
        import torch

        payload = torch.load(branch3_path, map_location="cpu", weights_only=False)
        train_config = payload.get("train_config", {})
        branch3.update({
            "path": str(branch3_path.relative_to(ROOT)),
            "seed": train_config.get("seed"),
            "seed_matches": train_config.get("seed") == args.seed,
            "has_test_metrics": payload.get("test_metrics") is not None,
            "test_metrics": payload.get("test_metrics"),
        })

    baseline = {"checked": False}
    if baseline_json is not None:
        payload = json.loads(baseline_json.read_text(encoding="utf-8"))
        baseline = {
            "checked": True,
            "path": str(baseline_json.relative_to(ROOT)),
            "seed": payload.get("seed"),
            "seed_matches": payload.get("seed") == args.seed,
            "n_train": payload.get("n_train"),
            "n_test": payload.get("n_test"),
            "n_train_expected": len(expected_train_pairs),
            "n_test_expected": len(expected_test_pairs),
            "counts_match": (
                payload.get("n_train") == len(expected_train_pairs)
                and payload.get("n_test") == len(expected_test_pairs)
            ),
            "split_source": payload.get("split_source"),
            "qc_csv": payload.get("qc_csv"),
        }

    counts = {
        "train": {ct: len(split["train"].get(ct, [])) for ct in ("pyramidal", "interneuron")},
        "test": {ct: len(split["test"].get(ct, [])) for ct in ("pyramidal", "interneuron")},
        "expected_train": {ct: len(expected["train"].get(ct, [])) for ct in ("pyramidal", "interneuron")},
        "expected_test": {ct: len(expected["test"].get(ct, [])) for ct in ("pyramidal", "interneuron")},
    }
    ok = (
        not overlap
        and not missing_from_model
        and not extra_in_model
        and not wrong_train
        and not wrong_test
        and not _duplicates(split["train"])
        and not _duplicates(split["test"])
        and (not branch3["present"] or branch3.get("seed_matches"))
        and (not baseline["checked"] or (baseline.get("seed_matches") and baseline.get("counts_match")))
    )
    report = {
        "ok": ok,
        "seed": args.seed,
        "test_size": test_size,
        "model_dir": str(model_dir.relative_to(ROOT)),
        "qc_csv": str(args.qc_csv.relative_to(ROOT)),
        "counts": counts,
        "train_test_overlap": overlap,
        "train_duplicates": _duplicates(split["train"]),
        "test_duplicates": _duplicates(split["test"]),
        "missing_from_model_split": missing_from_model,
        "extra_in_model_split": extra_in_model,
        "train_split_xor_expected": wrong_train,
        "test_split_xor_expected": wrong_test,
        "branch3": branch3,
        "baseline": baseline,
    }

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    out_stem.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        f"Leakage check seed={args.seed}{MODEL_DIR_SUFFIX}",
        f"ok: {ok}",
        f"model_dir: {report['model_dir']}",
        f"qc_csv: {report['qc_csv']}",
        f"counts: {json.dumps(counts, sort_keys=True)}",
        f"train/test overlap: {len(overlap)}",
        f"missing from model split: {len(missing_from_model)}",
        f"extra in model split: {len(extra_in_model)}",
        f"wrong train assignments: {len(wrong_train)}",
        f"wrong test assignments: {len(wrong_test)}",
        f"branch3: {json.dumps(branch3, sort_keys=True)}",
        f"baseline: {json.dumps(baseline, sort_keys=True)}",
    ]
    out_stem.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {out_stem.with_suffix('.json')}")
    print(f"Wrote {out_stem.with_suffix('.txt')}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
