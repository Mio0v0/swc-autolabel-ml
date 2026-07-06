"""Input QC gate for the auto-labeling pipeline.

Stage 1 of the three-stage selective-prediction pipeline:
    1.  *Input QC*  (this module)        — reject unlabelable files
    2.  Auto-label                        — run the 4-stage model
    3.  Confidence-based flagging         — see hybrid.confidence

The QC gate performs two kinds of checks:

  A.  STRUCTURAL  — deterministic rules on the SWC. Catches truly
      unparseable / broken files: malformed rows, multiple roots, orphan
      nodes, duplicate IDs, cycles, non-finite coords, invalid radii,
      etc. These checks are type-agnostic and do not require existing
      soma or neurite labels.

  B.  DISTRIBUTION  — out-of-distribution detection on the Stage 1
      feature vector. Files whose features are far from the training
      manifold are rejected even when they parse cleanly. Uses a
      Mahalanobis-style distance fit on the training feature
      distribution.

Output is a single ``QCResult`` with ``passed: bool`` and a list of
human-readable ``reasons`` so reviewers know why a file was rejected.

Usage:
    from hybrid.qc_input import QCGate
    gate = QCGate.load("paper/models/qc_gate.pkl")
    qc = gate.evaluate(path_to_swc)
    if qc.passed:
        run_pipeline_on_nodes(...)
    else:
        # log qc.reasons; do not auto-label
        ...
"""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from .features import extract_feature_vector, FEATURE_NAMES, SWCNode

# -----------------------------------------------------------------------------
# Constants — keep aligned with paper/_scan_corpus_qc.py
# -----------------------------------------------------------------------------
STANDARD_TYPES = {0, 1, 2, 3, 4, 5, 6, 7}
MIN_NODES = 10
MAX_NODES = 200_000


@dataclass
class QCResult:
    """Output of QCGate.evaluate."""
    passed: bool
    reasons: list[str] = field(default_factory=list)
    n_nodes: int = 0
    n_soma: int = 0
    n_roots: int = 0
    n_orphan: int = 0
    n_other_type: int = 0
    n_negative_type: int = 0
    n_duplicate_id: int = 0
    n_self_parent: int = 0
    n_cycle: int = 0
    ood_distance: float | None = None
    ood_threshold: float | None = None
    feature_vector: list[float] | None = None
    path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# -----------------------------------------------------------------------------
# Structural QC
# -----------------------------------------------------------------------------
def _is_finite(v: float) -> bool:
    return not (math.isnan(v) or math.isinf(v))


def _is_integer_float(v: float) -> bool:
    return _is_finite(v) and float(v).is_integer()


def _parse_swc_for_qc(path: Path) -> tuple[list[SWCNode], list[str]]:
    """Parse SWC rows strictly enough for input QC."""
    nodes: list[SWCNode] = []
    reasons: list[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 7:
                reasons.append(f"malformed_row:{line_no}:expected_7_columns")
                continue
            try:
                raw_id = float(parts[0])
                raw_type = float(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                radius = float(parts[5])
                raw_parent = float(parts[6])
            except ValueError:
                reasons.append(f"malformed_row:{line_no}:non_numeric")
                continue

            if not _is_integer_float(raw_id):
                reasons.append(f"malformed_row:{line_no}:non_integer_id")
                continue
            if not _is_integer_float(raw_type):
                reasons.append(f"malformed_row:{line_no}:non_integer_type")
                continue
            if not _is_integer_float(raw_parent):
                reasons.append(f"malformed_row:{line_no}:non_integer_parent")
                continue

            nodes.append(
                SWCNode(
                    id=int(raw_id),
                    type=int(raw_type),
                    x=x,
                    y=y,
                    z=z,
                    radius=radius,
                    parent=int(raw_parent),
                )
            )
    return nodes, reasons


def _count_cycle_nodes(nodes: list[SWCNode]) -> int:
    """Return the number of nodes whose parent chain enters a cycle."""
    by_id = {n.id: n for n in nodes}
    cycle_nodes: set[int] = set()
    for node in nodes:
        seen: dict[int, int] = {}
        chain: list[int] = []
        current = node.id
        while current != -1 and current in by_id:
            if current in seen:
                cycle_nodes.update(chain[seen[current]:])
                break
            seen[current] = len(chain)
            chain.append(current)
            parent = by_id[current].parent
            if parent == -1 or parent not in by_id:
                break
            current = parent
    return len(cycle_nodes)


def structural_qc(nodes: list[SWCNode]) -> tuple[bool, list[str], dict]:
    """Run deterministic structural checks. Returns (passed, reasons, counts)."""
    reasons: list[str] = []
    counts = {
        "n_nodes": len(nodes),
        "n_soma": 0,
        "n_roots": 0,
        "n_orphan": 0,
        "n_other_type": 0,
        "n_negative_type": 0,
        "n_duplicate_id": 0,
        "n_self_parent": 0,
        "n_cycle": 0,
    }
    if not nodes:
        return False, ["empty_after_parse"], counts
    if len(nodes) < MIN_NODES:
        reasons.append(f"too_few_nodes:{len(nodes)}<{MIN_NODES}")
    if len(nodes) > MAX_NODES:
        reasons.append(f"too_many_nodes:{len(nodes)}>{MAX_NODES}")

    ids = [n.id for n in nodes]
    by_id = {n.id: n for n in nodes}
    counts["n_soma"]      = sum(1 for n in nodes if n.type == 1)
    counts["n_roots"]     = sum(1 for n in nodes if n.parent == -1)
    counts["n_orphan"]    = sum(1 for n in nodes if n.parent != -1 and n.parent not in by_id)
    counts["n_other_type"] = sum(1 for n in nodes if n.type not in STANDARD_TYPES)
    counts["n_negative_type"] = sum(1 for n in nodes if n.type < 0)
    counts["n_duplicate_id"] = len(ids) - len(set(ids))
    counts["n_self_parent"] = sum(1 for n in nodes if n.parent == n.id)
    counts["n_cycle"] = _count_cycle_nodes(nodes) if counts["n_duplicate_id"] == 0 else 0

    if any(n.id <= 0 for n in nodes):
        reasons.append("non_positive_node_id")
    if counts["n_roots"] != 1:
        reasons.append(f"n_roots={counts['n_roots']}")
    if counts["n_orphan"] > 0:
        reasons.append(f"n_orphan={counts['n_orphan']}")
    if counts["n_negative_type"] > 0:
        reasons.append(f"negative_type_count={counts['n_negative_type']}")
    if counts["n_duplicate_id"] > 0:
        reasons.append(f"duplicate_id_count={counts['n_duplicate_id']}")
    if counts["n_self_parent"] > 0:
        reasons.append(f"self_parent_count={counts['n_self_parent']}")
    if counts["n_cycle"] > 0:
        reasons.append(f"cycle_node_count={counts['n_cycle']}")

    if not all(_is_finite(n.x) and _is_finite(n.y) and _is_finite(n.z) for n in nodes):
        reasons.append("non_finite_coords")
    if not all(_is_finite(n.radius) and n.radius >= 0 for n in nodes):
        reasons.append("invalid_radii")

    return (len(reasons) == 0), reasons, counts


# -----------------------------------------------------------------------------
# Distribution (OOD) check
# -----------------------------------------------------------------------------
@dataclass
class OODDetector:
    """Mahalanobis-distance OOD detector fit on the training feature distribution.

    Implementation notes:
      - We compute a robust mean + diagonal covariance on the training
        feature vectors (Stage 1's global morphology features). Robust
        because the training corpus has heavy-tailed feature distributions
        (a few cells with 100k nodes); raw mean/cov is fragile to those.
      - Diagonal covariance avoids singular-matrix issues with limited
        training data; in practice the off-diagonal terms add little
        signal here and a lot of fitting noise.
      - The distance threshold is set at the 99th percentile of training
        distances (configurable). Anything farther is flagged OOD.
    """

    mean: np.ndarray                   # (D,) — feature means
    std: np.ndarray                    # (D,) — robust scales (NOT variances)
    threshold: float                   # distance above this = OOD
    feature_names: list[str]           # for sanity-checking inputs
    quantile: float = 0.99             # threshold quantile used at fit time
    n_train: int = 0

    def distance(self, x: np.ndarray) -> float:
        """Standardized Euclidean (== Mahalanobis with diagonal Σ) distance."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.mean.shape[0]:
            raise ValueError(
                f"feature length {x.shape[0]} != fitted length {self.mean.shape[0]}"
            )
        std = np.where(self.std < 1e-9, 1e-9, self.std)
        z = (x - self.mean) / std
        # nan-safe: missing features count as max-distance contributors
        z = np.where(np.isfinite(z), z, 10.0)
        return float(np.sqrt(np.sum(z * z)))

    def is_in_distribution(self, x: np.ndarray) -> bool:
        return self.distance(x) <= self.threshold

    @classmethod
    def fit(
        cls,
        train_features: np.ndarray,         # (N, D)
        feature_names: list[str],
        quantile: float = 0.99,
    ) -> "OODDetector":
        X = np.asarray(train_features, dtype=np.float64)
        # Robust center and scale: median + IQR-based scale (1.349 ≈ Φ⁻¹(.75))
        # Replace non-finite entries with column median first so they don't poison.
        col_medians = np.nanmedian(X, axis=0)
        bad = ~np.isfinite(X)
        X = np.where(bad, np.tile(col_medians, (X.shape[0], 1)), X)
        med = np.median(X, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        scale = (q75 - q25) / 1.349
        scale = np.where(scale < 1e-9, 1.0, scale)
        # Fit threshold at training quantile
        z = (X - med) / scale
        dists = np.sqrt(np.sum(z * z, axis=1))
        threshold = float(np.quantile(dists, quantile))
        return cls(
            mean=med, std=scale,
            threshold=threshold,
            feature_names=list(feature_names),
            quantile=quantile,
            n_train=int(X.shape[0]),
        )

    def to_dict(self) -> dict:
        return {
            "mean":           self.mean.tolist(),
            "std":            self.std.tolist(),
            "threshold":      self.threshold,
            "feature_names":  self.feature_names,
            "quantile":       self.quantile,
            "n_train":        self.n_train,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OODDetector":
        return cls(
            mean=np.asarray(d["mean"], dtype=np.float64),
            std=np.asarray(d["std"], dtype=np.float64),
            threshold=float(d["threshold"]),
            feature_names=list(d["feature_names"]),
            quantile=float(d.get("quantile", 0.99)),
            n_train=int(d.get("n_train", 0)),
        )


# -----------------------------------------------------------------------------
# Combined QC gate
# -----------------------------------------------------------------------------
@dataclass
class QCGate:
    """Combined structural + OOD QC gate.

    If ``ood_detector`` is None, OOD checks are skipped (the gate is
    structural-only — useful early in development before the OOD detector
    has been fit on the training set).
    """

    ood_detector: OODDetector | None = None

    def evaluate(self, path: Path | str) -> QCResult:
        path = Path(path)
        # Parse
        try:
            nodes, parse_reasons = _parse_swc_for_qc(path)
        except Exception as exc:
            return QCResult(passed=False, reasons=[f"parse_error:{exc}"], path=str(path))

        # Structural
        struct_pass, struct_reasons, counts = structural_qc(nodes)
        passed = struct_pass and not parse_reasons
        res = QCResult(
            passed=passed,
            reasons=list(parse_reasons) + list(struct_reasons),
            n_nodes=counts["n_nodes"],
            n_soma=counts["n_soma"],
            n_roots=counts["n_roots"],
            n_orphan=counts["n_orphan"],
            n_other_type=counts["n_other_type"],
            n_negative_type=counts["n_negative_type"],
            n_duplicate_id=counts["n_duplicate_id"],
            n_self_parent=counts["n_self_parent"],
            n_cycle=counts["n_cycle"],
            path=str(path),
        )
        if not res.passed:
            return res

        # OOD
        if self.ood_detector is not None:
            try:
                # Structural QC above inspects the raw parsed file. The OOD
                # detector, however, was fit on the normalized Stage 1 feature
                # path, so keep that distribution stable here.
                from .swc_normalize import normalize_swc  # noqa: PLC0415

                feature_nodes, _ = normalize_swc(nodes)
                fv = extract_feature_vector(feature_nodes)
            except Exception as exc:
                res.passed = False
                res.reasons.append(f"feature_extraction_error:{exc}")
                return res
            res.feature_vector = list(map(float, fv))
            d = self.ood_detector.distance(fv)
            res.ood_distance = d
            res.ood_threshold = self.ood_detector.threshold
            if d > self.ood_detector.threshold:
                res.passed = False
                res.reasons.append(
                    f"ood:distance={d:.2f}>threshold={self.ood_detector.threshold:.2f}"
                )
        return res

    # ---- persistence ----
    def save(self, path: Path | str) -> None:
        payload = {
            "ood_detector": self.ood_detector.to_dict() if self.ood_detector else None,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path: Path | str) -> "QCGate":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        ood = None
        if payload.get("ood_detector"):
            ood = OODDetector.from_dict(payload["ood_detector"])
        return cls(ood_detector=ood)
