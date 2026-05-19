"""Input QC gate for the auto-labeling pipeline.

Stage 1 of the three-stage selective-prediction pipeline:
    1.  *Input QC*  (this module)        — reject unlabelable files
    2.  Auto-label                        — run the 4-stage model
    3.  Confidence-based flagging         — see hybrid.confidence

The QC gate performs two kinds of checks:

  A.  STRUCTURAL  — deterministic rules on the SWC. Catches truly
      unparseable / broken files: missing soma, multiple roots, orphan
      nodes, non-finite coords, etc. Same checks as the corpus-scan
      script (paper/_scan_corpus_qc.py), packaged for runtime use.

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

from .features import parse_swc, extract_feature_vector, FEATURE_NAMES, SWCNode

# -----------------------------------------------------------------------------
# Constants — keep aligned with paper/_scan_corpus_qc.py
# -----------------------------------------------------------------------------
VALID_TYPES = {0, 1, 2, 3, 4, 5, 6, 7}
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


def structural_qc(nodes: list[SWCNode]) -> tuple[bool, list[str], dict]:
    """Run deterministic structural checks. Returns (passed, reasons, counts)."""
    reasons: list[str] = []
    counts = {
        "n_nodes": len(nodes),
        "n_soma": 0,
        "n_roots": 0,
        "n_orphan": 0,
        "n_other_type": 0,
    }
    if not nodes:
        return False, ["empty_after_parse"], counts
    if len(nodes) < MIN_NODES:
        reasons.append(f"too_few_nodes:{len(nodes)}<{MIN_NODES}")
    if len(nodes) > MAX_NODES:
        reasons.append(f"too_many_nodes:{len(nodes)}>{MAX_NODES}")

    by_id = {n.id: n for n in nodes}
    counts["n_soma"]      = sum(1 for n in nodes if n.type == 1)
    counts["n_roots"]     = sum(1 for n in nodes if n.parent == -1)
    counts["n_orphan"]    = sum(1 for n in nodes if n.parent != -1 and n.parent not in by_id)
    counts["n_other_type"] = sum(1 for n in nodes if n.type not in VALID_TYPES)

    if counts["n_soma"] == 0:
        reasons.append("no_soma")
    if counts["n_roots"] != 1:
        reasons.append(f"n_roots={counts['n_roots']}")
    if counts["n_orphan"] > 0:
        reasons.append(f"n_orphan={counts['n_orphan']}")
    if counts["n_other_type"] > 0:
        reasons.append(f"non_standard_label_count={counts['n_other_type']}")

    if not all(_is_finite(n.x) and _is_finite(n.y) and _is_finite(n.z) for n in nodes):
        reasons.append("non_finite_coords")
    if not all(_is_finite(n.radius) and n.radius >= 0 for n in nodes):
        reasons.append("invalid_radii")

    n_neurites = sum(1 for n in nodes if n.type in (2, 3, 4))
    if n_neurites == 0:
        reasons.append("no_neurites")

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
            nodes = parse_swc(path)
        except Exception as exc:
            return QCResult(passed=False, reasons=[f"parse_error:{exc}"], path=str(path))

        # Structural
        struct_pass, struct_reasons, counts = structural_qc(nodes)
        res = QCResult(
            passed=struct_pass,
            reasons=list(struct_reasons),
            n_nodes=counts["n_nodes"],
            n_soma=counts["n_soma"],
            n_roots=counts["n_roots"],
            n_orphan=counts["n_orphan"],
            n_other_type=counts["n_other_type"],
            path=str(path),
        )
        if not struct_pass:
            return res

        # OOD
        if self.ood_detector is not None:
            try:
                fv = extract_feature_vector(nodes)
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
