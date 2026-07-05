"""Curator feedback loop — human-in-the-loop labeling with versioned models.

The flag step (Stage 4) is not a terminal filter but the entry point to a
human-review cycle whose corrections retrain BOTH the labeler and the flag
model, producing new *versioned* checkpoints while the original factory
model is preserved untouched:

    Raw SWC
       │
       ▼
    [ QC → cell type → Stage 2 → GNN → Stage 3 + Branch3 ]   (the labeler)
       │
       ▼
    Stage 4  Flag model  ── flagged? ──┐
       │ no                            │ yes
       ▼                               ▼
    accept labels             curator supplies CORRECTED labels
                                     │
                     ┌───────────────┴────────────────┐
                     ▼                                 ▼
        labeler training signal            flag training signal
        (cell, curator labels)             (flag features, was_flag_justified)
                     │                                 │
                     │   was_flag_justified is DERIVED, not chosen:
                     │   per_cell_F1(curator, model) < 0.60  →  truly bad (1)
                     │   otherwise                            →  false alarm (0)
                     ▼                                 ▼
                  CuratorVerifiedStore  (persistent, holds both signals)
                     │
                     ▼
        retrain → NEW versioned checkpoint  (factory preserved)

Two design commitments (both from the paper §4.7 and a PI requirement):

1. **The curator never judges the flag.** They only provide corrected
   labels. Whether the flag was a true positive or a false alarm is
   *computed* from the disagreement between the curator's labels and the
   model's, using the identical "per-cell F1 < 0.60" definition the flag
   model was originally trained on. Correcting flagged cells therefore
   improves the labeler AND teaches the flag model to stop firing on the
   low-confidence-but-actually-fine cells (false-alarm reduction).

2. **The factory model is immutable.** Every retrain writes a new
   versioned checkpoint under a separate directory with provenance; the
   original ("factory") checkpoint is never overwritten. This is enforced
   by `ModelRegistry`, not by convention.

This module wires together components that already exist
(`run_pipeline_on_nodes`, `hybrid.evaluate._train_stage2`) and delegates
flag-model training to a caller-supplied trainer (the flag feature
pipeline lives in paper/), so the loop structure stays in hybrid/.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional, Sequence

from .features import SWCNode, parse_swc
from .pipeline import run_pipeline_on_nodes, PipelineResult
from .evaluate import per_cell_neurite_f1

# Same "bad" definition the flag model is trained on (paper §4.4/§5.3).
DEFAULT_BAD_F1_THRESHOLD = 0.60


# ---------------------------------------------------------------------------
# Deriving the flag ground truth from curator labels (no explicit good/bad)
# ---------------------------------------------------------------------------

def derive_flag_truth(
    model_labels: Sequence[int],
    curator_labels: Sequence[int],
    cell_type: str,
    bad_f1_threshold: float = DEFAULT_BAD_F1_THRESHOLD,
) -> tuple[bool, float]:
    """Compute whether a flag was justified from curator vs. model labels.

    Treats the curator's labels as ground truth and measures how far the
    model's original labels fell from them, using the same per-cell neurite
    macro-F1 the flag model was trained against.

    Returns ``(flag_was_justified, agreement_f1)`` where
    ``flag_was_justified`` is ``agreement_f1 < bad_f1_threshold`` (i.e. the
    model really did mislabel the cell → a true-positive flag), and
    ``agreement_f1`` is the curator-vs-model per-cell F1 (1.0 = curator made
    no change → the flag was a false alarm).
    """
    agreement_f1 = per_cell_neurite_f1(
        list(curator_labels), list(model_labels), cell_type,
    )
    return (agreement_f1 < bad_f1_threshold, float(agreement_f1))


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------

@dataclass
class LabelResult:
    """Output of one labeling pass, before any curator review."""
    file: str
    cell_type: str
    node_labels: list[int]
    flag_score: float                 # higher = more likely mislabeled
    flagged: bool                     # flag_score >= flag_threshold
    flag_features: Optional[list[float]] = None  # captured for flag retraining


@dataclass
class VerifiedEntry:
    """One curator-reviewed cell, holding both training signals."""
    file: str
    cell_type: str
    source_swc: str                   # curator-trusted SWC (corrected or endorsed)
    flag_was_justified: bool          # DERIVED, not chosen
    agreement_f1: float               # curator-vs-model per-cell F1
    flag_features: Optional[list[float]]  # features for flag-model retraining
    added_at: float                   # caller-supplied epoch seconds
    note: str = ""


# ---------------------------------------------------------------------------
# Persistent curator-verified store (both labeler + flag signals)
# ---------------------------------------------------------------------------

@dataclass
class CuratorVerifiedStore:
    """Persistent pool of curator-reviewed cells.

    Physically: a directory of trusted SWC files plus ``manifest.json``.
    Every entry supplies (a) a labeler training example — the trusted SWC —
    and (b) a flag training example — ``flag_features`` labeled by the
    DERIVED ``flag_was_justified``.
    """
    root: Path
    entries: list[VerifiedEntry] = field(default_factory=list)

    @classmethod
    def load(cls, root: str | Path) -> "CuratorVerifiedStore":
        root = Path(root)
        manifest = root / "manifest.json"
        entries: list[VerifiedEntry] = []
        if manifest.is_file():
            for row in json.loads(manifest.read_text(encoding="utf-8")):
                entries.append(VerifiedEntry(**row))
        return cls(root=root, entries=entries)

    def _save_manifest(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "manifest.json").write_text(
            json.dumps([asdict(e) for e in self.entries], indent=2),
            encoding="utf-8",
        )

    def add(
        self,
        file: str,
        cell_type: str,
        corrected_swc_path: str | Path,
        flag_was_justified: bool,
        agreement_f1: float,
        flag_features: Optional[Sequence[float]],
        timestamp: float,
        note: str = "",
    ) -> VerifiedEntry:
        self.root.mkdir(parents=True, exist_ok=True)
        dest = self.root / file
        if Path(corrected_swc_path).resolve() != dest.resolve():
            shutil.copy(corrected_swc_path, dest)
        entry = VerifiedEntry(
            file=file, cell_type=cell_type, source_swc=str(dest),
            flag_was_justified=bool(flag_was_justified),
            agreement_f1=float(agreement_f1),
            flag_features=(list(flag_features) if flag_features is not None else None),
            added_at=timestamp, note=note,
        )
        self.entries = [e for e in self.entries if e.file != file]  # latest wins
        self.entries.append(entry)
        self._save_manifest()
        return entry

    def labeler_train_files(self) -> dict[str, list[Path]]:
        """Curator-trusted SWCs grouped by cell type (for `_train_stage2`)."""
        out: dict[str, list[Path]] = {"pyramidal": [], "interneuron": []}
        for e in self.entries:
            out.setdefault(e.cell_type, []).append(Path(e.source_swc))
        return out

    def flag_training_examples(self) -> tuple[list[list[float]], list[int]]:
        """(X, y) for flag-model retraining. y: 1 = truly bad, 0 = false alarm."""
        X: list[list[float]] = []
        y: list[int] = []
        for e in self.entries:
            if e.flag_features is None:
                continue
            X.append(e.flag_features)
            y.append(1 if e.flag_was_justified else 0)
        return X, y

    def __len__(self) -> int:
        return len(self.entries)


# ---------------------------------------------------------------------------
# Versioned model registry — factory preserved, feedback writes new versions
# ---------------------------------------------------------------------------

# Artifacts that make up a complete labeler checkpoint. On a new version we
# copy the ones we do NOT retrain from the parent so every version is a
# complete, runnable model.
_LABELER_ARTIFACTS = (
    "cell_type_classifier.pkl",   # Stage 1
    "branch_classifier.pkl",      # Stage 2  (retrained on feedback)
    "gnn_apical_basal.pt",        # GNN
    "gnn_branch3_rescue.pt",      # Branch3
    "qc_gate.pkl",                # QC
)


@dataclass
class ModelVersion:
    version: str
    path: Path
    parent: Optional[str]
    n_curator_examples: int
    created_at: float
    note: str = ""


class ModelRegistry:
    """Immutable factory checkpoint + append-only versioned checkpoints.

    The factory directory is NEVER written to. `new_version` materializes a
    complete checkpoint under ``versions_root/<name>/`` by copying every
    artifact from its parent, so a retrainer can then overwrite only the
    components it updates (Stage 2, flag model) while the rest remain the
    factory's.
    """

    def __init__(self, factory_dir: str | Path, versions_root: str | Path):
        self.factory_dir = Path(factory_dir)
        self.versions_root = Path(versions_root)
        if not self.factory_dir.is_dir():
            raise FileNotFoundError(f"factory model dir not found: {self.factory_dir}")
        self.versions_root.mkdir(parents=True, exist_ok=True)

    def _provenance_path(self, version_dir: Path) -> Path:
        return version_dir / "provenance.json"

    def parent_dir(self, parent: Optional[str]) -> Path:
        """Resolve a parent version name to a directory (None → factory)."""
        if parent is None or parent in ("factory", "v0", "v0_factory"):
            return self.factory_dir
        return self.versions_root / parent

    def new_version(
        self,
        name: str,
        parent: Optional[str],
        n_curator_examples: int,
        timestamp: float,
        note: str = "",
    ) -> ModelVersion:
        """Create a complete new checkpoint branched from ``parent``.

        Copies every labeler artifact from the parent so the version is
        runnable as-is; the caller then retrains and overwrites the
        components that changed. The factory dir is only ever read.
        """
        src = self.parent_dir(parent)
        dst = self.versions_root / name
        if dst.exists():
            raise FileExistsError(f"version already exists: {dst}")
        dst.mkdir(parents=True)
        for art in _LABELER_ARTIFACTS:
            p = src / art
            if p.is_file():
                shutil.copy(p, dst / art)
        # Carry the split forward if present (audit only; not required).
        for extra in ("train_test_split.json", "eval_split_for_gnn.json"):
            p = src / extra
            if p.is_file():
                shutil.copy(p, dst / extra)
        version = ModelVersion(
            version=name, path=dst,
            parent=(parent or "factory"),
            n_curator_examples=n_curator_examples,
            created_at=timestamp, note=note,
        )
        self._provenance_path(dst).write_text(
            json.dumps(asdict(version) | {"path": str(dst)}, indent=2),
            encoding="utf-8",
        )
        return version

    def assert_factory_intact(self) -> None:
        """Sanity guard: factory artifacts still present (never deleted)."""
        missing = [a for a in _LABELER_ARTIFACTS
                   if not (self.factory_dir / a).is_file()
                   and a in ("branch_classifier.pkl", "cell_type_classifier.pkl")]
        if missing:
            raise RuntimeError(f"factory checkpoint damaged, missing: {missing}")


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

# flag_scorer(nodes, pipeline_result) -> (score, features_or_None)
FlagScorer = Callable[[list[SWCNode], PipelineResult], tuple[float, Optional[Sequence[float]]]]
# flag_trainer(curator_X, curator_y, out_path) -> writes flag model to out_path
FlagTrainer = Callable[[list[list[float]], list[int], Path], Path]


class CuratorFeedbackLoop:
    """label → flag → curator correction → derived signals → versioned retrain.

    Parameters
    ----------
    registry : ModelRegistry
        Factory-preserving versioned checkpoint store. The loop always
        labels with the *current active version* (starts at factory).
    flag_scorer : FlagScorer
        Returns ``(score, features)`` per cell. ``features`` is captured so
        the flag model can be retrained on curator-derived labels.
    verified_store : CuratorVerifiedStore
        Persistent pool of curator-reviewed cells (both signals).
    flag_threshold : float
        Cells with score >= this are surfaced for curator review.
    bad_f1_threshold : float
        Curator-vs-model F1 below this ⇒ the flag was justified (truly bad).
    gnn_state, branch3_state : optional
        Loaded refinement checkpoints for the pipeline.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        flag_scorer: FlagScorer,
        verified_store: CuratorVerifiedStore,
        flag_threshold: float = 0.5,
        bad_f1_threshold: float = DEFAULT_BAD_F1_THRESHOLD,
        gnn_state=None,
        branch3_state=None,
        active_version: Optional[str] = None,
    ):
        self.registry = registry
        self.flag_scorer = flag_scorer
        self.store = verified_store
        self.flag_threshold = flag_threshold
        self.bad_f1_threshold = bad_f1_threshold
        self.gnn_state = gnn_state
        self.branch3_state = branch3_state
        # None → label with the factory checkpoint.
        self.active_version = active_version

    def _active_dir(self) -> Path:
        return self.registry.parent_dir(self.active_version)

    # -- Steps 1-4: label + flag -------------------------------------------

    def label(self, swc_path: str | Path, override_cell_type: str | None = None) -> tuple[LabelResult, PipelineResult]:
        active = self._active_dir()
        nodes = parse_swc(Path(swc_path))
        pr = run_pipeline_on_nodes(
            nodes, file_path=str(swc_path),
            stage1_model=active / "cell_type_classifier.pkl",
            stage2_model=active / "branch_classifier.pkl",
            gnn_state=self.gnn_state, branch3_state=self.branch3_state,
            use_subtree_stage2=True, override_cell_type=override_cell_type,
        )
        score, feats = self.flag_scorer(nodes, pr)
        result = LabelResult(
            file=Path(swc_path).name,
            cell_type=pr.stage1.cell_type,
            node_labels=list(pr.node_labels),
            flag_score=float(score),
            flagged=float(score) >= self.flag_threshold,
            flag_features=(list(feats) if feats is not None else None),
        )
        return result, pr

    # -- Step 5: curator submits corrected labels (NO good/bad choice) -----

    def submit_correction(
        self,
        label_result: LabelResult,
        corrected_swc_path: str | Path,
        timestamp: float,
        note: str = "",
    ) -> VerifiedEntry:
        """Ingest the curator's corrected labels for a reviewed cell.

        The curator supplies only labels (a corrected SWC — which may be
        byte-identical to the model's output if they judged it fine). This
        method DERIVES whether the flag was justified by comparing the
        curator's labels to the model's original labels, then stores both
        the labeler signal (the trusted SWC) and the flag signal
        (``flag_features`` labeled by the derived truth).
        """
        curator_nodes = parse_swc(Path(corrected_swc_path))
        curator_labels = [n.type for n in curator_nodes]
        justified, agree_f1 = derive_flag_truth(
            label_result.node_labels, curator_labels,
            label_result.cell_type, self.bad_f1_threshold,
        )
        return self.store.add(
            file=label_result.file,
            cell_type=label_result.cell_type,
            corrected_swc_path=corrected_swc_path,
            flag_was_justified=justified,
            agreement_f1=agree_f1,
            flag_features=label_result.flag_features,
            timestamp=timestamp,
            note=note,
        )

    def skip(self, label_result: LabelResult) -> None:
        """Curator declined to review — record nothing."""
        return None

    # -- Step 6: versioned retrain (factory preserved) ---------------------

    def retrain_new_version(
        self,
        version_name: str,
        base_labeler_train_files: dict[str, list[Path]],
        timestamp: float,
        flag_trainer: Optional[FlagTrainer] = None,
        base_flag_examples: Optional[tuple[list[list[float]], list[int]]] = None,
        note: str = "",
    ) -> ModelVersion:
        """Retrain labeler (Stage 2) and, if a trainer is given, the flag model.

        Writes a NEW complete checkpoint under the registry; the factory (and
        the parent version) are never modified. Sets ``active_version`` to the
        new version so subsequent labeling uses the improved model.
        """
        from hybrid.evaluate import _train_stage2  # heavy deps; local import

        self.registry.assert_factory_intact()
        version = self.registry.new_version(
            name=version_name, parent=self.active_version,
            n_curator_examples=len(self.store), timestamp=timestamp, note=note,
        )

        # --- Labeler: Stage 2 on base ∪ curator-corrected ---
        merged = {ct: list(paths) for ct, paths in base_labeler_train_files.items()}
        for ct, paths in self.store.labeler_train_files().items():
            merged.setdefault(ct, []).extend(paths)
        _train_stage2(merged, version.path / "branch_classifier.pkl")

        # --- Flag model: base ∪ curator-derived (features, truth) ---
        if flag_trainer is not None:
            cur_X, cur_y = self.store.flag_training_examples()
            X = list(base_flag_examples[0]) if base_flag_examples else []
            y = list(base_flag_examples[1]) if base_flag_examples else []
            X = X + cur_X
            y = y + cur_y
            if X:
                flag_trainer(X, y, version.path / "flag_model.joblib")

        self.active_version = version.version
        return version

    def stats(self) -> dict:
        justified = sum(1 for e in self.store.entries if e.flag_was_justified)
        false_alarms = len(self.store) - justified
        return {
            "verified_total": len(self.store),
            "flag_true_positives": justified,
            "flag_false_alarms": false_alarms,
            "active_version": self.active_version or "factory",
            "flag_threshold": self.flag_threshold,
            "bad_f1_threshold": self.bad_f1_threshold,
        }
