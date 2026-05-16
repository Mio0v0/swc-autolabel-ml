"""GPU-accelerated drop-in replacements for the sklearn ensemble classifiers
used in Stage 1 / Stage 2 training.

Each class below mirrors the keyword-argument signature of its sklearn
counterpart so the call sites in:

    hybrid/evaluate.py        (eval-time Stage 1)
    hybrid/train_stage1.py    (production Stage 1)
    hybrid/train_stage2.py    (subtree-owner + Stage 2 main)

can swap in via a one-line `from sklearn.ensemble import ...` -> `from
hybrid._xgb_classifiers import ...` change, with NO modification to the
constructor calls themselves.

Algorithmic mapping (kept as close as possible to sklearn behavior):

    sklearn.RandomForestClassifier
        -> XGB "random-forest mode": num_parallel_tree=n_estimators,
           n_estimators=1, subsample=0.8, colsample_bynode=0.8,
           learning_rate=1.0. Structurally a bagged ensemble of trees.

    sklearn.ExtraTreesClassifier
        -> XGB RF mode with more aggressive feature/row subsampling, to
           approximate ExtraTrees' extra-random-split behavior.

    sklearn.GradientBoostingClassifier
        -> XGB vanilla boosting (n_estimators rounds of additive trees).

    sklearn.HistGradientBoostingClassifier
        -> XGB boosting with tree_method='hist' (histogram-based, same
           algorithmic family as sklearn's HGB).

Implementation note
-------------------
We compose (have-a XGBClassifier) rather than inherit (is-a) so the public
constructor signature can use sklearn parameter names without colliding
with the parent class's own parameters of the same name. The wrappers
inherit sklearn.base.BaseEstimator / ClassifierMixin so they remain fully
sklearn-API-compatible: ``fit``, ``predict``, ``predict_proba``,
``classes_``, plus ``get_params`` / ``set_params`` / ``sklearn.base.clone``
required by VotingClassifier and CalibratedClassifierCV.

All four classes require an NVIDIA GPU (set ``device='cuda'``). They will
raise at fit time if CUDA is unavailable.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
def _label_encode(y):
    """Map arbitrary integer labels (e.g. {2,3,4}) to {0..N-1}.

    XGBoost 3.x rejects non-contiguous labels. We encode at fit time and
    decode at predict time so the wrapper behaves identically to sklearn
    classifiers from the caller's point of view (``classes_`` and the
    return value of ``predict`` are in the original label space).
    """
    y_arr = np.asarray(y)
    classes = np.unique(y_arr)
    lookup = {c: i for i, c in enumerate(classes.tolist())}
    enc = np.array([lookup[v] for v in y_arr.tolist()], dtype=np.int64)
    return enc, classes


def _switch_inference_to_cpu(model: xgb.XGBClassifier) -> None:
    """Flip the booster's runtime device to CPU after training on GPU.

    GPU launch overhead (~1-5 ms per call) dominates for the small per-cell
    predict batches the pipeline issues (~2000 branches at a time). Training
    keeps the GPU win on the big batched fit; inference switches to CPU
    where small predicts are essentially free. Same weights, same
    predictions, just faster.
    """
    try:
        model.get_booster().set_param({"device": "cpu"})
    except Exception:
        # If anything goes wrong, leave the model as-is; predictions will
        # still be correct, just slower.
        pass


# ---------------------------------------------------------------------------
# Random Forest (XGBoost "random forest mode")
# ---------------------------------------------------------------------------
class XGBRandomForestClassifier(BaseEstimator, ClassifierMixin):
    """GPU drop-in for sklearn.ensemble.RandomForestClassifier.

    Algorithm: XGBoost RF mode. One boosting round in which
    ``num_parallel_tree`` independent trees are fit in parallel (bagging),
    with ``subsample`` and ``colsample_bynode`` introducing per-tree
    randomness. Structurally a bagged ensemble of trees like sklearn RF.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        class_weight: Any = None,
        random_state: int = 42,
        n_jobs: int | None = None,   # ignored on GPU
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

    # Subclasses override this to add extra randomness etc.
    def _make_xgb(self) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            n_estimators=1,                                       # 1 boosting round
            num_parallel_tree=self.n_estimators,                  # N trees in parallel
            max_depth=0 if self.max_depth is None else self.max_depth,
            min_child_weight=self.min_samples_leaf,
            subsample=0.8,
            colsample_bynode=0.8,
            learning_rate=1.0,
            device="cuda",
            tree_method="hist",
            random_state=self.random_state,
            verbosity=0,
        )

    def fit(self, X, y, sample_weight=None):
        y_enc, classes = _label_encode(y)
        if self.class_weight == "balanced" and sample_weight is None:
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weight = compute_sample_weight(class_weight="balanced", y=np.asarray(y))
        self._model_ = self._make_xgb()
        self._model_.fit(X, y_enc, sample_weight=sample_weight)
        _switch_inference_to_cpu(self._model_)
        self.classes_ = classes
        return self

    def predict(self, X):
        enc = self._model_.predict(X)
        return self.classes_[np.asarray(enc).astype(int)]

    def predict_proba(self, X):
        return self._model_.predict_proba(X)


# ---------------------------------------------------------------------------
# Extra Trees (RF + more aggressive randomness)
# ---------------------------------------------------------------------------
class XGBExtraTreesClassifier(XGBRandomForestClassifier):
    """GPU drop-in for sklearn.ensemble.ExtraTreesClassifier.

    sklearn ExtraTrees picks split thresholds randomly rather than
    optimally. XGBoost has no direct "random-split" mode; we approximate
    via more aggressive row + column subsampling.
    """

    def _make_xgb(self) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            n_estimators=1,
            num_parallel_tree=self.n_estimators,
            max_depth=0 if self.max_depth is None else self.max_depth,
            min_child_weight=self.min_samples_leaf,
            subsample=0.7,                                        # more aggressive
            colsample_bynode=0.6,                                 # than RF
            learning_rate=1.0,
            device="cuda",
            tree_method="hist",
            random_state=self.random_state,
            verbosity=0,
        )


# ---------------------------------------------------------------------------
# Vanilla gradient boosting
# ---------------------------------------------------------------------------
class XGBGradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    """GPU drop-in for sklearn.ensemble.GradientBoostingClassifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        min_samples_leaf: int = 1,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        y_enc, classes = _label_encode(y)
        self._model_ = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_samples_leaf,
            device="cuda",
            tree_method="hist",
            random_state=self.random_state,
            verbosity=0,
        )
        self._model_.fit(X, y_enc, sample_weight=sample_weight)
        _switch_inference_to_cpu(self._model_)
        self.classes_ = classes
        return self

    def predict(self, X):
        enc = self._model_.predict(X)
        return self.classes_[np.asarray(enc).astype(int)]

    def predict_proba(self, X):
        return self._model_.predict_proba(X)


# ---------------------------------------------------------------------------
# Histogram gradient boosting
# ---------------------------------------------------------------------------
class XGBHistGradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    """GPU drop-in for sklearn.ensemble.HistGradientBoostingClassifier.

    Note: sklearn HGB uses ``max_iter`` for the number of boosting rounds,
    not ``n_estimators``. We accept the sklearn keyword and map it to
    XGBoost's ``n_estimators`` internally.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        min_samples_leaf: int = 20,
        random_state: int = 42,
    ) -> None:
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        y_enc, classes = _label_encode(y)
        self._model_ = xgb.XGBClassifier(
            n_estimators=self.max_iter,
            max_depth=0 if self.max_depth is None else self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_samples_leaf,
            device="cuda",
            tree_method="hist",
            random_state=self.random_state,
            verbosity=0,
        )
        self._model_.fit(X, y_enc, sample_weight=sample_weight)
        _switch_inference_to_cpu(self._model_)
        self.classes_ = classes
        return self

    def predict(self, X):
        enc = self._model_.predict(X)
        return self.classes_[np.asarray(enc).astype(int)]

    def predict_proba(self, X):
        return self._model_.predict_proba(X)
