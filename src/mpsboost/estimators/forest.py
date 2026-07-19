"""Forest and random-split ensemble estimators for MPSBoost.

This module owns row sampling, feature sampling, independent tree scheduling, forest model I/O,
feature-importance aggregation, and prediction aggregation. Individual tree math remains in the
shared native single-tree path.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..matrix import as_dense_matrix, as_sample_weight
from .base import MPSBoostRegressor
from .classification import MPSBoostClassifier
from .errors import NotFittedError
from .single_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)


from .forest_io import ForestPersistenceMixin
from .forest_sampling import ForestSamplingMixin
class _ForestMixin(ForestPersistenceMixin, ForestSamplingMixin):
    """Shared random-forest training, prediction, and importance logic."""

    _tree_type: type[DecisionTreeRegressor] | type[DecisionTreeClassifier]
    _PARAMETER_NAMES = (
        "n_estimators",
        "max_depth",
        "max_bins",
        "min_child_weight",
        "min_samples_leaf",
        "reg_lambda",
        "max_features",
        "sample_fraction",
        "bootstrap",
        "n_jobs",
        "random_state",
        "device",
        "verbosity",
    )

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        max_bins: int = 256,
        min_child_weight: float = 1.0,
        min_samples_leaf: int = 20,
        reg_lambda: float = 1.0,
        max_features: float = 1.0,
        sample_fraction: float = 1.0,
        bootstrap: bool = True,
        n_jobs: int = 1,
        random_state: int | None = None,
        device: str = "mps",
        verbosity: int = 1,
    ) -> None:
        """Store forest parameters without allocating trees or device resources."""

        super().__init__(
            n_estimators=n_estimators,
            learning_rate=1.0,
            max_depth=max_depth,
            max_bins=max_bins,
            min_child_weight=min_child_weight,
            min_samples_leaf=min_samples_leaf,
            reg_lambda=reg_lambda,
            random_state=random_state,
            device=device,
            verbosity=verbosity,
        )
        self.max_features = max_features
        self.sample_fraction = sample_fraction
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any = None,
    ) -> "_ForestMixin":
        """Train independent native decision trees on sampled rows and feature subsets."""

        if not self._fit_lock.acquire(blocking=False):
            raise RuntimeError("concurrent fit is not supported for one estimator")
        try:
            self._validate_forest_parameters()
            matrix = as_dense_matrix(X)
            labels = self._training_labels(y, matrix.shape[0])
            weights = as_sample_weight(sample_weight, matrix.shape[0])
            generator = np.random.default_rng(self.random_state)
            jobs = tuple(
                (
                    int(generator.integers(0, np.iinfo(np.int32).max)),
                    int(generator.integers(0, np.iinfo(np.int32).max)),
                )
                for _ in range(self.n_estimators)
            )
            if self.n_jobs == 1:
                fitted = tuple(
                    self._fit_one_tree(matrix, labels, weights, row_seed, feature_seed)
                    for row_seed, feature_seed in jobs
                )
            else:
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    fitted = tuple(
                        executor.map(
                            lambda seeds: self._fit_one_tree(
                                matrix, labels, weights, seeds[0], seeds[1]
                            ),
                            jobs,
                        )
                    )
            estimators, feature_subsets = zip(*fitted, strict=True)
            self.estimators_ = tuple(estimators)
            self.feature_subsets_ = tuple(feature_subsets)
            self.n_features_in_ = matrix.shape[1]
            self.n_estimators_ = len(estimators)
            self.device_ = self.device
            self.training_summary_ = {
                "n_estimators": len(estimators),
                "bootstrap": self.bootstrap,
                "sample_fraction": float(self.sample_fraction),
                "max_features": float(self.max_features),
                "n_jobs": self.n_jobs,
                "weighted": bool(sample_weight is not None),
            }
            self._finalize_fitted_metadata()
            return self
        except Exception:
            self._clear_fitted_state()
            raise
        finally:
            self._fit_lock.release()

    def feature_importance(self, kind: str = "gain") -> NDArray[np.float32]:
        """Aggregate native per-tree feature importance back into original feature space."""

        self._require_model()
        if kind not in {"gain", "split"}:
            raise ValueError("feature importance kind must be 'gain' or 'split'")
        values = np.zeros(self.n_features_in_, dtype=np.float64)
        for tree, features in zip(self.estimators_, self.feature_subsets_):
            local = tree.feature_importance(kind=kind).astype(np.float64, copy=False)
            values[features] += local
        total = float(values.sum())
        if total > 0.0:
            values /= total
        return values.astype(np.float32, copy=False)

    def _require_model(self) -> tuple[DecisionTreeRegressor | DecisionTreeClassifier, ...]:
        """Return fitted trees or raise the same stable unfitted exception contract."""

        if not hasattr(self, "estimators_"):
            raise NotFittedError(self._fitted_error_message)
        return self.estimators_

    def _clear_fitted_state(self) -> None:
        """Delete fitted forest state in addition to base estimator fields."""

        super()._clear_fitted_state()
        for name in ("estimators_", "feature_subsets_"):
            self.__dict__.pop(name, None)
