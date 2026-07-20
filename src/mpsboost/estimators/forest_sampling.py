"""Sampling and per-tree fitting helpers for forest estimators.

The mixin owns row sampling, feature sampling, sampled-weight slicing, and forest-specific parameter
validation. Forest training orchestration imports it instead of carrying all helper methods inline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..randomization import (
    bootstrap_sample_indices,
    sample_without_replacement_indices,
    subsample_feature_indices,
)
from .single_tree import DecisionTreeClassifier, DecisionTreeRegressor


class ForestSamplingMixin:
    def _make_tree(self, random_state: int) -> DecisionTreeRegressor | DecisionTreeClassifier:
        """Create one native decision tree with the forest's tree parameters."""

        return self._tree_type(
            max_depth=self.max_depth,
            max_bins=self.max_bins,
            min_child_weight=self.min_child_weight,
            min_samples_leaf=self.min_samples_leaf,
            reg_lambda=self.reg_lambda,
            categorical_features=None,
            random_state=random_state,
            device=self.device,
            verbosity=self.verbosity,
        )

    def _fit_one_tree(
        self,
        matrix: NDArray[np.float32],
        labels: NDArray[np.float32],
        sample_weights: NDArray[np.float64],
        row_seed: int,
        feature_seed: int,
    ) -> tuple[DecisionTreeRegressor | DecisionTreeClassifier, NDArray[np.int64]]:
        """Fit one independent tree and return it with its original feature subset."""

        rows = self._sample_rows(labels, row_seed)
        features = self._sample_features(matrix.shape[1], feature_seed)
        tree = self._make_tree(feature_seed).fit(
            matrix[rows][:, features],
            labels[rows],
            sample_weight=sample_weights[rows],
        )
        return tree, features

    def _sample_rows(self, labels: NDArray[np.float32], random_state: int) -> NDArray[np.int64]:
        """Sample training rows for one tree."""

        if self.bootstrap:
            return bootstrap_sample_indices(
                labels.shape[0],
                sample_fraction=float(self.sample_fraction),
                random_state=random_state,
            )
        return sample_without_replacement_indices(
            labels.shape[0],
            sample_fraction=float(self.sample_fraction),
            random_state=random_state,
        )

    def _sample_features(self, n_features: int, random_state: int) -> NDArray[np.int64]:
        """Sample feature columns for one tree."""

        return subsample_feature_indices(
            n_features,
            feature_fraction=float(self.max_features),
            random_state=random_state,
        )

    def _validate_forest_parameters(self) -> None:
        """Validate forest-specific parameters plus shared tree parameters."""

        if isinstance(self.n_estimators, bool) or not isinstance(self.n_estimators, int):
            raise TypeError("n_estimators must be an integer")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if isinstance(self.n_jobs, bool) or not isinstance(self.n_jobs, int):
            raise TypeError("n_jobs must be an integer")
        if self.n_jobs <= 0:
            raise ValueError("n_jobs must be positive")
        if not isinstance(self.bootstrap, bool):
            raise TypeError("bootstrap must be a boolean")
        for name in ("max_features", "sample_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            numeric = float(value)
            if not np.isfinite(numeric) or not 0.0 < numeric <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        saved_n_estimators = self.n_estimators
        self.learning_rate = 1.0
        try:
            super()._validate_parameters()
        finally:
            self.n_estimators = saved_n_estimators

