"""Public forest estimator classes for MPSBoost.

The classes in this module keep prediction aggregation and estimator names separate from shared
forest training, sampling, and model-container logic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..categorical import transform_categorical
from .base import MPSBoostRegressor
from .classification import MPSBoostClassifier
from .forest import _ForestMixin
from .prediction import predict_forest_proba, predict_forest_regression
from .single_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)


class RandomForestRegressor(_ForestMixin, MPSBoostRegressor):
    """Train a random forest regressor from independent native decision trees."""

    _tree_type = DecisionTreeRegressor
    _fitted_error_message = "RandomForestRegressor is not fitted"

    def predict(self, X: Any) -> NDArray[np.float32]:
        """Return the mean prediction across fitted native decision trees."""

        self._require_model()
        matrix = transform_categorical(X, getattr(self, "categorical_metadata_", None))
        return predict_forest_regression(
            self.estimators_, self.feature_subsets_, matrix, self.n_features_in_
        )


class RandomForestClassifier(_ForestMixin, MPSBoostClassifier):
    """Train a binary random forest classifier from independent native decision trees."""

    _tree_type = DecisionTreeClassifier
    _fitted_error_message = "RandomForestClassifier is not fitted"

    def _sample_rows(self, labels: NDArray[np.float32], random_state: int) -> NDArray[np.int64]:
        """Sample rows while ensuring every classifier tree sees both binary classes."""

        for attempt in range(16):
            rows = super()._sample_rows(labels, random_state + attempt)
            if np.unique(labels[rows]).size == 2:
                return rows
        return np.arange(labels.shape[0], dtype=np.int64)

    def predict_proba(self, X: Any) -> NDArray[np.float32]:
        """Return mean class probabilities across fitted native decision trees."""

        self._require_model()
        matrix = transform_categorical(X, getattr(self, "categorical_metadata_", None))
        return predict_forest_proba(
            self.estimators_, self.feature_subsets_, matrix, self.n_features_in_
        )


class ExtraTreesRegressor(RandomForestRegressor):
    """Train an ExtraTrees-style regressor from native random-threshold trees."""

    _tree_type = ExtraTreeRegressor
    _fitted_error_message = "ExtraTreesRegressor is not fitted"


class ExtraTreesClassifier(RandomForestClassifier):
    """Train an ExtraTrees-style binary classifier from native random-threshold trees."""

    _tree_type = ExtraTreeClassifier
    _fitted_error_message = "ExtraTreesClassifier is not fitted"
