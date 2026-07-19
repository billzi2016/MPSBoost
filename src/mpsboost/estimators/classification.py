"""Binary classification estimators for MPSBoost.

This module adapts the shared native boosting base class to strict binary 0/1 labels, probability
prediction, weighted accuracy, and sklearn classifier tags. It does not implement tree math.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..matrix import as_labels, as_sample_weight
from .base import MPSBoostRegressor


class MPSBoostClassifier(MPSBoostRegressor):
    """Train a binary-logistic histogram GBDT classifier on the shared native backend."""

    _estimator_type = "classifier"
    _native_objective = "binary_logistic"
    _fitted_error_message = "MPSBoostClassifier is not fitted or loaded"

    def _training_labels(self, y: Any, n_samples: int) -> NDArray[np.float32]:
        """Validate and encode strict binary 0/1 labels for the native logistic objective."""

        labels = as_labels(y, n_samples)
        if not np.all((labels == 0.0) | (labels == 1.0)):
            raise ValueError("binary classification labels must be exactly 0 and 1")
        if np.unique(labels).size != 2:
            raise ValueError("binary classification requires both class 0 and class 1")
        return labels

    def _finalize_fitted_metadata(self) -> None:
        """Store the fixed binary class mapping used by the current model format."""

        self.classes_ = np.array([0, 1], dtype=np.int64)

    def predict_proba(self, X: Any) -> NDArray[np.float32]:
        """Return binary probabilities from native raw margins using stable sigmoid semantics."""

        margins = self._predict_raw(X).astype(np.float64, copy=False)
        probabilities = np.empty_like(margins, dtype=np.float64)
        positive = margins >= 0.0
        probabilities[positive] = 1.0 / (1.0 + np.exp(-margins[positive]))
        exp_margin = np.exp(margins[~positive])
        probabilities[~positive] = exp_margin / (1.0 + exp_margin)
        return np.column_stack((1.0 - probabilities, probabilities)).astype(np.float32)

    def predict(self, X: Any) -> NDArray[np.int64]:
        """Return class labels by thresholding class-1 probability at 0.5."""

        probabilities = self.predict_proba(X)[:, 1]
        return self.classes_[(probabilities >= 0.5).astype(np.int64)]

    def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:
        """Return binary classification accuracy for sklearn model-selection tools."""

        predictions = self.predict(X)
        labels = as_labels(y, predictions.shape[0])
        if not np.all((labels == 0.0) | (labels == 1.0)):
            raise ValueError("binary classification labels must be exactly 0 and 1")
        weights = as_sample_weight(sample_weight, predictions.shape[0])
        correct = predictions == labels.astype(np.int64)
        return float(np.average(correct.astype(np.float64), weights=weights))

    def _more_tags(self) -> dict[str, Any]:
        """Return old-style sklearn classifier tags without importing sklearn."""

        tags = super()._more_tags()
        tags["binary_only"] = True
        return tags

    def __sklearn_tags__(self) -> Any:
        """Return sklearn 1.6+ structured classifier tags while keeping sklearn optional."""

        try:
            from sklearn.utils import ClassifierTags, InputTags, Tags, TargetTags
        except ImportError as exc:  # pragma: no cover - only reachable when sklearn is absent.
            raise AttributeError("sklearn tag classes are unavailable") from exc
        return Tags(
            estimator_type="classifier",
            target_tags=TargetTags(required=True, one_d_labels=True, single_output=True),
            classifier_tags=ClassifierTags(poor_score=False),
            input_tags=InputTags(two_d_array=True, allow_nan=False, sparse=False),
            requires_fit=True,
        )


