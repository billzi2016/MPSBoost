"""Multiclass classifier built on the shared binary native objective.

The native backend currently exposes a stable binary-logistic objective. This
module adds sklearn-style multiclass behavior with one-vs-rest binary models
instead of pretending that the binary objective is natively multiclass. Binary
0/1 data still uses the direct native classifier path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..matrix import as_labels, as_sample_weight
from .classification import MPSBoostClassifier as _BinaryMPSBoostClassifier
from .errors import NotFittedError


class MPSBoostClassifier(_BinaryMPSBoostClassifier):
    """Train binary or multiclass gradient boosting classifiers."""

    _fitted_error_message = "MPSBoostClassifier is not fitted or loaded"

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any = None,
    ) -> "MPSBoostClassifier":
        """Fit direct binary logistic or one-vs-rest multiclass models."""

        labels = as_labels(y, np.asarray(X).shape[0])
        classes = np.unique(labels)
        if classes.size < 2:
            raise ValueError("classification requires at least two classes")
        if classes.size == 2 and np.array_equal(classes, np.asarray([0.0, 1.0])):
            self._multiclass_strategy_ = "binary_logistic"
            fitted = super().fit(X, y, sample_weight=sample_weight)
            fitted.classes_ = fitted.classes_.astype(np.int64, copy=False)
            return fitted

        self._validate_parameters()
        weights = as_sample_weight(sample_weight, labels.shape[0])
        estimators: list[_BinaryMPSBoostClassifier] = []
        for index, class_value in enumerate(classes):
            estimator = _BinaryMPSBoostClassifier(**self.get_params())
            estimator.random_state = (
                None if self.random_state is None else int(self.random_state) + index
            )
            binary_labels = (labels == class_value).astype(np.float64)
            estimator.fit(X, binary_labels, sample_weight=weights)
            estimators.append(estimator)
        self.classes_ = classes.astype(np.int64, copy=False)
        self.estimators_ = tuple(estimators)
        self.n_features_in_ = estimators[0].n_features_in_
        self.n_estimators_ = sum(item.n_estimators_ for item in estimators)
        self.device_ = estimators[0].device_
        self._multiclass_strategy_ = "one_vs_rest"
        self.training_summary_ = {
            "strategy": "one_vs_rest",
            "classes": self.classes_.tolist(),
            "n_classes": int(classes.size),
            "n_estimators": self.n_estimators_,
            "weighted": bool(sample_weight is not None),
        }
        return self

    def predict_proba(self, X: Any) -> NDArray[np.float32]:
        """Return binary probabilities or normalized one-vs-rest probabilities."""

        if getattr(self, "_multiclass_strategy_", None) != "one_vs_rest":
            return super().predict_proba(X)
        self._require_multiclass_estimators()
        scores = np.column_stack([estimator.predict_proba(X)[:, 1] for estimator in self.estimators_])
        row_sums = scores.sum(axis=1, keepdims=True)
        zero_rows = row_sums[:, 0] <= 0.0
        if np.any(zero_rows):
            scores[zero_rows, :] = 1.0 / float(len(self.estimators_))
            row_sums = scores.sum(axis=1, keepdims=True)
        return (scores / row_sums).astype(np.float32)

    def predict(self, X: Any) -> NDArray[np.int64]:
        """Return class labels from maximum probability."""

        if getattr(self, "_multiclass_strategy_", None) != "one_vs_rest":
            return super().predict(X)
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:
        """Return accuracy for binary and multiclass classification."""

        predictions = self.predict(X)
        labels = as_labels(y, predictions.shape[0])
        weights = as_sample_weight(sample_weight, predictions.shape[0])
        correct = predictions == labels.astype(np.int64)
        return float(np.average(correct.astype(np.float64), weights=weights))

    def feature_importance(self, kind: str = "gain") -> NDArray[np.float32]:
        """Average OvR feature importance across class-specific binary models."""

        if getattr(self, "_multiclass_strategy_", None) != "one_vs_rest":
            return super().feature_importance(kind=kind)
        self._require_multiclass_estimators()
        values = np.zeros(self.n_features_in_, dtype=np.float64)
        for estimator in self.estimators_:
            values += estimator.feature_importance(kind=kind).astype(np.float64, copy=False)
        values /= float(len(self.estimators_))
        return values.astype(np.float32)

    def save_model(self, path: Any) -> None:
        """Save binary models and reject OvR until the container format is extended."""

        if getattr(self, "_multiclass_strategy_", None) == "one_vs_rest":
            raise NotImplementedError("multiclass model persistence requires OvR container support")
        super().save_model(path)

    def load_model(self, path: Any) -> "MPSBoostClassifier":
        """Load binary native classifier models."""

        self._multiclass_strategy_ = "binary_logistic"
        return super().load_model(path)

    def _require_multiclass_estimators(self) -> tuple[_BinaryMPSBoostClassifier, ...]:
        """Return fitted OvR estimators or raise the stable not-fitted error."""

        if not hasattr(self, "estimators_"):
            raise NotFittedError(self._fitted_error_message)
        return self.estimators_
