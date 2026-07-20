"""Multiclass classifier built on the shared binary native objective.

The native backend currently exposes a stable binary-logistic objective. This
module adds sklearn-style multiclass behavior with one-vs-rest binary models
instead of pretending that the binary objective is natively multiclass. Binary
0/1 data still uses the direct native classifier path.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
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
        encoded = np.searchsorted(classes, labels)
        if classes.size == 2:
            self._multiclass_strategy_ = "binary_logistic"
            fitted = super().fit(X, encoded, sample_weight=sample_weight)
            fitted.classes_ = classes
            return fitted

        self._validate_parameters()
        weights = as_sample_weight(sample_weight, labels.shape[0])
        worker_count = self._resolved_ovr_jobs(classes.size)

        def train_binary(index: int) -> _BinaryMPSBoostClassifier:
            """Train one class-vs-rest native binary model with deterministic seeding."""

            estimator = self._make_binary_estimator(index)
            binary_labels = (encoded == index).astype(np.float64)
            estimator.fit(X, binary_labels, sample_weight=weights)
            return estimator

        if worker_count == 1:
            estimators = [train_binary(index) for index in range(classes.size)]
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                estimators = list(executor.map(train_binary, range(classes.size)))
        self.classes_ = classes.copy()
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
            "n_jobs": worker_count,
            "weighted": bool(sample_weight is not None),
        }
        return self

    def predict_proba(self, X: Any) -> NDArray[np.float32]:
        """Return binary probabilities or normalized one-vs-rest probabilities."""

        if getattr(self, "_multiclass_strategy_", None) != "one_vs_rest":
            return super().predict_proba(X)
        self._require_multiclass_estimators()
        scores = self.decision_function(X)
        scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -709.0, 709.0)))
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
        correct = predictions == labels
        return float(np.average(correct.astype(np.float64), weights=weights))

    def decision_function(self, X: Any) -> NDArray[np.float32]:
        """Return binary margins or one-vs-rest class margins."""

        if getattr(self, "_multiclass_strategy_", None) != "one_vs_rest":
            return self._predict_raw(X).astype(np.float32, copy=False)
        self._require_multiclass_estimators()
        return np.column_stack(
            [estimator._predict_raw(X) for estimator in self.estimators_]
        ).astype(np.float32, copy=False)

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

    def _resolved_ovr_jobs(self, n_classes: int) -> int:
        """Normalize sklearn-style n_jobs for class-parallel OvR training."""

        if self.n_jobs is None:
            return 1
        if self.n_jobs == -1:
            return min(n_classes, os.cpu_count() or 1)
        return min(n_classes, int(self.n_jobs))

    def _make_binary_estimator(self, class_index: int) -> _BinaryMPSBoostClassifier:
        """Build the binary native estimator used by one OvR branch."""

        estimator = _BinaryMPSBoostClassifier(**self.get_params())
        estimator.random_state = (
            None if self.random_state is None else int(self.random_state) + class_index
        )
        return estimator
