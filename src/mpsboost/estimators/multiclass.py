"""Multiclass classifier built on the shared binary native objective.

The native backend currently exposes a stable binary-logistic objective. This
module adds sklearn-style multiclass behavior with one-vs-rest binary models
instead of pretending that the binary objective is natively multiclass. Binary
0/1 data still uses the direct native classifier path.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .. import _native
from ..categorical import fit_transform_categorical, transform_categorical
from ..matrix import as_labels, as_sample_weight
from .classification import MPSBoostClassifier as _BinaryMPSBoostClassifier
from .errors import NotFittedError


class MPSBoostClassifier(_BinaryMPSBoostClassifier):
    """Train binary or multiclass gradient boosting classifiers."""

    _fitted_error_message = "MPSBoostClassifier is not fitted or loaded"
    _PARAMETER_NAMES = _BinaryMPSBoostClassifier._PARAMETER_NAMES + (
        "multi_strategy",
    )

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        max_bins: int = 256,
        growth_strategy: str = "level_wise",
        max_leaves: int | None = None,
        max_active_leaves: int | None = None,
        min_gain_to_split: float = 0.0,
        loss: str = "squared_error",
        quantile_alpha: float = 0.5,
        tweedie_variance_power: float = 1.5,
        min_child_weight: float = 1.0,
        min_samples_leaf: int = 20,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        max_delta_step: float = 0.0,
        monotonic_constraints: Any = None,
        interaction_constraints: Any = None,
        categorical_features: Any = None,
        random_state: int | None = None,
        n_jobs: int | None = None,
        device: str = "mps",
        verbosity: int = 1,
        multi_strategy: str = "auto",
    ) -> None:
        """Store classifier parameters while keeping sklearn constructor introspection stable."""

        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_bins=max_bins,
            growth_strategy=growth_strategy,
            max_leaves=max_leaves,
            max_active_leaves=max_active_leaves,
            min_gain_to_split=min_gain_to_split,
            loss=loss,
            quantile_alpha=quantile_alpha,
            tweedie_variance_power=tweedie_variance_power,
            min_child_weight=min_child_weight,
            min_samples_leaf=min_samples_leaf,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            max_delta_step=max_delta_step,
            monotonic_constraints=monotonic_constraints,
            interaction_constraints=interaction_constraints,
            categorical_features=categorical_features,
            random_state=random_state,
            n_jobs=n_jobs,
            device=device,
            verbosity=verbosity,
        )
        self.multi_strategy = multi_strategy

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
        if self.multi_strategy not in {"auto", "softmax", "ovr"}:
            raise ValueError("multi_strategy must be 'auto', 'softmax', or 'ovr'")
        encoded = np.searchsorted(classes, labels)
        if classes.size == 2:
            self._multiclass_strategy_ = "binary_logistic"
            fitted = super().fit(X, encoded, sample_weight=sample_weight)
            fitted.classes_ = classes
            return fitted

        self._validate_parameters()
        weights = as_sample_weight(sample_weight, labels.shape[0])
        if self._should_use_native_softmax():
            return self._fit_native_softmax(X, encoded, classes, weights, sample_weight)
        if self.multi_strategy == "softmax":
            warnings.warn(
                "Compatibility strategy selected: one-vs-rest multiclass is used for this "
                'device="mps" request so the run remains executable on the requested backend.',
                RuntimeWarning,
                stacklevel=2,
            )
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
            "requested_strategy": self.multi_strategy,
            "classes": self.classes_.tolist(),
            "n_classes": int(classes.size),
            "n_estimators": self.n_estimators_,
            "n_jobs": worker_count,
            "weighted": bool(sample_weight is not None),
        }
        return self

    def predict_proba(self, X: Any) -> NDArray[np.float32]:
        """Return binary probabilities or normalized one-vs-rest probabilities."""

        if getattr(self, "_multiclass_strategy_", None) == "native_softmax":
            margins = self._native_softmax_margins(X)
            shifted = margins - margins.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            return probabilities.astype(np.float32, copy=False)
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

        if getattr(self, "_multiclass_strategy_", None) == "native_softmax":
            probabilities = self.predict_proba(X)
            return self.classes_[np.argmax(probabilities, axis=1)]
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

        if getattr(self, "_multiclass_strategy_", None) == "native_softmax":
            return self._native_softmax_margins(X).astype(np.float32, copy=False)
        if getattr(self, "_multiclass_strategy_", None) != "one_vs_rest":
            return self._predict_raw(X).astype(np.float32, copy=False)
        self._require_multiclass_estimators()
        return np.column_stack(
            [estimator._predict_raw(X) for estimator in self.estimators_]
        ).astype(np.float32, copy=False)

    def feature_importance(self, kind: str = "gain") -> NDArray[np.float32]:
        """Average OvR feature importance across class-specific binary models."""

        if getattr(self, "_multiclass_strategy_", None) == "native_softmax":
            warnings.warn(
                "Native softmax feature-importance metadata is unavailable in the compact "
                "multiclass model record; returning zeros. Use permutation_importance(X, y) "
                "for model-agnostic multiclass explanations.",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.zeros(self.n_features_in_, dtype=np.float32)
        if getattr(self, "_multiclass_strategy_", None) != "one_vs_rest":
            return super().feature_importance(kind=kind)
        self._require_multiclass_estimators()
        values = np.zeros(self.n_features_in_, dtype=np.float64)
        for estimator in self.estimators_:
            values += estimator.feature_importance(kind=kind).astype(np.float64, copy=False)
        values /= float(len(self.estimators_))
        return values.astype(np.float32)

    def save_model(self, path: Any) -> None:
        """Save binary or native softmax models in the versioned native format."""

        if getattr(self, "_multiclass_strategy_", None) == "native_softmax":
            if getattr(self, "categorical_metadata_", None) is not None:
                raise NotImplementedError(
                    "categorical model persistence requires categorical metadata support"
                )
            self._require_model().save(str(path))
            return
        if getattr(self, "_multiclass_strategy_", None) == "one_vs_rest":
            raise NotImplementedError("multiclass model persistence requires container support")
        super().save_model(path)

    def load_model(self, path: Any) -> "MPSBoostClassifier":
        """Load binary or native softmax classifier models."""

        acquired = self._fit_lock.acquire(blocking=False)
        if not acquired:
            raise RuntimeError("Model training or loading is already in progress")
        try:
            try:
                candidate = _native._load_multiclass_model(str(path))
            except ValueError as exc:
                if "multiclass loader" not in str(exc):
                    raise
                self._fit_lock.release()
                acquired = False
                self._multiclass_strategy_ = "binary_logistic"
                return super().load_model(path)
            self.model_ = candidate
            self.classes_ = np.asarray(candidate.class_labels, dtype=np.float64)
            self.n_features_in_ = candidate.feature_count
            self.device_ = self.device if self.device != "auto" else "cpu"
            self.n_estimators_ = candidate.tree_count
            self._multiclass_strategy_ = "native_softmax"
            self.training_summary_ = {
                "loaded": True,
                "device": self.device_,
                "strategy": "native_softmax",
                "n_classes": int(candidate.class_count),
            }
            return self
        finally:
            if acquired:
                self._fit_lock.release()

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

    def _should_use_native_softmax(self) -> bool:
        """Return whether this fit should use the native CPU softmax trainer."""

        if self.device not in {"cpu", "auto"}:
            return False
        return self.multi_strategy in {"auto", "softmax"}

    def _fit_native_softmax(
        self,
        X: Any,
        encoded: NDArray[np.int64],
        classes: NDArray[Any],
        weights: NDArray[np.float64],
        sample_weight: Any,
    ) -> "MPSBoostClassifier":
        """Fit the native CPU multiclass softmax model."""

        matrix, categorical_metadata = fit_transform_categorical(
            X, self.categorical_features, encoded.astype(np.float64), weights
        )
        parameters = _native._TrainingParameters(
            self.n_estimators,
            self.learning_rate,
            self.max_bins,
            self.max_depth,
            self.min_samples_leaf,
            self.min_child_weight,
            self.reg_lambda,
            reg_alpha=self.reg_alpha,
            max_delta_step=self.max_delta_step,
            min_gain_to_split=self.min_gain_to_split,
            objective="squared_error",
            split_strategy=self._split_strategy,
            growth_strategy=self.growth_strategy,
            max_leaves=0 if self.max_leaves is None else self.max_leaves,
            max_active_leaves=(
                0 if self.max_active_leaves is None else self.max_active_leaves
            ),
            random_seed=0 if self.random_state is None else self.random_state,
            monotonic_constraints=self._normalized_monotonic_constraints(
                matrix.shape[1]
            ),
            interaction_constraints=self._normalized_interaction_constraints(
                matrix.shape[1]
            ),
        )
        started = perf_counter()
        model = _native._train_multiclass_softmax_cpu(
            matrix,
            encoded.astype(np.float64, copy=False),
            weights,
            parameters,
            int(classes.size),
        )
        model.set_class_labels(classes.astype(np.float64, copy=False).tolist())
        self.model_ = model
        self.classes_ = classes.copy()
        self.n_features_in_ = matrix.shape[1]
        self.categorical_metadata_ = categorical_metadata
        self.device_ = "cpu"
        self.n_estimators_ = model.tree_count
        self._multiclass_strategy_ = "native_softmax"
        self.training_summary_ = {
            "strategy": "native_softmax",
            "classes": self.classes_.tolist(),
            "n_classes": int(classes.size),
            "n_estimators": self.n_estimators_,
            "fit_seconds": perf_counter() - started,
            "weighted": bool(sample_weight is not None),
        }
        return self

    def _native_softmax_margins(self, X: Any) -> NDArray[np.float64]:
        """Predict row-major native softmax margins and reshape them by class."""

        if not hasattr(self, "model_") or not hasattr(self, "classes_"):
            raise NotFittedError(self._fitted_error_message)
        matrix = transform_categorical(X, getattr(self, "categorical_metadata_", None))
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("prediction feature count does not match training data")
        margins = np.asarray(self.model_.predict_margins(matrix), dtype=np.float64)
        return margins.reshape((-1, int(len(self.classes_))))

    def _make_binary_estimator(self, class_index: int) -> _BinaryMPSBoostClassifier:
        """Build the binary native estimator used by one OvR branch."""

        parameters = {
            name: value
            for name, value in self.get_params().items()
            if name != "multi_strategy"
        }
        estimator = _BinaryMPSBoostClassifier(**parameters)
        estimator.random_state = (
            None if self.random_state is None else int(self.random_state) + class_index
        )
        return estimator
