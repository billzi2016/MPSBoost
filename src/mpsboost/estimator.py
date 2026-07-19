"""sklearn-style estimators for MPSBoost.

This module owns parameter storage, input adaptation, concurrent-fit protection, fitted-state
replacement, and sklearn protocol methods. Quantization, boosting, Metal scheduling, and model
format logic stay in their single native implementations instead of being duplicated in Python.
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import _native
from .device_policy import choose_device, decision_to_dict
from .diagnostics import _metallib_path, is_available
from .matrix import as_dense_matrix, as_labels, as_sample_weight
from .randomization import (
    bootstrap_sample_indices,
    ordered_boosting_permutations,
    sample_without_replacement_indices,
    subsample_feature_indices,
)


class NotFittedError(RuntimeError):
    """Raised when fitted-only functionality is called before a complete model exists."""


class MPSBoostRegressor:
    """Train a squared-error histogram GBDT regressor on the CPU oracle or MPS backend.

    The constructor only stores parameters. It does not initialize devices, create caches, or
    allocate training memory. ``device="cpu"`` selects the explicit oracle/diagnostic backend;
    ``device="mps"`` fails clearly when MPS is unavailable and never silently falls back.
    """

    _estimator_type = "regressor"
    _native_objective = "squared_error"
    _split_strategy = "best_gain"
    _fitted_error_message = "MPSBoostRegressor is not fitted or loaded"
    _PARAMETER_NAMES = (
        "n_estimators",
        "learning_rate",
        "max_depth",
        "max_bins",
        "min_child_weight",
        "min_samples_leaf",
        "reg_lambda",
        "random_state",
        "device",
        "verbosity",
    )

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        max_bins: int = 256,
        min_child_weight: float = 1.0,
        min_samples_leaf: int = 20,
        reg_lambda: float = 1.0,
        random_state: int | None = None,
        device: str = "mps",
        verbosity: int = 1,
    ) -> None:
        """Store estimator parameters without expensive side effects."""

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_bins = max_bins
        self.min_child_weight = min_child_weight
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.device = device
        self.verbosity = verbosity
        self._fit_lock = Lock()

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return all constructor parameters for sklearn-style model selection.

        ``deep`` is accepted for protocol compatibility. This estimator currently has no nested
        estimators, so the returned mapping is identical for both values.
        """

        del deep
        return {name: getattr(self, name) for name in self._PARAMETER_NAMES}

    def set_params(self, **parameters: Any) -> "MPSBoostRegressor":
        """Set known constructor parameters and return ``self``.

        Unknown parameters fail early. Any real parameter change clears fitted state so
        ``get_params`` cannot describe one model while prediction uses another.
        """

        unknown = sorted(set(parameters) - set(self._PARAMETER_NAMES))
        if unknown:
            raise ValueError(f"未知参数：{', '.join(unknown)}")
        for name, value in parameters.items():
            setattr(self, name, value)
        if parameters:
            self._clear_fitted_state()
        return self

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any = None,
    ) -> "MPSBoostRegressor":
        """Train a complete model and replace fitted state only after success.

        The same estimator instance does not support concurrent ``fit`` calls. Failures clear any
        partial state so callers never observe a partially trained ensemble.
        """

        if not self._fit_lock.acquire(blocking=False):
            raise RuntimeError("同一 estimator 不支持并发 fit")
        try:
            self._validate_parameters()
            matrix = as_dense_matrix(X)
            labels = self._training_labels(y, matrix.shape[0])
            weights = as_sample_weight(sample_weight, matrix.shape[0])
            parameters = _native._TrainingParameters(
                self.n_estimators,
                self.learning_rate,
                self.max_bins,
                self.max_depth,
                self.min_samples_leaf,
                self.min_child_weight,
                self.reg_lambda,
                objective=self._native_objective,
                split_strategy=self._split_strategy,
                random_seed=0 if self.random_state is None else self.random_state,
            )
            mps_available = is_available()
            device_decision = choose_device(
                requested=self.device,
                n_samples=matrix.shape[0],
                n_features=matrix.shape[1],
                n_estimators=self.n_estimators,
                max_bins=self.max_bins,
                mps_available=mps_available,
            )
            started = perf_counter()
            if device_decision.selected == "mps":
                if not mps_available:
                    raise _native.BackendError(
                        "MPS 后端不可用；请在受支持的 Apple Silicon Mac 上运行"
                    )
                with _metallib_path() as metallib_path:
                    candidate = _native._train_regressor_mps(
                        matrix, labels, weights, parameters, metallib_path
                    )
            else:
                candidate = _native._train_regressor_cpu(
                    matrix, labels, weights, parameters
                )
            elapsed = perf_counter() - started

            # Commit fitted fields only at the end of the success path. Device errors, OOM, or
            # input errors must not leave a half-trained model behind.
            self.model_ = candidate
            self.n_features_in_ = matrix.shape[1]
            self.device_ = device_decision.selected
            self.device_decision_ = decision_to_dict(device_decision)
            self.n_estimators_ = candidate.tree_count
            self._finalize_fitted_metadata()
            self.training_summary_ = {
                "fit_seconds": elapsed,
                "input_contiguous": bool(matrix.flags.c_contiguous),
                "device": device_decision.selected,
                "device_decision": self.device_decision_,
                "n_estimators": candidate.tree_count,
                "weighted": bool(sample_weight is not None),
            }
            return self
        except Exception:
            self._clear_fitted_state()
            raise
        finally:
            self._fit_lock.release()

    def predict(self, X: Any) -> NDArray[np.float32]:
        """Return one-dimensional float32 predictions using the frozen training schema."""

        return self._predict_raw(X)

    def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:
        """Return the default regression R² score for sklearn model-selection tools."""

        predictions = self.predict(X).astype(np.float64, copy=False)
        labels = as_labels(y, predictions.shape[0]).astype(np.float64, copy=False)
        weights = as_sample_weight(sample_weight, predictions.shape[0])
        residual_sum = float(np.sum(weights * (labels - predictions) ** 2))
        mean = float(np.average(labels, weights=weights))
        centered = labels - mean
        total_sum = float(np.sum(weights * centered**2))
        if total_sum == 0.0:
            return 1.0 if residual_sum == 0.0 else 0.0
        score = 1.0 - residual_sum / total_sum
        if not np.isfinite(score):
            raise ValueError("R2 score is not finite")
        return float(score)

    @property
    def feature_importances_(self) -> NDArray[np.float32]:
        """Return normalized gain-based feature importance for sklearn-style tooling."""

        return self.feature_importance(kind="gain")

    def feature_importance(self, kind: str = "gain") -> NDArray[np.float32]:
        """Return normalized split statistics from the real trained native trees.

        ``kind="gain"`` accumulates native split gains. ``kind="split"`` counts split usage.
        Both variants read the single C++ tree representation exposed by the binding instead of
        re-implementing split logic in Python. A model with no internal split returns all zeros.
        """

        model = self._require_model()
        if kind not in {"gain", "split"}:
            raise ValueError("feature importance kind must be 'gain' or 'split'")
        values = np.zeros(self.n_features_in_, dtype=np.float64)
        for tree in model.trees:
            for node in tree.nodes:
                if node["is_leaf"]:
                    continue
                feature_index = int(node["feature_index"])
                if kind == "gain":
                    values[feature_index] += max(float(node["gain"]), 0.0)
                else:
                    values[feature_index] += 1.0
        total = float(values.sum())
        if total > 0.0:
            values /= total
        return values.astype(np.float32, copy=False)

    def permutation_importance(
        self,
        X: Any,
        y: Any,
        *,
        n_repeats: int = 5,
        random_state: int | None = None,
    ) -> dict[str, NDArray[np.float32] | float]:
        """Estimate score drop after permuting each feature with the estimator's own score.

        This method intentionally delegates all prediction and scoring semantics to ``score``.
        Regression therefore uses R², classification uses accuracy, and future estimators can reuse
        the same implementation without copying metric logic into the explanation layer.
        """

        self._require_model()
        if isinstance(n_repeats, bool) or not isinstance(n_repeats, int):
            raise TypeError("n_repeats must be an integer")
        if n_repeats <= 0:
            raise ValueError("n_repeats must be positive")
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, int)
        ):
            raise TypeError("random_state must be an integer or None")
        matrix = as_dense_matrix(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("permutation feature count does not match training data")
        baseline_score = float(self.score(matrix, y))
        generator = np.random.default_rng(random_state)
        importances = np.zeros((self.n_features_in_, n_repeats), dtype=np.float32)
        for feature_index in range(self.n_features_in_):
            for repeat_index in range(n_repeats):
                permuted = np.array(matrix, copy=True)
                order = generator.permutation(matrix.shape[0])
                permuted[:, feature_index] = permuted[order, feature_index]
                importances[feature_index, repeat_index] = baseline_score - float(
                    self.score(permuted, y)
                )
        return {
            "baseline_score": baseline_score,
            "importances": importances,
            "importances_mean": importances.mean(axis=1),
            "importances_std": importances.std(axis=1),
        }

    def _more_tags(self) -> dict[str, Any]:
        """Return sklearn compatibility tags without importing sklearn at runtime."""

        return {
            "X_types": ["2darray"],
            "allow_nan": False,
            "requires_y": True,
            "poor_score": False,
        }

    def __sklearn_tags__(self) -> Any:
        """Return sklearn 1.6+ structured tags while keeping sklearn optional.

        sklearn 1.6 changed several meta-estimator paths from the historical ``_more_tags``
        dictionary to a structured ``Tags`` object. Importing those classes at module import time
        would make sklearn a hard runtime dependency for normal MPSBoost users, so the import stays
        inside this compatibility hook and only runs when sklearn itself asks for tags.
        """

        try:
            from sklearn.utils import InputTags, RegressorTags, Tags, TargetTags
        except ImportError as exc:  # pragma: no cover - only reachable when sklearn is absent.
            raise AttributeError("sklearn tag classes are unavailable") from exc
        return Tags(
            estimator_type="regressor",
            target_tags=TargetTags(required=True, one_d_labels=True, single_output=True),
            regressor_tags=RegressorTags(poor_score=False),
            input_tags=InputTags(two_d_array=True, allow_nan=False, sparse=False),
            requires_fit=True,
        )

    def save_model(self, path: str | Path) -> None:
        """Save the model in a versioned format without training data or device identifiers."""

        self._require_model().save(str(path))

    def load_model(self, path: str | Path) -> "MPSBoostRegressor":
        """Load and validate a model, replacing fitted state only after complete success."""

        if not self._fit_lock.acquire(blocking=False):
            raise RuntimeError("模型训练或加载正在进行")
        try:
            candidate = _native._load_regression_model(str(path))
            if candidate.objective != self._native_objective:
                raise ValueError(
                    f"model objective '{candidate.objective}' is incompatible with "
                    f"{type(self).__name__}"
                )
            self._validate_loaded_model(candidate)
            self.model_ = candidate
            self.n_features_in_ = candidate.feature_count
            self.device_ = self.device if self.device != "auto" else "cpu"
            self.n_estimators_ = candidate.tree_count
            self._finalize_fitted_metadata()
            self.training_summary_ = {"loaded": True, "device": self.device_}
            return self
        finally:
            self._fit_lock.release()

    def _require_model(self) -> Any:
        """Return the complete native model or raise a stable unfitted exception."""

        if not hasattr(self, "model_"):
            raise NotFittedError(self._fitted_error_message)
        return self.model_

    def _predict_raw(self, X: Any) -> NDArray[np.float32]:
        """Return native raw margins or regression values after feature-count validation."""

        model = self._require_model()
        matrix = as_dense_matrix(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("prediction feature count does not match training data")
        return np.asarray(model.predict(matrix), dtype=np.float32)

    def _training_labels(self, y: Any, n_samples: int) -> NDArray[np.float32]:
        """Normalize labels for the native training objective."""

        return as_labels(y, n_samples)

    def _finalize_fitted_metadata(self) -> None:
        """Attach estimator-specific fitted metadata after native model assignment."""

    def _validate_loaded_model(self, model: Any) -> None:
        """Validate estimator-specific metadata before accepting a loaded native model."""

    def _clear_fitted_state(self) -> None:
        """Delete every fitted field through one path to avoid stale partial state."""

        for name in (
            "model_",
            "n_features_in_",
            "device_",
            "device_decision_",
            "n_estimators_",
            "training_summary_",
            "classes_",
        ):
            self.__dict__.pop(name, None)

    def _validate_parameters(self) -> None:
        """Validate all public Python parameters before device initialization."""

        integer_ranges = {
            "n_estimators": (self.n_estimators, 1, 2**32 - 1),
            "max_depth": (self.max_depth, 0, 31),
            "max_bins": (self.max_bins, 2, 65536),
            "min_samples_leaf": (self.min_samples_leaf, 1, 2**63 - 1),
            "verbosity": (self.verbosity, 0, 3),
        }
        for name, (value, minimum, maximum) in integer_ranges.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} 必须是整数")
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} 必须位于 [{minimum}, {maximum}]")
        for name, numeric_value, lower, upper, lower_inclusive in (
            ("learning_rate", self.learning_rate, 0.0, 1.0, False),
            ("min_child_weight", self.min_child_weight, 0.0, np.inf, True),
            ("reg_lambda", self.reg_lambda, 0.0, np.inf, True),
        ):
            if isinstance(numeric_value, bool) or not isinstance(
                numeric_value, (int, float)
            ):
                raise TypeError(f"{name} 必须是实数")
            numeric = float(numeric_value)
            valid_lower = numeric >= lower if lower_inclusive else numeric > lower
            if not np.isfinite(numeric) or not valid_lower or numeric > upper:
                bracket = "[" if lower_inclusive else "("
                raise ValueError(f"{name} 必须位于 {bracket}{lower}, {upper}]")
        if self.device not in {"mps", "cpu", "auto"}:
            raise ValueError("device 只能是 'mps'、'cpu' 或 'auto'")
        if self.random_state is not None and (
            isinstance(self.random_state, bool) or not isinstance(self.random_state, int)
        ):
            raise TypeError("random_state 必须是整数或 None")


class DecisionTreeRegressor(MPSBoostRegressor):
    """Train one squared-error histogram tree through the shared native tree engine."""

    _fitted_error_message = "DecisionTreeRegressor is not fitted or loaded"
    _PARAMETER_NAMES = (
        "max_depth",
        "max_bins",
        "min_child_weight",
        "min_samples_leaf",
        "reg_lambda",
        "random_state",
        "device",
        "verbosity",
    )

    def __init__(
        self,
        max_depth: int = 6,
        max_bins: int = 256,
        min_child_weight: float = 1.0,
        min_samples_leaf: int = 20,
        reg_lambda: float = 1.0,
        random_state: int | None = None,
        device: str = "mps",
        verbosity: int = 1,
    ) -> None:
        """Store decision-tree parameters while fixing the native trainer to one tree."""

        super().__init__(
            n_estimators=1,
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

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any = None,
    ) -> "DecisionTreeRegressor":
        """Fit exactly one native tree even if private attributes were mutated."""

        self.n_estimators = 1
        self.learning_rate = 1.0
        return super().fit(X, y, sample_weight=sample_weight)

    def _validate_loaded_model(self, model: Any) -> None:
        """Reject boosted ensembles because this estimator promises one tree."""

        if model.tree_count != 1:
            raise ValueError("DecisionTreeRegressor can only load one-tree models")


class ExtraTreeRegressor(DecisionTreeRegressor):
    """Train one squared-error tree with native random-threshold split candidates."""

    _split_strategy = "random_threshold"
    _fitted_error_message = "ExtraTreeRegressor is not fitted or loaded"


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


class _CatBoostLikeMixin:
    """Shared controlled CatBoost-like public parameters and validation.

    The current implementation uses the real native histogram boosting backend while exposing the
    ordered-boosting contract that future categorical and ordered-statistics work will extend. The
    class deliberately rejects categorical features until native categorical split semantics exist,
    so users cannot accidentally depend on a silent preprocessing shortcut.
    """

    _CATBOOST_PARAMETER_NAMES = (
        "ordered_boosting",
        "permutation_count",
        "cat_features",
    )

    def _store_catboost_parameters(
        self,
        *,
        ordered_boosting: bool,
        permutation_count: int,
        cat_features: Any,
    ) -> None:
        """Store CatBoost-like parameters without allocating permutations or devices."""

        self.ordered_boosting = ordered_boosting
        self.permutation_count = permutation_count
        self.cat_features = cat_features

    def _validate_parameters(self) -> None:
        """Validate CatBoost-like parameters before delegating to the native trainer."""

        super()._validate_parameters()
        if not isinstance(self.ordered_boosting, bool):
            raise TypeError("ordered_boosting must be a boolean")
        if isinstance(self.permutation_count, bool) or not isinstance(
            self.permutation_count, int
        ):
            raise TypeError("permutation_count must be an integer")
        if self.permutation_count <= 0:
            raise ValueError("permutation_count must be positive")
        if self.cat_features is not None:
            raise NotImplementedError(
                "cat_features require native categorical split support and are not available yet"
            )

    def _catboost_training_metadata(self, n_samples: int) -> dict[str, Any]:
        """Build deterministic ordered-boosting metadata for diagnostics and reproducibility."""

        permutations = ordered_boosting_permutations(
            n_samples=n_samples,
            n_permutations=self.permutation_count,
            random_state=self.random_state,
        )
        return {
            "ordered_boosting": self.ordered_boosting,
            "permutation_count": self.permutation_count,
            "permutation_heads": [
                item[: min(8, n_samples)].tolist() for item in permutations
            ],
            "cat_features": None,
        }


class CatBoostRegressor(_CatBoostLikeMixin, MPSBoostRegressor):
    """Train a controlled CatBoost-like regressor on the shared native boosting backend."""

    _fitted_error_message = "CatBoostRegressor is not fitted or loaded"
    _PARAMETER_NAMES = (
        MPSBoostRegressor._PARAMETER_NAMES
        + _CatBoostLikeMixin._CATBOOST_PARAMETER_NAMES
    )

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        max_bins: int = 256,
        min_child_weight: float = 1.0,
        min_samples_leaf: int = 20,
        reg_lambda: float = 1.0,
        random_state: int | None = None,
        device: str = "mps",
        verbosity: int = 1,
        ordered_boosting: bool = True,
        permutation_count: int = 1,
        cat_features: Any = None,
    ) -> None:
        """Store CatBoost-like regressor parameters while preserving sklearn constructor rules."""

        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_bins=max_bins,
            min_child_weight=min_child_weight,
            min_samples_leaf=min_samples_leaf,
            reg_lambda=reg_lambda,
            random_state=random_state,
            device=device,
            verbosity=verbosity,
        )
        self._store_catboost_parameters(
            ordered_boosting=ordered_boosting,
            permutation_count=permutation_count,
            cat_features=cat_features,
        )

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any = None,
    ) -> "CatBoostRegressor":
        """Fit using the real native boosting path and attach ordered-boosting diagnostics."""

        n_samples = as_dense_matrix(X).shape[0]
        fitted = super().fit(X, y, sample_weight=sample_weight)
        fitted.training_summary_.update(self._catboost_training_metadata(n_samples))
        return fitted


class CatBoostClassifier(_CatBoostLikeMixin, MPSBoostClassifier):
    """Train a controlled CatBoost-like binary classifier on the native boosting backend."""

    _fitted_error_message = "CatBoostClassifier is not fitted or loaded"
    _PARAMETER_NAMES = (
        MPSBoostClassifier._PARAMETER_NAMES
        + _CatBoostLikeMixin._CATBOOST_PARAMETER_NAMES
    )

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        max_bins: int = 256,
        min_child_weight: float = 1.0,
        min_samples_leaf: int = 20,
        reg_lambda: float = 1.0,
        random_state: int | None = None,
        device: str = "mps",
        verbosity: int = 1,
        ordered_boosting: bool = True,
        permutation_count: int = 1,
        cat_features: Any = None,
    ) -> None:
        """Store CatBoost-like classifier parameters without training side effects."""

        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_bins=max_bins,
            min_child_weight=min_child_weight,
            min_samples_leaf=min_samples_leaf,
            reg_lambda=reg_lambda,
            random_state=random_state,
            device=device,
            verbosity=verbosity,
        )
        self._store_catboost_parameters(
            ordered_boosting=ordered_boosting,
            permutation_count=permutation_count,
            cat_features=cat_features,
        )

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any = None,
    ) -> "CatBoostClassifier":
        """Fit using the real native logistic boosting path and attach ordered diagnostics."""

        n_samples = as_dense_matrix(X).shape[0]
        fitted = super().fit(X, y, sample_weight=sample_weight)
        fitted.training_summary_.update(self._catboost_training_metadata(n_samples))
        return fitted


class DecisionTreeClassifier(MPSBoostClassifier):
    """Train one binary-logistic histogram tree through the shared native tree engine."""

    _fitted_error_message = "DecisionTreeClassifier is not fitted or loaded"
    _PARAMETER_NAMES = DecisionTreeRegressor._PARAMETER_NAMES

    def __init__(
        self,
        max_depth: int = 6,
        max_bins: int = 256,
        min_child_weight: float = 1.0,
        min_samples_leaf: int = 20,
        reg_lambda: float = 1.0,
        random_state: int | None = None,
        device: str = "mps",
        verbosity: int = 1,
    ) -> None:
        """Store classifier tree parameters while fixing the native trainer to one tree."""

        super().__init__(
            n_estimators=1,
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

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any = None,
    ) -> "DecisionTreeClassifier":
        """Fit exactly one native logistic tree even if private attributes were mutated."""

        self.n_estimators = 1
        self.learning_rate = 1.0
        return super().fit(X, y, sample_weight=sample_weight)

    def _validate_loaded_model(self, model: Any) -> None:
        """Reject boosted ensembles because this estimator promises one tree."""

        if model.tree_count != 1:
            raise ValueError("DecisionTreeClassifier can only load one-tree models")


class ExtraTreeClassifier(DecisionTreeClassifier):
    """Train one binary-logistic tree with native random-threshold split candidates."""

    _split_strategy = "random_threshold"
    _fitted_error_message = "ExtraTreeClassifier is not fitted or loaded"


class _ForestMixin:
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

    def save_model(self, path: str | Path) -> None:
        """Save a forest container while reusing the native format for every tree."""

        self._require_model()
        target = Path(path)
        directory = target.parent if target.parent != Path("") else Path(".")
        if not directory.is_dir():
            raise ValueError("model save directory does not exist")
        manifest = {
            "format": "mpsboost-forest",
            "version": 1,
            "estimator": type(self).__name__,
            "n_features_in": int(self.n_features_in_),
            "feature_subsets": [features.tolist() for features in self.feature_subsets_],
            "parameters": self.get_params(),
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.mpsboost-",
            suffix=".tmp",
            dir=directory,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with zipfile.ZipFile(temporary, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
                with tempfile.TemporaryDirectory(dir=directory) as tree_directory:
                    tree_root = Path(tree_directory)
                    for index, tree in enumerate(self.estimators_):
                        tree_path = tree_root / f"tree_{index}.mb"
                        tree.save_model(tree_path)
                        archive.write(tree_path, f"trees/tree_{index}.mb")
            os.replace(temporary, target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def load_model(self, path: str | Path) -> "_ForestMixin":
        """Load a forest container and validate every embedded native tree."""

        if not self._fit_lock.acquire(blocking=False):
            raise RuntimeError("model training or loading is already in progress")
        try:
            with zipfile.ZipFile(Path(path), mode="r") as archive:
                manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
                if manifest.get("format") != "mpsboost-forest" or manifest.get("version") != 1:
                    raise ValueError("unsupported random forest model format")
                if manifest.get("estimator") != type(self).__name__:
                    raise ValueError("random forest estimator type is incompatible")
                feature_subsets = tuple(
                    np.asarray(features, dtype=np.int64)
                    for features in manifest["feature_subsets"]
                )
                estimators: list[DecisionTreeRegressor | DecisionTreeClassifier] = []
                with tempfile.TemporaryDirectory() as tree_directory:
                    tree_root = Path(tree_directory)
                    for index, features in enumerate(feature_subsets):
                        tree_path = tree_root / f"tree_{index}.mb"
                        tree_path.write_bytes(archive.read(f"trees/tree_{index}.mb"))
                        tree = self._tree_type(device=self.device).load_model(tree_path)
                        if tree.n_features_in_ != len(features):
                            raise ValueError("forest tree feature subset does not match model")
                        estimators.append(tree)
            self.estimators_ = tuple(estimators)
            self.feature_subsets_ = feature_subsets
            self.n_features_in_ = int(manifest["n_features_in"])
            self.n_estimators_ = len(estimators)
            self.device_ = self.device if self.device != "auto" else "cpu"
            self.training_summary_ = {"loaded": True, "n_estimators": len(estimators)}
            self._finalize_fitted_metadata()
            return self
        except Exception:
            self._clear_fitted_state()
            raise
        finally:
            self._fit_lock.release()

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

    def _make_tree(self, random_state: int) -> DecisionTreeRegressor | DecisionTreeClassifier:
        """Create one native decision tree with the forest's tree parameters."""

        return self._tree_type(
            max_depth=self.max_depth,
            max_bins=self.max_bins,
            min_child_weight=self.min_child_weight,
            min_samples_leaf=self.min_samples_leaf,
            reg_lambda=self.reg_lambda,
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


class RandomForestRegressor(_ForestMixin, MPSBoostRegressor):
    """Train a random forest regressor from independent native decision trees."""

    _tree_type = DecisionTreeRegressor
    _fitted_error_message = "RandomForestRegressor is not fitted"

    def predict(self, X: Any) -> NDArray[np.float32]:
        """Return the mean prediction across fitted native decision trees."""

        self._require_model()
        matrix = as_dense_matrix(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("prediction feature count does not match training data")
        predictions = np.zeros(matrix.shape[0], dtype=np.float64)
        for tree, features in zip(self.estimators_, self.feature_subsets_):
            predictions += tree.predict(matrix[:, features]).astype(np.float64, copy=False)
        predictions /= float(len(self.estimators_))
        return predictions.astype(np.float32)


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
        matrix = as_dense_matrix(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("prediction feature count does not match training data")
        probabilities = np.zeros((matrix.shape[0], 2), dtype=np.float64)
        for tree, features in zip(self.estimators_, self.feature_subsets_):
            probabilities += tree.predict_proba(matrix[:, features]).astype(
                np.float64, copy=False
            )
        probabilities /= float(len(self.estimators_))
        return probabilities.astype(np.float32)


class ExtraTreesRegressor(RandomForestRegressor):
    """Train an ExtraTrees-style regressor from native random-threshold trees."""

    _tree_type = ExtraTreeRegressor
    _fitted_error_message = "ExtraTreesRegressor is not fitted"


class ExtraTreesClassifier(RandomForestClassifier):
    """Train an ExtraTrees-style binary classifier from native random-threshold trees."""

    _tree_type = ExtraTreeClassifier
    _fitted_error_message = "ExtraTreesClassifier is not fitted"
