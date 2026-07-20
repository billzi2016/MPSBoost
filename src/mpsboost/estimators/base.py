"""Base gradient-boosting regressor for MPSBoost estimators.

This module owns shared parameter storage, input adaptation, fitted-state replacement, prediction,
and weighted regression scoring. Optional tooling and persistence live in mixins.
"""

from __future__ import annotations

from threading import Lock
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .. import _native
from ..categorical import fit_transform_categorical
from ..device_policy import choose_device, decision_to_dict
from ..diagnostics import _metallib_path, is_available
from ..matrix import as_labels, as_sample_weight
from .importance import FeatureImportanceMixin
from .model_state import SklearnAndPersistenceMixin


class MPSBoostRegressor(FeatureImportanceMixin, SklearnAndPersistenceMixin):
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
        "growth_strategy",
        "max_leaves",
        "max_active_leaves",
        "min_gain_to_split",
        "min_child_weight",
        "min_samples_leaf",
        "reg_lambda",
        "reg_alpha",
        "max_delta_step",
        "monotonic_constraints",
        "interaction_constraints",
        "categorical_features",
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
        growth_strategy: str = "level_wise",
        max_leaves: int | None = None,
        max_active_leaves: int | None = None,
        min_gain_to_split: float = 0.0,
        min_child_weight: float = 1.0,
        min_samples_leaf: int = 20,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        max_delta_step: float = 0.0,
        monotonic_constraints: Any = None,
        interaction_constraints: Any = None,
        categorical_features: Any = None,
        random_state: int | None = None,
        device: str = "mps",
        verbosity: int = 1,
    ) -> None:
        """Store estimator parameters without expensive side effects."""

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_bins = max_bins
        self.growth_strategy = growth_strategy
        self.max_leaves = max_leaves
        self.max_active_leaves = max_active_leaves
        self.min_gain_to_split = min_gain_to_split
        self.min_child_weight = min_child_weight
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.max_delta_step = max_delta_step
        self.monotonic_constraints = monotonic_constraints
        self.interaction_constraints = interaction_constraints
        self.categorical_features = categorical_features
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
            raw_rows = np.asarray(X).shape[0]
            labels = self._training_labels(y, raw_rows)
            weights = as_sample_weight(sample_weight, raw_rows)
            matrix, categorical_metadata = fit_transform_categorical(
                X, self.categorical_features, labels, weights
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
                objective=self._native_objective,
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
            self.categorical_metadata_ = categorical_metadata
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
                "categorical_features": (
                    []
                    if categorical_metadata is None
                    else list(categorical_metadata.features)
                ),
                "monotonic_constraints": self._normalized_monotonic_constraints(
                    matrix.shape[1]
                ),
                "interaction_constraints": self._normalized_interaction_constraints(
                    matrix.shape[1]
                ),
                "growth_strategy": self.growth_strategy,
                "max_leaves": self.max_leaves,
                "max_active_leaves": self.max_active_leaves,
                "min_gain_to_split": float(self.min_gain_to_split),
                "reg_alpha": float(self.reg_alpha),
                "max_delta_step": float(self.max_delta_step),
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
