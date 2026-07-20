"""sklearn compatibility, model persistence, and fitted-state helpers.

This module assumes concrete estimators provide native objective names, fitted metadata, and a fit
lock. It keeps optional sklearn imports and model I/O outside the training implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .. import _native
from ..matrix import as_labels
from ..categorical import transform_categorical
from .errors import NotFittedError
from .prediction import predict_native_model



class SklearnAndPersistenceMixin:
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

        if getattr(self, "categorical_metadata_", None) is not None:
            raise NotImplementedError(
                "categorical model persistence requires categorical metadata support"
            )
        self._require_model().save(str(path))

    def load_model(self, path: str | Path) -> "MPSBoostRegressor":
        """Load and validate a model, replacing fitted state only after complete success."""

        if not self._fit_lock.acquire(blocking=False):
            raise RuntimeError("Model training or loading is already in progress")
        try:
            candidate = _native._load_regression_model(str(path))
            if candidate.objective != self._resolved_native_objective():
                raise ValueError(
                    f"model objective '{candidate.objective}' is incompatible with "
                    f"{type(self).__name__}"
                )
            if candidate.objective == "quantile" and not np.isclose(
                candidate.objective_alpha, float(self.quantile_alpha)
            ):
                raise ValueError("model quantile alpha is incompatible with estimator")
            if candidate.objective == "tweedie" and not np.isclose(
                candidate.tweedie_variance_power, float(self.tweedie_variance_power)
            ):
                raise ValueError("model tweedie variance power is incompatible with estimator")
            self._validate_loaded_model(candidate)
            self.model_ = candidate
            self.n_features_in_ = candidate.feature_count
            self.device_ = self.device if self.device != "auto" else "cpu"
            self.n_estimators_ = candidate.tree_count
            self._resolved_objective_ = candidate.objective
            self._finalize_fitted_metadata()
            self.training_summary_ = {
                "loaded": True,
                "device": self.device_,
                "native_objective": candidate.objective,
                "loss": getattr(self, "loss", "squared_error"),
                "quantile_alpha": float(getattr(self, "quantile_alpha", 0.5)),
                "tweedie_variance_power": float(
                    getattr(self, "tweedie_variance_power", 1.5)
                ),
            }
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
        matrix = transform_categorical(X, getattr(self, "categorical_metadata_", None))
        return predict_native_model(model, matrix, self.n_features_in_)

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
            "categorical_metadata_",
            "estimators_",
            "_multiclass_strategy_",
            "_resolved_objective_",
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
        for optional_name in ("max_leaves", "max_active_leaves"):
            optional_value = getattr(self, optional_name)
            if optional_value is None:
                continue
            integer_ranges[optional_name] = (optional_value, 2, 2**32 - 1)
        if self.n_jobs is not None:
            integer_ranges["n_jobs"] = (self.n_jobs, -1, 2**31 - 1)
        for name, (value, minimum, maximum) in integer_ranges.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
        if self.n_jobs == 0:
            raise ValueError("n_jobs must be None, -1, or a non-zero integer")
        for name, numeric_value, lower, upper, lower_inclusive in (
            ("learning_rate", self.learning_rate, 0.0, 1.0, False),
            ("min_child_weight", self.min_child_weight, 0.0, np.inf, True),
            ("min_gain_to_split", self.min_gain_to_split, 0.0, np.inf, True),
            ("reg_lambda", self.reg_lambda, 0.0, np.inf, True),
            ("reg_alpha", self.reg_alpha, 0.0, np.inf, True),
            ("max_delta_step", self.max_delta_step, 0.0, np.inf, True),
        ):
            if isinstance(numeric_value, bool) or not isinstance(
                numeric_value, (int, float)
            ):
                raise TypeError(f"{name} must be a real number")
            numeric = float(numeric_value)
            valid_lower = numeric >= lower if lower_inclusive else numeric > lower
            if not np.isfinite(numeric) or not valid_lower or numeric > upper:
                bracket = "[" if lower_inclusive else "("
                raise ValueError(f"{name} must be in {bracket}{lower}, {upper}]")
        if self.growth_strategy not in {"level_wise", "leaf_wise"}:
            raise ValueError("growth_strategy must be 'level_wise' or 'leaf_wise'")
        if (
            self.max_active_leaves is not None
            and self.max_leaves is not None
            and self.max_active_leaves > self.max_leaves
        ):
            raise ValueError("max_active_leaves must not exceed max_leaves")
        if self.device not in {"mps", "cpu", "auto"}:
            raise ValueError("device must be 'mps', 'cpu', or 'auto'")
        if self.random_state is not None and (
            isinstance(self.random_state, bool) or not isinstance(self.random_state, int)
        ):
            raise TypeError("random_state must be an integer or None")
        self._validate_objective_parameters()

    def _resolved_native_objective(self) -> str:
        """Return the native objective selected by this estimator configuration."""

        if self._native_objective != "squared_error":
            return self._native_objective
        return self.loss

    def _validate_objective_parameters(self) -> None:
        """Validate advanced regression objective controls before native allocation."""

        if self._native_objective == "squared_error":
            if self.loss not in {"squared_error", "quantile", "poisson", "tweedie"}:
                raise ValueError(
                    "loss must be 'squared_error', 'quantile', 'poisson', or 'tweedie'"
                )
        elif getattr(self, "loss", "squared_error") != "squared_error":
            raise ValueError("loss is only supported by regression estimators")
        for name, lower, upper in (
            ("quantile_alpha", 0.0, 1.0),
            ("tweedie_variance_power", 1.0, 2.0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            numeric = float(value)
            if not np.isfinite(numeric) or not lower < numeric < upper:
                raise ValueError(f"{name} must be in ({lower}, {upper})")

    def _normalized_monotonic_constraints(self, n_features: int) -> list[int]:
        """Return validated monotonic constraints for native split and leaf checks."""

        constraints = getattr(self, "monotonic_constraints", None)
        if constraints is None:
            return []
        normalized = list(constraints)
        if len(normalized) != n_features:
            raise ValueError("monotonic_constraints length must match feature count")
        result: list[int] = []
        for value in normalized:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError("monotonic_constraints values must be integers")
            integer = int(value)
            if integer not in {-1, 0, 1}:
                raise ValueError("monotonic_constraints values must be -1, 0, or 1")
            result.append(integer)
        return result

    def _normalized_interaction_constraints(self, n_features: int) -> list[list[int]]:
        """Return validated feature groups for native path interaction checks."""

        constraints = getattr(self, "interaction_constraints", None)
        if constraints is None:
            return []
        groups: list[list[int]] = []
        for group in constraints:
            normalized_group: list[int] = []
            for value in group:
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                    raise TypeError("interaction_constraints values must be integers")
                index = int(value)
                if index < 0:
                    index += n_features
                if not 0 <= index < n_features:
                    raise ValueError("interaction constraint feature index is out of range")
                normalized_group.append(index)
            if not normalized_group:
                raise ValueError("interaction constraint groups must be non-empty")
            if len(set(normalized_group)) != len(normalized_group):
                raise ValueError("interaction constraint groups must not contain duplicates")
            groups.append(sorted(normalized_group))
        return groups
