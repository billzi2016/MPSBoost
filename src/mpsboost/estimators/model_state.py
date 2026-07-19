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
from ..matrix import as_dense_matrix, as_labels
from .errors import NotFittedError



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
        for optional_name in ("max_leaves", "max_active_leaves"):
            optional_value = getattr(self, optional_name)
            if optional_value is None:
                continue
            integer_ranges[optional_name] = (optional_value, 2, 2**32 - 1)
        for name, (value, minimum, maximum) in integer_ranges.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} 必须是整数")
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} 必须位于 [{minimum}, {maximum}]")
        for name, numeric_value, lower, upper, lower_inclusive in (
            ("learning_rate", self.learning_rate, 0.0, 1.0, False),
            ("min_child_weight", self.min_child_weight, 0.0, np.inf, True),
            ("min_gain_to_split", self.min_gain_to_split, 0.0, np.inf, True),
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
        if self.growth_strategy not in {"level_wise", "leaf_wise"}:
            raise ValueError("growth_strategy must be 'level_wise' or 'leaf_wise'")
        if (
            self.max_active_leaves is not None
            and self.max_leaves is not None
            and self.max_active_leaves > self.max_leaves
        ):
            raise ValueError("max_active_leaves must not exceed max_leaves")
        if self.device not in {"mps", "cpu", "auto"}:
            raise ValueError("device 只能是 'mps'、'cpu' 或 'auto'")
        if self.random_state is not None and (
            isinstance(self.random_state, bool) or not isinstance(self.random_state, int)
        ):
            raise TypeError("random_state 必须是整数或 None")
