"""Controlled CatBoost-like estimator entries for MPSBoost.

These classes expose ordered-boosting parameters while reusing the real native histogram boosting
backend. Categorical feature parameters fail early until native categorical split semantics exist.
"""

from __future__ import annotations

from typing import Any

from ..matrix import as_dense_matrix
from ..randomization import ordered_boosting_permutations
from .base import MPSBoostRegressor
from .classification import MPSBoostClassifier


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


