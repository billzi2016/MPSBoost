"""Controlled CatBoost-like estimator entries for MPSBoost.

These classes expose ordered-boosting parameters while reusing the real native histogram boosting
backend. CatBoost-style ``cat_features`` is routed into the shared categorical encoder.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..randomization import ordered_boosting_permutations
from .base import MPSBoostRegressor
from .classification import MPSBoostClassifier


class _CatBoostLikeMixin:
    """Shared controlled CatBoost-like public parameters and validation.

    The current implementation uses the real native histogram boosting backend while exposing the
    ordered-boosting contract that future ordered-statistics work will extend. Categorical
    features use the same ordered native split adapter as the other estimator families.
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
        if cat_features is not None:
            self.categorical_features = cat_features

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
            "cat_features": self.cat_features,
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
        n_jobs: int | None = None,
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
            growth_strategy=growth_strategy,
            max_leaves=max_leaves,
            max_active_leaves=max_active_leaves,
            min_gain_to_split=min_gain_to_split,
            min_child_weight=min_child_weight,
            min_samples_leaf=min_samples_leaf,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            max_delta_step=max_delta_step,
            monotonic_constraints=monotonic_constraints,
            interaction_constraints=interaction_constraints,
            categorical_features=(
                categorical_features if categorical_features is not None else cat_features
            ),
            random_state=random_state,
            n_jobs=n_jobs,
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

        n_samples = np.asarray(X).shape[0]
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
        n_jobs: int | None = None,
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
            growth_strategy=growth_strategy,
            max_leaves=max_leaves,
            max_active_leaves=max_active_leaves,
            min_gain_to_split=min_gain_to_split,
            min_child_weight=min_child_weight,
            min_samples_leaf=min_samples_leaf,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            max_delta_step=max_delta_step,
            monotonic_constraints=monotonic_constraints,
            interaction_constraints=interaction_constraints,
            categorical_features=(
                categorical_features if categorical_features is not None else cat_features
            ),
            random_state=random_state,
            n_jobs=n_jobs,
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

        n_samples = np.asarray(X).shape[0]
        fitted = super().fit(X, y, sample_weight=sample_weight)
        fitted.training_summary_.update(self._catboost_training_metadata(n_samples))
        return fitted
