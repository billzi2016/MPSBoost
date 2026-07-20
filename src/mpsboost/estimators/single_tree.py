"""Single-tree estimator classes for MPSBoost.

Decision-tree and ExtraTree estimators are thin public wrappers around the shared native boosting
base class. They fix the ensemble to one tree and must not duplicate split, objective, or model I/O
logic.
"""

from __future__ import annotations

from typing import Any

from .base import MPSBoostRegressor
from .classification import MPSBoostClassifier


class DecisionTreeRegressor(MPSBoostRegressor):
    """Train one squared-error histogram tree through the shared native tree engine."""

    _fitted_error_message = "DecisionTreeRegressor is not fitted or loaded"
    _PARAMETER_NAMES = (
        "max_depth",
        "max_bins",
        "min_child_weight",
        "min_samples_leaf",
        "reg_lambda",
        "monotonic_constraints",
        "interaction_constraints",
        "categorical_features",
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
        monotonic_constraints: Any = None,
        interaction_constraints: Any = None,
        categorical_features: Any = None,
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
            monotonic_constraints=monotonic_constraints,
            interaction_constraints=interaction_constraints,
            categorical_features=categorical_features,
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
        monotonic_constraints: Any = None,
        interaction_constraints: Any = None,
        categorical_features: Any = None,
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
            monotonic_constraints=monotonic_constraints,
            interaction_constraints=interaction_constraints,
            categorical_features=categorical_features,
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
