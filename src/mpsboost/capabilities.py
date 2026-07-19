"""Capability registry for public tree estimator names.

This module is the single public source for estimator availability. Planned estimator names are
documented here instead of being exported as incomplete classes, so users get clear discovery and
clear early failures without accidentally depending on placeholder implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .families import TreeFamilySpec, tree_family_spec

EstimatorStatus = Literal["available", "planned"]


@dataclass(frozen=True)
class EstimatorCapability:
    """Immutable public capability record for one estimator name."""

    name: str
    family_key: str
    status: EstimatorStatus
    primary: bool
    alias_for: str | None = None

    @property
    def family(self) -> TreeFamilySpec:
        """Return the shared tree-family semantic contract for this estimator."""

        return tree_family_spec(self.family_key)


_ESTIMATOR_CAPABILITIES: tuple[EstimatorCapability, ...] = (
    EstimatorCapability(
        name="GradientBoostingRegressor",
        family_key="histogram_gbdt_regression",
        status="available",
        primary=True,
    ),
    EstimatorCapability(
        name="MPSBoostRegressor",
        family_key="histogram_gbdt_regression",
        status="available",
        primary=False,
        alias_for="GradientBoostingRegressor",
    ),
    EstimatorCapability(
        name="GradientBoostingClassifier",
        family_key="histogram_gbdt_classification",
        status="available",
        primary=True,
    ),
    EstimatorCapability(
        name="MPSBoostClassifier",
        family_key="histogram_gbdt_classification",
        status="available",
        primary=False,
        alias_for="GradientBoostingClassifier",
    ),
    EstimatorCapability(
        name="RandomForestRegressor",
        family_key="random_forest_regression",
        status="available",
        primary=True,
    ),
    EstimatorCapability(
        name="RandomForestClassifier",
        family_key="random_forest_classification",
        status="available",
        primary=True,
    ),
    EstimatorCapability(
        name="ExtraTreesRegressor",
        family_key="extra_trees_regression",
        status="available",
        primary=True,
    ),
    EstimatorCapability(
        name="ExtraTreesClassifier",
        family_key="extra_trees_classification",
        status="available",
        primary=True,
    ),
    EstimatorCapability(
        name="ExtraTreeRegressor",
        family_key="extra_trees_regression",
        status="available",
        primary=False,
        alias_for="ExtraTreesRegressor",
    ),
    EstimatorCapability(
        name="ExtraTreeClassifier",
        family_key="extra_trees_classification",
        status="available",
        primary=False,
        alias_for="ExtraTreesClassifier",
    ),
    EstimatorCapability(
        name="DecisionTreeRegressor",
        family_key="decision_tree_regression",
        status="available",
        primary=True,
    ),
    EstimatorCapability(
        name="DecisionTreeClassifier",
        family_key="decision_tree_classification",
        status="available",
        primary=True,
    ),
    EstimatorCapability(
        name="CatBoostRegressor",
        family_key="catboost_regression",
        status="planned",
        primary=True,
    ),
    EstimatorCapability(
        name="CatBoostClassifier",
        family_key="catboost_classification",
        status="planned",
        primary=True,
    ),
    EstimatorCapability(
        name="IsolationForest",
        family_key="isolation_forest",
        status="planned",
        primary=True,
    ),
    EstimatorCapability(
        name="LearningToRankRegressor",
        family_key="learning_to_rank",
        status="planned",
        primary=True,
    ),
)

_CAPABILITY_BY_NAME = {
    capability.name: capability for capability in _ESTIMATOR_CAPABILITIES
}


def estimator_capabilities() -> tuple[EstimatorCapability, ...]:
    """Return all known estimator capability records in stable documentation order."""

    return _ESTIMATOR_CAPABILITIES


def estimator_status(name: str) -> EstimatorStatus:
    """Return whether an estimator name is currently available or planned.

    Unknown names fail early because silently treating a typo as a planned estimator would hide
    user mistakes and make migration scripts unsafe.
    """

    try:
        return _CAPABILITY_BY_NAME[name].status
    except KeyError as exc:
        known = ", ".join(sorted(_CAPABILITY_BY_NAME))
        raise ValueError(f"Unknown estimator '{name}'. Known estimators: {known}") from exc


def estimator_capability(name: str) -> EstimatorCapability:
    """Return one estimator capability record or fail early for unknown public names."""

    try:
        return _CAPABILITY_BY_NAME[name]
    except KeyError as exc:
        known = ", ".join(sorted(_CAPABILITY_BY_NAME))
        raise ValueError(f"Unknown estimator '{name}'. Known estimators: {known}") from exc


def available_estimators() -> tuple[str, ...]:
    """Return public estimator names that are implemented and safe to instantiate."""

    return tuple(
        capability.name
        for capability in _ESTIMATOR_CAPABILITIES
        if capability.status == "available"
    )


def planned_estimators() -> tuple[str, ...]:
    """Return planned estimator names that are documented but not yet exported as classes."""

    return tuple(
        capability.name
        for capability in _ESTIMATOR_CAPABILITIES
        if capability.status == "planned"
    )


def require_estimator_supported(name: str) -> None:
    """Fail early with a clear message when an estimator is known but not implemented."""

    status = estimator_status(name)
    if status == "available":
        return
    raise NotImplementedError(
        f"{name} is planned for MPSBoost v2 but is not implemented yet. "
        "Use GradientBoostingRegressor or GradientBoostingClassifier in this release."
    )
