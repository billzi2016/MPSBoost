"""Capability registry for public tree estimator names.

This module is the single public source for estimator availability. Planned estimator names are
documented here instead of being exported as incomplete classes, so users get clear discovery and
clear early failures without accidentally depending on placeholder implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EstimatorStatus = Literal["available", "planned"]


@dataclass(frozen=True)
class EstimatorCapability:
    """Immutable public capability record for one estimator name."""

    name: str
    family: str
    status: EstimatorStatus
    primary: bool
    alias_for: str | None = None


_ESTIMATOR_CAPABILITIES: tuple[EstimatorCapability, ...] = (
    EstimatorCapability(
        name="GradientBoostingRegressor",
        family="histogram gradient boosting",
        status="available",
        primary=True,
    ),
    EstimatorCapability(
        name="MPSBoostRegressor",
        family="histogram gradient boosting",
        status="available",
        primary=False,
        alias_for="GradientBoostingRegressor",
    ),
    EstimatorCapability(
        name="GradientBoostingClassifier",
        family="histogram gradient boosting classification",
        status="planned",
        primary=True,
    ),
    EstimatorCapability(
        name="RandomForestRegressor",
        family="random forest",
        status="planned",
        primary=True,
    ),
    EstimatorCapability(
        name="RandomForestClassifier",
        family="random forest",
        status="planned",
        primary=True,
    ),
    EstimatorCapability(
        name="ExtraTreesRegressor",
        family="extra trees",
        status="planned",
        primary=True,
    ),
    EstimatorCapability(
        name="ExtraTreesClassifier",
        family="extra trees",
        status="planned",
        primary=True,
    ),
    EstimatorCapability(
        name="DecisionTreeRegressor",
        family="single decision tree",
        status="planned",
        primary=True,
    ),
    EstimatorCapability(
        name="DecisionTreeClassifier",
        family="single decision tree",
        status="planned",
        primary=True,
    ),
    EstimatorCapability(
        name="IsolationForest",
        family="isolation forest",
        status="planned",
        primary=True,
    ),
    EstimatorCapability(
        name="LearningToRankRegressor",
        family="ranking trees",
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
        "Use GradientBoostingRegressor for the current 0.2.x release."
    )
