"""Shared tree-family specifications for MPSBoost v2.

The objects in this module describe product semantics only: task type, objective family,
sampling strategy, tree growth strategy, and prediction aggregation. They intentionally do not
train trees, allocate Metal buffers, or duplicate split-gain math. Runtime implementations must
depend on these immutable specs instead of inventing per-estimator behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TaskType = Literal["regression", "classification", "anomaly_detection", "ranking"]
ObjectiveName = Literal[
    "squared_error",
    "logistic",
    "random_split",
    "ordered_boosting",
    "path_length",
    "ranking",
]
SamplingStrategy = Literal["none", "bootstrap", "feature_subsample", "random_threshold"]
GrowthStrategy = Literal["level_wise", "leaf_wise", "independent_trees", "ordered_boosting"]
PredictionAggregation = Literal["sum", "mean", "vote", "path_length", "rank_score"]


@dataclass(frozen=True)
class TreeFamilySpec:
    """Immutable semantic contract shared by one or more public estimator names."""

    key: str
    task: TaskType
    objective: ObjectiveName
    sampling: tuple[SamplingStrategy, ...]
    growth: GrowthStrategy
    aggregation: PredictionAggregation
    supports_mps_training: bool
    supports_mps_prediction: bool


_TREE_FAMILY_SPECS: tuple[TreeFamilySpec, ...] = (
    TreeFamilySpec(
        key="histogram_gbdt_regression",
        task="regression",
        objective="squared_error",
        sampling=("none",),
        growth="level_wise",
        aggregation="sum",
        supports_mps_training=True,
        supports_mps_prediction=False,
    ),
    TreeFamilySpec(
        key="histogram_gbdt_classification",
        task="classification",
        objective="logistic",
        sampling=("none",),
        growth="level_wise",
        aggregation="sum",
        supports_mps_training=True,
        supports_mps_prediction=False,
    ),
    TreeFamilySpec(
        key="random_forest_regression",
        task="regression",
        objective="squared_error",
        sampling=("bootstrap", "feature_subsample"),
        growth="independent_trees",
        aggregation="mean",
        supports_mps_training=False,
        supports_mps_prediction=False,
    ),
    TreeFamilySpec(
        key="random_forest_classification",
        task="classification",
        objective="logistic",
        sampling=("bootstrap", "feature_subsample"),
        growth="independent_trees",
        aggregation="vote",
        supports_mps_training=False,
        supports_mps_prediction=False,
    ),
    TreeFamilySpec(
        key="extra_trees_regression",
        task="regression",
        objective="random_split",
        sampling=("feature_subsample", "random_threshold"),
        growth="independent_trees",
        aggregation="mean",
        supports_mps_training=False,
        supports_mps_prediction=False,
    ),
    TreeFamilySpec(
        key="extra_trees_classification",
        task="classification",
        objective="random_split",
        sampling=("feature_subsample", "random_threshold"),
        growth="independent_trees",
        aggregation="vote",
        supports_mps_training=False,
        supports_mps_prediction=False,
    ),
    TreeFamilySpec(
        key="decision_tree_regression",
        task="regression",
        objective="squared_error",
        sampling=("none",),
        growth="level_wise",
        aggregation="sum",
        supports_mps_training=True,
        supports_mps_prediction=False,
    ),
    TreeFamilySpec(
        key="decision_tree_classification",
        task="classification",
        objective="logistic",
        sampling=("none",),
        growth="level_wise",
        aggregation="vote",
        supports_mps_training=True,
        supports_mps_prediction=False,
    ),
    TreeFamilySpec(
        key="catboost_regression",
        task="regression",
        objective="ordered_boosting",
        sampling=("none",),
        growth="ordered_boosting",
        aggregation="sum",
        supports_mps_training=False,
        supports_mps_prediction=False,
    ),
    TreeFamilySpec(
        key="catboost_classification",
        task="classification",
        objective="ordered_boosting",
        sampling=("none",),
        growth="ordered_boosting",
        aggregation="sum",
        supports_mps_training=False,
        supports_mps_prediction=False,
    ),
    TreeFamilySpec(
        key="isolation_forest",
        task="anomaly_detection",
        objective="path_length",
        sampling=("feature_subsample", "random_threshold"),
        growth="independent_trees",
        aggregation="path_length",
        supports_mps_training=False,
        supports_mps_prediction=False,
    ),
    TreeFamilySpec(
        key="learning_to_rank",
        task="ranking",
        objective="ranking",
        sampling=("none",),
        growth="level_wise",
        aggregation="rank_score",
        supports_mps_training=False,
        supports_mps_prediction=False,
    ),
)

_SPEC_BY_KEY = {spec.key: spec for spec in _TREE_FAMILY_SPECS}


def tree_family_specs() -> tuple[TreeFamilySpec, ...]:
    """Return every tree-family semantic contract in stable planning order."""

    return _TREE_FAMILY_SPECS


def tree_family_spec(key: str) -> TreeFamilySpec:
    """Return one family spec or fail early for invalid internal wiring."""

    try:
        return _SPEC_BY_KEY[key]
    except KeyError as exc:
        known = ", ".join(sorted(_SPEC_BY_KEY))
        raise ValueError(f"Unknown tree family '{key}'. Known families: {known}") from exc


def mps_training_families() -> tuple[str, ...]:
    """Return tree-family keys with implemented MPS training support."""

    return tuple(spec.key for spec in _TREE_FAMILY_SPECS if spec.supports_mps_training)
