"""Stable public entry points for MPSBoost.

This module only collects stable public symbols and intentionally avoids training logic.
That separation keeps two invariants simple:

1. users can always import the package through ``import mpsboost as mb``;
2. internal modules can be refactored without leaking implementation details as public API.

The concise estimator names follow the familiar scikit-learn style, while the
``MPSBoost*`` names remain available as explicit project-branded aliases.
"""

from ._native import __version__  # type: ignore[import-not-found]
from .capabilities import (
    EstimatorCapability,
    available_estimators,
    estimator_capability,
    estimator_capabilities,
    estimator_status,
    planned_estimators,
    require_estimator_supported,
)
from .diagnostics import (
    cache_info,
    clear_cache,
    create_cache,
    is_available,
    system_info,
)
from .device_policy import DeviceDecision, choose_device
from .estimator import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    MPSBoostClassifier,
    MPSBoostRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from .families import (
    TreeFamilySpec,
    mps_training_families,
    tree_family_spec,
    tree_family_specs,
)
from .monitoring import (
    EarlyStoppingDecision,
    EarlyStoppingMonitor,
    MetricHistory,
    MetricObservation,
)
from .randomization import (
    bootstrap_sample_indices,
    ordered_boosting_permutations,
    random_threshold_candidates,
    sample_without_replacement_indices,
    subsample_feature_indices,
    validate_indices_cover_range,
)

# Keep one implementation and expose a shorter sklearn-style public name. This is an alias,
# not a wrapper, so model behavior, serialization, and type checks remain identical.
GradientBoostingRegressor = MPSBoostRegressor
GradientBoostingClassifier = MPSBoostClassifier

# Maintain __all__ explicitly so ``from mpsboost import *`` does not leak helper modules.
__all__ = [
    "DeviceDecision",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "EarlyStoppingDecision",
    "EarlyStoppingMonitor",
    "EstimatorCapability",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "MPSBoostClassifier",
    "MPSBoostRegressor",
    "MetricHistory",
    "MetricObservation",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "__version__",
    "available_estimators",
    "bootstrap_sample_indices",
    "cache_info",
    "clear_cache",
    "choose_device",
    "create_cache",
    "estimator_capability",
    "estimator_capabilities",
    "estimator_status",
    "is_available",
    "mps_training_families",
    "ordered_boosting_permutations",
    "planned_estimators",
    "random_threshold_candidates",
    "require_estimator_supported",
    "sample_without_replacement_indices",
    "system_info",
    "subsample_feature_indices",
    "tree_family_spec",
    "tree_family_specs",
    "TreeFamilySpec",
    "validate_indices_cover_range",
]
