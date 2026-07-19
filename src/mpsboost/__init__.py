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
from .estimator import MPSBoostRegressor

# Keep one implementation and expose a shorter sklearn-style public name. This is an alias,
# not a wrapper, so model behavior, serialization, and type checks remain identical.
GradientBoostingRegressor = MPSBoostRegressor

# Maintain __all__ explicitly so ``from mpsboost import *`` does not leak helper modules.
__all__ = [
    "EstimatorCapability",
    "GradientBoostingRegressor",
    "MPSBoostRegressor",
    "__version__",
    "available_estimators",
    "cache_info",
    "clear_cache",
    "create_cache",
    "estimator_capabilities",
    "estimator_status",
    "is_available",
    "planned_estimators",
    "require_estimator_supported",
    "system_info",
]
