"""Public estimator family exports for MPSBoost.

The package keeps estimator families split by responsibility while preserving the stable symbols
re-exported by ``mpsboost.estimator`` and the package root.
"""

from .base import MPSBoostRegressor
from .catboost_like import CatBoostClassifier, CatBoostRegressor
from .classification import MPSBoostClassifier
from .errors import NotFittedError
from .forest_public import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from .single_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)

__all__ = [
    "CatBoostClassifier",
    "CatBoostRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "ExtraTreeClassifier",
    "ExtraTreeRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "MPSBoostClassifier",
    "MPSBoostRegressor",
    "NotFittedError",
    "RandomForestClassifier",
    "RandomForestRegressor",
]
