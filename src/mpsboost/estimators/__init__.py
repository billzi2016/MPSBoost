"""Public estimator family exports for MPSBoost.

The package keeps estimator families split by responsibility while preserving the stable symbols
re-exported by ``mpsboost.estimator`` and the package root.
"""

from .base import MPSBoostRegressor
from .catboost_like import CatBoostClassifier, CatBoostRegressor
from .anomaly import MPSIsolationForest
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
from .multiclass import MPSBoostClassifier
from .ranking import LearningToRankRegressor

IsolationForest = MPSIsolationForest

__all__ = [
    "CatBoostClassifier",
    "CatBoostRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "ExtraTreeClassifier",
    "ExtraTreeRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "IsolationForest",
    "LearningToRankRegressor",
    "MPSBoostClassifier",
    "MPSIsolationForest",
    "MPSBoostRegressor",
    "NotFittedError",
    "RandomForestClassifier",
    "RandomForestRegressor",
]
