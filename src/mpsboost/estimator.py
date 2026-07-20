"""Backward-compatible estimator re-exports for MPSBoost.

Estimator implementations live in ``mpsboost.estimators`` so each model family has a focused module.
This file preserves the historical ``mpsboost.estimator`` import path without adding behavior.
"""

from .estimators import (
    CatBoostClassifier,
    CatBoostRegressor,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    IsolationForest,
    LearningToRankRegressor,
    MPSBoostClassifier,
    MPSIsolationForest,
    MPSBoostRegressor,
    NotFittedError,
    RandomForestClassifier,
    RandomForestRegressor,
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
    "IsolationForest",
    "LearningToRankRegressor",
    "MPSBoostClassifier",
    "MPSIsolationForest",
    "MPSBoostRegressor",
    "NotFittedError",
    "RandomForestClassifier",
    "RandomForestRegressor",
]
