"""Shared imports for estimator unit tests."""

import numpy as np
import pytest

from mpsboost import (
    CatBoostClassifier,
    CatBoostRegressor,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    MPSBoostClassifier,
    MPSBoostRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from mpsboost.estimator import NotFittedError


def test_decision_tree_regressor_trains_exactly_one_native_tree():
    """DecisionTreeRegressor should expose a one-tree estimator without duplicating training."""

    X = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32)
    y = np.array([0.0, 0.0, 4.0, 4.0], dtype=np.float32)
    model = DecisionTreeRegressor(
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        device="cpu",
    ).fit(X, y)

    assert model.n_estimators_ == 1
    assert model.get_params()["max_depth"] == 1
    assert "n_estimators" not in model.get_params()
    assert model.feature_importances_.tolist() == [1.0]
    assert model.score(X, y) > 0.0


def test_decision_tree_classifier_trains_exactly_one_native_tree():
    """DecisionTreeClassifier should reuse the binary-logistic one-tree objective."""

    X = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    model = DecisionTreeClassifier(
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        device="cpu",
    ).fit(X, y)

    assert model.n_estimators_ == 1
    assert model.predict_proba(X).shape == (4, 2)
    assert model.predict(X).tolist() == y.tolist()
    assert model.score(X, y) == 1.0


def test_extra_tree_estimators_use_random_threshold_strategy():
    """Single ExtraTree estimators should be seeded native random-threshold trees."""

    X = np.array([[float(value)] for value in range(8)], dtype=np.float32)
    y_reg = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0], dtype=np.float32)
    first = ExtraTreeRegressor(
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        reg_lambda=0.0,
        random_state=101,
        device="cpu",
    ).fit(X, y_reg)
    second = ExtraTreeRegressor(
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        reg_lambda=0.0,
        random_state=101,
        device="cpu",
    ).fit(X, y_reg)
    different = ExtraTreeRegressor(
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        reg_lambda=0.0,
        random_state=202,
        device="cpu",
    ).fit(X, y_reg)

    assert first.model_.trees[0].nodes == second.model_.trees[0].nodes
    assert (
        first.model_.trees[0].nodes[0]["threshold_bin"]
        != different.model_.trees[0].nodes[0]["threshold_bin"]
    )

    y_clf = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    classifier = ExtraTreeClassifier(
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        random_state=101,
        device="cpu",
    ).fit(X, y_clf)
    assert classifier.predict_proba(X).shape == (8, 2)
