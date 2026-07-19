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


def test_catboost_like_regressor_uses_native_boosting_with_ordered_metadata():
    """CatBoostRegressor should be a real estimator with deterministic ordered metadata."""

    X = np.array([[0.0], [0.1], [1.0], [1.1], [2.0], [2.1]], dtype=np.float32)
    y = np.array([0.0, 0.0, 4.0, 4.0, 8.0, 8.0], dtype=np.float32)
    model = CatBoostRegressor(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        random_state=13,
        permutation_count=2,
        device="cpu",
    ).fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == (6,)
    assert model.score(X, y) > 0.0
    assert model.training_summary_["ordered_boosting"] is True
    assert model.training_summary_["permutation_count"] == 2
    assert len(model.training_summary_["permutation_heads"]) == 2


def test_catboost_like_classifier_uses_native_logistic_boosting():
    """CatBoostClassifier should expose classifier methods through the real native backend."""

    X = np.array([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]], dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    model = CatBoostClassifier(
        n_estimators=3,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        random_state=17,
        permutation_count=2,
        device="cpu",
    ).fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (6, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert model.predict(X).tolist() == y.tolist()
    assert model.score(X, y) == 1.0
    assert model.training_summary_["ordered_boosting"] is True


def test_catboost_like_estimators_reject_unsupported_categorical_features():
    """Categorical feature parameters must fail until native categorical splits exist."""

    X = np.ones((4, 2), dtype=np.float32)
    y_reg = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    with pytest.raises(NotImplementedError, match="cat_features"):
        CatBoostRegressor(cat_features=[0], device="cpu").fit(X, y_reg)

    y_clf = np.array([0, 0, 1, 1], dtype=np.int64)
    with pytest.raises(NotImplementedError, match="cat_features"):
        CatBoostClassifier(cat_features=[0], device="cpu").fit(X, y_clf)


def test_catboost_like_parameter_validation_fails_explicitly():
    """Ordered-boosting control parameters should fail before native training."""

    X = np.ones((4, 1), dtype=np.float32)
    y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    with pytest.raises(TypeError, match="ordered_boosting"):
        CatBoostRegressor(ordered_boosting=1, device="cpu").fit(X, y)
    with pytest.raises(ValueError, match="permutation_count"):
        CatBoostRegressor(permutation_count=0, device="cpu").fit(X, y)
