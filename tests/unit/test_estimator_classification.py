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


def test_permutation_importance_uses_classifier_accuracy():
    """Classifier permutation importance should reuse accuracy score through the shared method."""

    X = np.array([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]], dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    model = MPSBoostClassifier(
        n_estimators=3,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        device="cpu",
    ).fit(X, y)

    result = model.permutation_importance(X, y, n_repeats=3, random_state=11)

    assert result["importances"].shape == (1, 3)
    assert result["baseline_score"] == 1.0
    assert result["importances_mean"][0] >= 0.0


def test_binary_classifier_trains_predicts_probabilities_and_scores():
    """GradientBoostingClassifier must use the real native binary-logistic objective."""

    X = np.array(
        [[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]],
        dtype=np.float32,
    )
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    model = MPSBoostClassifier(
        n_estimators=3,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        device="cpu",
    ).fit(X, y)

    probabilities = model.predict_proba(X)
    predictions = model.predict(X)

    assert model.classes_.tolist() == [0, 1]
    assert probabilities.shape == (6, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[:3, 1].mean() < probabilities[3:, 1].mean()
    assert predictions.tolist() == y.tolist()
    assert model.score(X, y) == 1.0
    assert model.feature_importances_.shape == (1,)


def test_sample_weight_changes_classifier_score():
    """Classifier score should use weighted accuracy when sample_weight is provided."""

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    model = MPSBoostClassifier(
        n_estimators=1,
        max_depth=0,
        min_samples_leaf=1,
        min_child_weight=0.0,
        device="cpu",
    ).fit(X, y)

    assert model.score(X, np.array([0, 1, 1, 1])) == 0.75
    assert model.score(
        X,
        np.array([0, 1, 1, 1]),
        sample_weight=np.array([10.0, 1.0, 1.0, 1.0]),
    ) == pytest.approx(3.0 / 13.0)


def test_classifier_accepts_multiclass_training_labels():
    """Public classifier should train multiclass labels through real OvR binary models."""

    model = MPSBoostClassifier(device="cpu")
    X = np.ones((3, 1), dtype=np.float32)
    fitted = model.fit(X, np.array([0, 1, 2]))
    assert fitted.classes_.tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="at least two classes"):
        model.fit(X, np.array([1, 1, 1]))
