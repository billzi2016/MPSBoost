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




def test_feature_importance_reads_native_tree_splits():
    """feature_importances_ must be derived from real fitted tree nodes, not a mock summary."""

    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    model = MPSBoostRegressor(
        n_estimators=1,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        device="cpu",
    ).fit(X, y)

    gain_importance = model.feature_importances_
    split_importance = model.feature_importance(kind="split")

    assert gain_importance.shape == (2,)
    assert split_importance.shape == (2,)
    assert np.isclose(float(gain_importance.sum()), 1.0)
    assert np.isclose(float(split_importance.sum()), 1.0)
    assert gain_importance[0] == pytest.approx(1.0)
    assert split_importance[0] == pytest.approx(1.0)


def test_feature_importance_requires_fitted_model_and_valid_kind():
    """feature importance should share the estimator fitted-state contract."""

    model = MPSBoostRegressor(device="cpu")
    with pytest.raises(NotFittedError):
        _ = model.feature_importances_

    fitted = MPSBoostRegressor(
        n_estimators=1,
        max_depth=0,
        min_samples_leaf=1,
        device="cpu",
    ).fit(np.ones((2, 1), dtype=np.float32), np.array([1.0, 2.0]))
    assert fitted.feature_importances_.tolist() == [0.0]
    with pytest.raises(ValueError, match="feature importance kind"):
        fitted.feature_importance(kind="permutation")


def test_permutation_importance_uses_estimator_score_for_regression():
    """Permutation importance should use score() instead of duplicating prediction metrics."""

    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0.0, 0.0, 5.0, 5.0, 10.0, 10.0], dtype=np.float32)
    model = MPSBoostRegressor(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        device="cpu",
    ).fit(X, y)

    result = model.permutation_importance(X, y, n_repeats=4, random_state=7)

    assert result["importances"].shape == (2, 4)
    assert np.isfinite(result["baseline_score"])
    assert result["importances_mean"][0] > result["importances_mean"][1]


def test_approximate_shap_values_explain_prediction_delta():
    """SHAP-like approximation should decompose prediction minus baseline."""

    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0.0, 0.0, 5.0, 5.0, 10.0, 10.0], dtype=np.float32)
    model = MPSBoostRegressor(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        device="cpu",
    ).fit(X, y)

    result = model.approximate_shap_values(X, background=X[:2])
    contributions = result["contributions"]
    approximation = result["prediction_approximation"]

    assert contributions.shape == X.shape
    np.testing.assert_allclose(approximation, model.predict(X), atol=1e-5)
    assert np.mean(np.abs(contributions[:, 0])) > np.mean(np.abs(contributions[:, 1]))


def test_permutation_importance_validates_state_and_arguments():
    """Permutation importance should share fitted-state and input validation contracts."""

    model = MPSBoostRegressor(device="cpu")
    with pytest.raises(NotFittedError):
        model.permutation_importance(np.ones((2, 1), dtype=np.float32), np.ones(2))

    fitted = MPSBoostRegressor(
        n_estimators=1,
        max_depth=0,
        min_samples_leaf=1,
        device="cpu",
    ).fit(np.ones((2, 1), dtype=np.float32), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="n_repeats"):
        fitted.permutation_importance(
            np.ones((2, 1), dtype=np.float32), np.array([1.0, 2.0]), n_repeats=0
        )
    with pytest.raises(ValueError, match="feature count"):
        fitted.permutation_importance(
            np.ones((2, 2), dtype=np.float32), np.array([1.0, 2.0])
        )
    with pytest.raises(ValueError, match="background feature count"):
        fitted.approximate_shap_values(
            np.ones((2, 1), dtype=np.float32),
            background=np.ones((2, 2), dtype=np.float32),
        )
