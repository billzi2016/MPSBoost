"""Real sklearn compatibility tests for estimator protocol behavior."""

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
clone = pytest.importorskip("sklearn.base").clone
model_selection = pytest.importorskip("sklearn.model_selection")
GridSearchCV = model_selection.GridSearchCV
cross_val_score = model_selection.cross_val_score
get_tags = pytest.importorskip("sklearn.utils").get_tags

import mpsboost as mb


def test_sklearn_clone_preserves_constructor_parameters():
    """sklearn.clone must rebuild the estimator from get_params without fitted state."""

    model = mb.GradientBoostingRegressor(
        n_estimators=2,
        max_depth=0,
        min_samples_leaf=1,
        device="cpu",
    )
    cloned = clone(model)

    assert cloned is not model
    assert cloned.get_params() == model.get_params()
    assert not hasattr(cloned, "model_")


def test_sklearn_get_tags_recognizes_regressor_type():
    """sklearn should recognize the estimator as a regressor without runtime inheritance."""

    tags = get_tags(mb.GradientBoostingRegressor(device="cpu"))

    assert tags.estimator_type == "regressor"
    assert tags.target_tags.required is True


def test_cross_val_score_uses_estimator_score_method():
    """cross_val_score should work through fit, predict, and score without custom wrappers."""

    X = np.arange(12, dtype=np.float32).reshape(6, 2)
    y = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], dtype=np.float32)
    model = mb.GradientBoostingRegressor(
        n_estimators=1,
        max_depth=0,
        min_samples_leaf=1,
        device="cpu",
    )

    scores = cross_val_score(model, X, y, cv=3)

    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))


def test_grid_search_cv_runs_with_standard_estimator_protocol():
    """GridSearchCV should tune constructor parameters without an MPS-specific search class."""

    X = np.arange(16, dtype=np.float32).reshape(8, 2)
    y = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0], dtype=np.float32)
    search = GridSearchCV(
        mb.GradientBoostingRegressor(
            max_depth=0,
            min_samples_leaf=1,
            device="cpu",
        ),
        param_grid={"n_estimators": [1, 2], "learning_rate": [0.1]},
        cv=2,
        n_jobs=1,
    )

    search.fit(X, y)

    assert isinstance(search.best_estimator_, mb.GradientBoostingRegressor)
    assert search.best_params_["n_estimators"] in {1, 2}
    assert np.isfinite(search.best_score_)
