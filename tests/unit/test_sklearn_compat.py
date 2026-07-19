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


def _tag_value(tags, name: str):
    """Read a sklearn tag from either old dictionary tags or new structured tags."""

    if isinstance(tags, dict):
        return tags[name]
    return getattr(tags, name)


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

    assert _tag_value(tags, "estimator_type") == "regressor"
    if isinstance(tags, dict):
        assert tags["requires_y"] is True
    else:
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


def test_classifier_works_with_standard_sklearn_protocol():
    """Binary classifier should work with sklearn tags, clone, and GridSearchCV."""

    X = np.array([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]], dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    classifier = mb.GradientBoostingClassifier(
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        device="cpu",
    )

    tags = get_tags(classifier)
    cloned = clone(classifier)

    assert _tag_value(tags, "estimator_type") == "classifier"
    assert cloned.get_params() == classifier.get_params()

    search = GridSearchCV(
        classifier,
        param_grid={"n_estimators": [1, 2], "learning_rate": [0.5]},
        cv=2,
        n_jobs=1,
    )
    search.fit(X, y)

    assert isinstance(search.best_estimator_, mb.GradientBoostingClassifier)
    assert search.best_score_ >= 0.0


def test_decision_tree_estimators_follow_standard_sklearn_protocol():
    """Decision tree estimators should clone and tune through the standard protocol."""

    X_reg = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32)
    y_reg = np.array([0.0, 0.0, 4.0, 4.0], dtype=np.float32)
    regressor = mb.DecisionTreeRegressor(
        min_samples_leaf=1,
        min_child_weight=0.0,
        device="cpu",
    )
    cloned_regressor = clone(regressor)

    assert cloned_regressor.get_params() == regressor.get_params()
    assert "n_estimators" not in regressor.get_params()
    assert cross_val_score(regressor, X_reg, y_reg, cv=2).shape == (2,)

    X_clf = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32)
    y_clf = np.array([0, 0, 1, 1], dtype=np.int64)
    classifier_search = GridSearchCV(
        mb.DecisionTreeClassifier(
            min_samples_leaf=1,
            min_child_weight=0.0,
            device="cpu",
        ),
        param_grid={"max_depth": [0, 1]},
        cv=2,
        n_jobs=1,
    )
    classifier_search.fit(X_clf, y_clf)

    assert isinstance(classifier_search.best_estimator_, mb.DecisionTreeClassifier)


def test_random_forest_estimators_follow_standard_sklearn_protocol():
    """Random forest estimators should clone and tune through standard sklearn tools."""

    X_reg = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32)
    y_reg = np.array([0.0, 0.0, 4.0, 4.0], dtype=np.float32)
    regressor = mb.RandomForestRegressor(
        n_estimators=2,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        sample_fraction=1.0,
        n_jobs=1,
        random_state=1,
        device="cpu",
    )
    assert clone(regressor).get_params() == regressor.get_params()
    assert cross_val_score(regressor, X_reg, y_reg, cv=2).shape == (2,)

    X_clf = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32)
    y_clf = np.array([0, 0, 1, 1], dtype=np.int64)
    classifier_search = GridSearchCV(
        mb.RandomForestClassifier(
            n_estimators=2,
            min_samples_leaf=1,
            min_child_weight=0.0,
            sample_fraction=1.0,
            n_jobs=1,
            random_state=2,
            device="cpu",
        ),
        param_grid={"max_depth": [0, 1]},
        cv=2,
        n_jobs=1,
    )
    classifier_search.fit(X_clf, y_clf)

    assert isinstance(classifier_search.best_estimator_, mb.RandomForestClassifier)


def test_extra_trees_estimators_follow_standard_sklearn_protocol():
    """ExtraTrees estimators should clone and tune through standard sklearn tools."""

    X_reg = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32)
    y_reg = np.array([0.0, 0.0, 4.0, 4.0], dtype=np.float32)
    regressor = mb.ExtraTreesRegressor(
        n_estimators=2,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        sample_fraction=1.0,
        n_jobs=1,
        random_state=41,
        device="cpu",
    )
    assert clone(regressor).get_params() == regressor.get_params()
    assert cross_val_score(regressor, X_reg, y_reg, cv=2).shape == (2,)

    X_clf = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32)
    y_clf = np.array([0, 0, 1, 1], dtype=np.int64)
    classifier_search = GridSearchCV(
        mb.ExtraTreesClassifier(
            n_estimators=2,
            min_samples_leaf=1,
            min_child_weight=0.0,
            sample_fraction=1.0,
            n_jobs=1,
            random_state=43,
            device="cpu",
        ),
        param_grid={"max_depth": [0, 1]},
        cv=2,
        n_jobs=1,
    )
    classifier_search.fit(X_clf, y_clf)

    assert isinstance(classifier_search.best_estimator_, mb.ExtraTreesClassifier)
