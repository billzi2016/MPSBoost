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


def test_random_forest_regressor_trains_independent_native_trees():
    """RandomForestRegressor should aggregate real native decision trees."""

    X = np.array(
        [
            [0.0, 1.0],
            [0.1, 1.0],
            [1.0, 0.0],
            [1.1, 0.0],
            [2.0, 1.0],
            [2.1, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0.0, 0.0, 4.0, 4.0, 8.0, 8.0], dtype=np.float32)
    model = RandomForestRegressor(
        n_estimators=3,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        max_features=1.0,
        sample_fraction=1.0,
        random_state=3,
        device="cpu",
    ).fit(X, y)

    predictions = model.predict(X)

    assert model.n_estimators_ == 3
    assert len(model.estimators_) == 3
    assert predictions.shape == (6,)
    assert model.score(X, y) > 0.0
    assert model.feature_importances_.shape == (2,)
    with pytest.raises(ValueError, match="feature count"):
        model.predict(np.ones((2, 3), dtype=np.float32))


def test_random_forest_forwards_sample_weight_to_native_trees():
    """Forest row sampling should slice sample weights together with labels and features."""

    X = np.ones((4, 1), dtype=np.float32)
    y = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    model = RandomForestRegressor(
        n_estimators=2,
        max_depth=0,
        min_samples_leaf=1,
        min_child_weight=0.0,
        sample_fraction=1.0,
        bootstrap=False,
        random_state=19,
        device="cpu",
    ).fit(X, y, sample_weight=np.array([10.0, 10.0, 1.0, 1.0]))

    assert float(model.predict(X)[0]) < 5.0
    assert model.training_summary_["weighted"] is True


def test_random_forest_classifier_trains_independent_native_trees():
    """RandomForestClassifier should average real native tree probabilities."""

    X = np.array([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]], dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    model = RandomForestClassifier(
        n_estimators=3,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        sample_fraction=1.0,
        random_state=5,
        device="cpu",
    ).fit(X, y)

    probabilities = model.predict_proba(X)

    assert model.n_estimators_ == 3
    assert probabilities.shape == (6, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[:3, 1].mean() < probabilities[3:, 1].mean()
    assert model.score(X, y) >= 5.0 / 6.0
    with pytest.raises(ValueError, match="feature count"):
        model.predict_proba(np.ones((2, 2), dtype=np.float32))


def test_random_forest_random_state_is_deterministic():
    """The same random_state should reproduce row and feature sampling exactly."""

    X = np.array(
        [[0.0, 1.0], [0.1, 1.0], [1.0, 0.0], [1.1, 0.0], [2.0, 1.0], [2.1, 1.0]],
        dtype=np.float32,
    )
    y = np.array([0.0, 0.0, 4.0, 4.0, 8.0, 8.0], dtype=np.float32)
    parameters = dict(
        n_estimators=3,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        max_features=0.5,
        sample_fraction=0.8,
        random_state=23,
        device="cpu",
    )
    first = RandomForestRegressor(**parameters).fit(X, y)
    second = RandomForestRegressor(**parameters).fit(X, y)

    np.testing.assert_allclose(first.predict(X), second.predict(X))
    assert [item.tolist() for item in first.feature_subsets_] == [
        item.tolist() for item in second.feature_subsets_
    ]


def test_random_forest_n_jobs_preserves_deterministic_results():
    """Parallel tree fitting should preserve the deterministic sampling contract."""

    X = np.array(
        [[0.0, 1.0], [0.1, 1.0], [1.0, 0.0], [1.1, 0.0], [2.0, 1.0], [2.1, 1.0]],
        dtype=np.float32,
    )
    y = np.array([0.0, 0.0, 4.0, 4.0, 8.0, 8.0], dtype=np.float32)
    parameters = dict(
        n_estimators=4,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        max_features=0.5,
        sample_fraction=0.8,
        random_state=29,
        device="cpu",
    )
    serial = RandomForestRegressor(n_jobs=1, **parameters).fit(X, y)
    parallel = RandomForestRegressor(n_jobs=2, **parameters).fit(X, y)

    np.testing.assert_allclose(serial.predict(X), parallel.predict(X))
    assert parallel.training_summary_["n_jobs"] == 2
    assert [item.tolist() for item in serial.feature_subsets_] == [
        item.tolist() for item in parallel.feature_subsets_
    ]


def test_random_forest_validates_parameters():
    """Forest parameter boundaries should fail explicitly."""

    X = np.ones((4, 1), dtype=np.float32)
    y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="max_features"):
        RandomForestRegressor(max_features=0.0, device="cpu").fit(X, y)
    with pytest.raises(TypeError, match="bootstrap"):
        RandomForestRegressor(bootstrap=1, device="cpu").fit(X, y)
    with pytest.raises(ValueError, match="n_jobs"):
        RandomForestRegressor(n_jobs=0, device="cpu").fit(X, y)


def test_extra_trees_regressor_and_classifier_share_forest_contracts():
    """ExtraTrees ensembles should reuse RF aggregation with native random-threshold trees."""

    X = np.array([[float(value)] for value in range(8)], dtype=np.float32)
    y_reg = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0], dtype=np.float32)
    regressor = ExtraTreesRegressor(
        n_estimators=3,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        sample_fraction=1.0,
        random_state=303,
        device="cpu",
    ).fit(X, y_reg)
    assert regressor.predict(X).shape == (8,)
    assert regressor.feature_importances_.shape == (1,)

    y_clf = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    classifier = ExtraTreesClassifier(
        n_estimators=3,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        sample_fraction=1.0,
        random_state=307,
        device="cpu",
    ).fit(X, y_clf)
    assert classifier.predict_proba(X).shape == (8, 2)
