"""Multiclass classifier behavior for the public sklearn-style API."""

import numpy as np
import pytest

from mpsboost import GradientBoostingClassifier, MPSBoostClassifier

sklearn_model_selection = pytest.importorskip("sklearn.model_selection")


def test_multiclass_classifier_trains_one_vs_rest_models():
    """Three-class labels should train real OvR binary native models."""

    X = np.asarray(
        [[0.0], [0.1], [0.2], [1.0], [1.1], [1.2], [2.0], [2.1], [2.2]],
        dtype=np.float32,
    )
    y = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    model = MPSBoostClassifier(
        n_estimators=3,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        device="cpu",
    ).fit(X, y)

    probabilities = model.predict_proba(X)

    assert model.classes_.tolist() == [0, 1, 2]
    assert probabilities.shape == (9, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert model.predict(X).tolist() == y.tolist()
    assert model.score(X, y) == 1.0
    assert len(model.estimators_) == 3
    assert model.decision_function(X).shape == (9, 3)


def test_binary_classifier_preserves_non_zero_numeric_labels():
    """Binary classification should map arbitrary numeric labels back to users."""

    X = np.asarray([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]], dtype=np.float32)
    y = np.asarray([10.0, 10.0, 10.0, 20.0, 20.0, 20.0], dtype=np.float64)
    model = MPSBoostClassifier(
        n_estimators=3,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        device="cpu",
    ).fit(X, y)

    assert model.classes_.tolist() == [10.0, 20.0]
    assert model.predict(X).tolist() == y.tolist()
    assert model.score(X, y) == 1.0


def test_gradient_boosting_classifier_alias_supports_multiclass():
    """The concise sklearn-style alias should expose the same multiclass behavior."""

    model = GradientBoostingClassifier(
        n_estimators=1,
        max_depth=0,
        min_samples_leaf=1,
        device="cpu",
    ).fit(np.ones((6, 1), dtype=np.float32), np.asarray([0, 0, 1, 1, 2, 2]))

    assert model.predict_proba(np.ones((2, 1), dtype=np.float32)).shape == (2, 3)


def test_multiclass_supports_non_zero_classes_sample_weight_and_parallel_ovr():
    """OvR should support numeric class values, weighted scoring, and n_jobs."""

    X = np.asarray(
        [[0.0], [0.1], [0.2], [1.0], [1.1], [1.2], [2.0], [2.1], [2.2]],
        dtype=np.float32,
    )
    y = np.asarray([10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0])
    weights = np.asarray([1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0])
    model = GradientBoostingClassifier(
        n_estimators=3,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        n_jobs=2,
        device="cpu",
    ).fit(X, y, sample_weight=weights)

    assert model.classes_.tolist() == [10.0, 20.0, 30.0]
    assert model.training_summary_["n_jobs"] == 2
    assert model.score(X, y, sample_weight=weights) == 1.0


def test_multiclass_grid_search_uses_standard_sklearn_protocol():
    """GridSearchCV should tune multiclass classifiers without custom wrappers."""

    X = np.asarray(
        [[0.0], [0.1], [0.2], [1.0], [1.1], [1.2], [2.0], [2.1], [2.2]],
        dtype=np.float32,
    )
    y = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    search = sklearn_model_selection.GridSearchCV(
        GradientBoostingClassifier(
            max_depth=1,
            min_samples_leaf=1,
            min_child_weight=0.0,
            n_jobs=2,
            device="cpu",
        ),
        param_grid={"n_estimators": [1, 2], "learning_rate": [0.5]},
        cv=3,
        n_jobs=1,
    )

    search.fit(X, y)

    assert isinstance(search.best_estimator_, GradientBoostingClassifier)
    assert search.best_score_ >= 0.0


def test_multiclass_model_persistence_fails_until_container_exists(tmp_path):
    """OvR models must not be saved through the binary native model format."""

    model = MPSBoostClassifier(
        n_estimators=1,
        max_depth=0,
        min_samples_leaf=1,
        device="cpu",
    ).fit(np.ones((6, 1), dtype=np.float32), np.asarray([0, 0, 1, 1, 2, 2]))

    with pytest.raises(NotImplementedError, match="multiclass model persistence"):
        model.save_model(tmp_path / "multiclass.mb")
