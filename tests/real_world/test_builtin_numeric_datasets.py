"""Default real-world acceptance tests for built-in sklearn datasets.

These tests use public datasets bundled with scikit-learn. They do not download
data, do not write raw datasets into the repository, and do not mock MPSBoost
estimators. Thresholds are intentionally modest because S18 is an acceptance
gate for reliability and workflow coverage, not a leaderboard benchmark.
"""

from __future__ import annotations

import numpy as np
import pytest

import mpsboost as mb

from .dataset_matrix import active_default_datasets

sklearn_datasets = pytest.importorskip("sklearn.datasets")
sklearn_metrics = pytest.importorskip("sklearn.metrics")
sklearn_model_selection = pytest.importorskip("sklearn.model_selection")
sklearn_preprocessing = pytest.importorskip("sklearn.preprocessing")


def test_dataset_matrix_has_active_default_entries():
    """S18 should expose a real dataset matrix instead of ad hoc test names."""

    names = {item.name for item in active_default_datasets()}
    assert names == {"Breast Cancer Wisconsin", "Diabetes", "Digits", "Iris"}


def test_iris_multiclass_classification_cpu_acceptance():
    """Multiclass classification should work on the bundled Iris dataset."""

    dataset = sklearn_datasets.load_iris()
    X = dataset.data.astype(np.float32, copy=False)
    y = dataset.target.astype(np.int64, copy=False)
    X_train, X_test, y_train, y_test = sklearn_model_selection.train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=1803,
        stratify=y,
    )
    model = mb.GradientBoostingClassifier(
        n_estimators=12,
        learning_rate=0.2,
        max_depth=2,
        max_bins=32,
        min_samples_leaf=2,
        min_child_weight=0.0,
        reg_lambda=1.0,
        random_state=1803,
        device="cpu",
    ).fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)
    predictions = model.predict(X_test)

    assert model.training_summary_["strategy"] == "native_softmax"
    assert probabilities.shape == (y_test.shape[0], 3)
    assert float(sklearn_metrics.accuracy_score(y_test, predictions)) >= 0.90


def test_iris_multiclass_grid_search_cv_acceptance():
    """Multiclass classifiers should work with standard sklearn GridSearchCV."""

    dataset = sklearn_datasets.load_iris()
    X = dataset.data.astype(np.float32, copy=False)
    y = dataset.target.astype(np.int64, copy=False)
    search = sklearn_model_selection.GridSearchCV(
        mb.GradientBoostingClassifier(
            n_estimators=4,
            min_samples_leaf=2,
            min_child_weight=0.0,
            device="cpu",
        ),
        param_grid={
            "learning_rate": [0.1, 0.2],
            "max_depth": [1, 2],
        },
        cv=3,
        n_jobs=1,
    )

    search.fit(X, y)

    assert search.best_estimator_.training_summary_["strategy"] == "native_softmax"
    assert search.best_estimator_.classes_.tolist() == [0, 1, 2]
    assert float(search.best_score_) >= 0.90


def test_digits_multiclass_classification_cpu_acceptance():
    """Flattened image-like multiclass data should work on the bundled Digits dataset."""

    dataset = sklearn_datasets.load_digits()
    X = dataset.data.astype(np.float32, copy=False) / 16.0
    y = dataset.target.astype(np.int64, copy=False)
    X_train, X_test, y_train, y_test = sklearn_model_selection.train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=1804,
        stratify=y,
    )
    model = mb.GradientBoostingClassifier(
        n_estimators=16,
        learning_rate=0.08,
        max_depth=2,
        max_bins=32,
        min_samples_leaf=24,
        min_child_weight=0.0,
        reg_lambda=1.0,
        max_delta_step=0.25,
        random_state=1804,
        device="cpu",
    ).fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert model.training_summary_["strategy"] == "native_softmax"
    assert float(sklearn_metrics.accuracy_score(y_test, predictions)) >= 0.75


def test_breast_cancer_binary_classification_cpu_acceptance():
    """Binary classification should work on a bundled real medical tabular dataset."""

    dataset = sklearn_datasets.load_breast_cancer()
    X = dataset.data.astype(np.float32, copy=False)
    y = dataset.target.astype(np.int64, copy=False)
    X_train, X_test, y_train, y_test = sklearn_model_selection.train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=1801,
        stratify=y,
    )
    scaler = sklearn_preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32, copy=False)
    X_test = scaler.transform(X_test).astype(np.float32, copy=False)
    model = mb.GradientBoostingClassifier(
        n_estimators=10,
        learning_rate=0.1,
        max_depth=2,
        max_bins=64,
        min_samples_leaf=8,
        min_child_weight=0.0,
        reg_lambda=1.0,
        max_delta_step=1.0,
        random_state=1801,
        device="cpu",
    ).fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)
    predictions = model.predict(X_test)

    assert probabilities.shape == (y_test.shape[0], 2)
    assert float(sklearn_metrics.accuracy_score(y_test, predictions)) >= 0.90
    assert float(sklearn_metrics.roc_auc_score(y_test, probabilities[:, 1])) >= 0.95


def test_diabetes_regression_cpu_acceptance():
    """Regression should work on a bundled real clinical tabular dataset."""

    dataset = sklearn_datasets.load_diabetes()
    X = dataset.data.astype(np.float32, copy=False)
    y = dataset.target.astype(np.float32, copy=False)
    X_train, X_test, y_train, y_test = sklearn_model_selection.train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=1802,
    )
    model = mb.GradientBoostingRegressor(
        n_estimators=32,
        learning_rate=0.08,
        max_depth=2,
        max_bins=64,
        min_samples_leaf=6,
        min_child_weight=0.0,
        reg_lambda=1.0,
        random_state=1802,
        device="cpu",
    ).fit(X_train, y_train)

    predictions = model.predict(X_test)
    r2 = float(sklearn_metrics.r2_score(y_test, predictions))

    assert predictions.shape == y_test.shape
    assert np.all(np.isfinite(predictions))
    assert r2 >= 0.20
