"""Real-world model-family stability checks.

These tests use bundled sklearn datasets so they never download data. They cover
implemented estimator families on real numeric tabular data and verify
persistence, repeatability, and cache independence without replacing unit-level
model I/O tests.
"""

from __future__ import annotations

import numpy as np
import pytest

import mpsboost as mb
from mpsboost._cache import CacheKey, clear_cache, read_cache_bytes, write_cache_bytes

sklearn_datasets = pytest.importorskip("sklearn.datasets")
sklearn_model_selection = pytest.importorskip("sklearn.model_selection")
sklearn_preprocessing = pytest.importorskip("sklearn.preprocessing")


def _diabetes_split():
    """Return a compact deterministic Diabetes regression split."""

    dataset = sklearn_datasets.load_diabetes()
    X = dataset.data.astype(np.float32, copy=False)
    y = dataset.target.astype(np.float32, copy=False)
    return sklearn_model_selection.train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=1851,
    )


def _breast_cancer_split():
    """Return a deterministic scaled Breast Cancer binary-classification split."""

    dataset = sklearn_datasets.load_breast_cancer()
    X = dataset.data.astype(np.float32, copy=False)
    y = dataset.target.astype(np.int64, copy=False)
    X_train, X_test, y_train, y_test = sklearn_model_selection.train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=1852,
        stratify=y,
    )
    scaler = sklearn_preprocessing.StandardScaler()
    return (
        scaler.fit_transform(X_train).astype(np.float32, copy=False),
        scaler.transform(X_test).astype(np.float32, copy=False),
        y_train,
        y_test,
    )


def test_real_world_regression_families_round_trip_and_repeatability(tmp_path):
    """Implemented regression families should be persistent and deterministic."""

    X_train, X_test, y_train, _ = _diabetes_split()
    estimators = (
        mb.GradientBoostingRegressor(
            n_estimators=8,
            learning_rate=0.08,
            max_depth=2,
            max_bins=32,
            min_samples_leaf=6,
            min_child_weight=0.0,
            random_state=1851,
            device="cpu",
        ),
        mb.DecisionTreeRegressor(
            max_depth=3,
            max_bins=32,
            min_samples_leaf=6,
            min_child_weight=0.0,
            random_state=1851,
            device="cpu",
        ),
        mb.RandomForestRegressor(
            n_estimators=3,
            max_depth=3,
            max_bins=32,
            min_samples_leaf=6,
            min_child_weight=0.0,
            sample_fraction=0.8,
            bootstrap=True,
            random_state=1851,
            device="cpu",
        ),
        mb.ExtraTreesRegressor(
            n_estimators=3,
            max_depth=3,
            max_bins=32,
            min_samples_leaf=6,
            min_child_weight=0.0,
            sample_fraction=0.8,
            bootstrap=True,
            random_state=1851,
            device="cpu",
        ),
        mb.CatBoostRegressor(
            n_estimators=8,
            learning_rate=0.08,
            max_depth=2,
            max_bins=32,
            min_samples_leaf=6,
            min_child_weight=0.0,
            random_state=1851,
            device="cpu",
        ),
    )

    for index, estimator in enumerate(estimators):
        fitted = estimator.fit(X_train, y_train)
        predictions = fitted.predict(X_test)
        repeat = type(estimator)(**estimator.get_params()).fit(X_train, y_train)
        path = tmp_path / f"regression-{index}.mb"
        fitted.save_model(path)
        restored = type(estimator)(device="cpu").load_model(path)

        assert np.all(np.isfinite(predictions))
        np.testing.assert_allclose(repeat.predict(X_test), predictions)
        np.testing.assert_allclose(restored.predict(X_test), predictions)


def test_real_world_classification_families_round_trip_and_repeatability(tmp_path):
    """Implemented binary classifiers should preserve probabilities across reloads."""

    X_train, X_test, y_train, _ = _breast_cancer_split()
    estimators = (
        mb.GradientBoostingClassifier(
            n_estimators=8,
            learning_rate=0.1,
            max_depth=2,
            max_bins=32,
            min_samples_leaf=8,
            min_child_weight=0.0,
            random_state=1852,
            device="cpu",
        ),
        mb.DecisionTreeClassifier(
            max_depth=3,
            max_bins=32,
            min_samples_leaf=8,
            min_child_weight=0.0,
            random_state=1852,
            device="cpu",
        ),
        mb.RandomForestClassifier(
            n_estimators=3,
            max_depth=3,
            max_bins=32,
            min_samples_leaf=8,
            min_child_weight=0.0,
            sample_fraction=0.8,
            bootstrap=True,
            random_state=1852,
            device="cpu",
        ),
        mb.ExtraTreesClassifier(
            n_estimators=3,
            max_depth=3,
            max_bins=32,
            min_samples_leaf=8,
            min_child_weight=0.0,
            sample_fraction=0.8,
            bootstrap=True,
            random_state=1852,
            device="cpu",
        ),
        mb.CatBoostClassifier(
            n_estimators=8,
            learning_rate=0.1,
            max_depth=2,
            max_bins=32,
            min_samples_leaf=8,
            min_child_weight=0.0,
            random_state=1852,
            device="cpu",
        ),
    )

    for index, estimator in enumerate(estimators):
        fitted = estimator.fit(X_train, y_train)
        probabilities = fitted.predict_proba(X_test)
        repeat = type(estimator)(**estimator.get_params()).fit(X_train, y_train)
        path = tmp_path / f"classification-{index}.mb"
        fitted.save_model(path)
        restored = type(estimator)(device="cpu").load_model(path)

        assert np.all(np.isfinite(probabilities))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        np.testing.assert_allclose(repeat.predict_proba(X_test), probabilities)
        np.testing.assert_allclose(restored.predict_proba(X_test), probabilities)


def test_real_world_cache_corruption_and_deletion_do_not_change_training(
    tmp_path, monkeypatch
):
    """Damaged or deleted L2 cache records must not affect real-data training."""

    monkeypatch.setenv("MPSBOOST_CACHE_DIR", str(tmp_path / "cache"))
    X_train, X_test, y_train, _ = _diabetes_split()
    parameters = dict(
        n_estimators=8,
        learning_rate=0.08,
        max_depth=2,
        max_bins=32,
        min_samples_leaf=6,
        min_child_weight=0.0,
        random_state=1853,
        device="cpu",
    )
    key = CacheKey("tuning", "v-real-world", {"dataset": "diabetes"})

    baseline = mb.GradientBoostingRegressor(**parameters).fit(X_train, y_train).predict(X_test)
    path = write_cache_bytes(key, b"valid-but-unused")
    path.write_bytes(b"corrupted")
    assert read_cache_bytes(key) is None
    after_corruption = (
        mb.GradientBoostingRegressor(**parameters).fit(X_train, y_train).predict(X_test)
    )
    write_cache_bytes(key, b"valid-but-unused")
    assert clear_cache() == 1
    after_clear = (
        mb.GradientBoostingRegressor(**parameters).fit(X_train, y_train).predict(X_test)
    )

    np.testing.assert_allclose(after_corruption, baseline)
    np.testing.assert_allclose(after_clear, baseline)
