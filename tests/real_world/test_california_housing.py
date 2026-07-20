"""Cached California Housing real-world regression acceptance test."""

from __future__ import annotations

import numpy as np
import pytest

import mpsboost as mb

sklearn_datasets = pytest.importorskip("sklearn.datasets")
sklearn_metrics = pytest.importorskip("sklearn.metrics")
sklearn_model_selection = pytest.importorskip("sklearn.model_selection")
sklearn_preprocessing = pytest.importorskip("sklearn.preprocessing")

from .download_datasets import DATA_ROOT


def _load_cached_california_housing():
    """Load California Housing from the ignored local cache without network access."""

    data_home = DATA_ROOT / "sklearn"
    try:
        return sklearn_datasets.fetch_california_housing(
            data_home=data_home,
            download_if_missing=False,
        )
    except OSError as exc:
        pytest.skip(
            "California Housing cache is missing; run "
            "`python tests/real_world/download_datasets.py california-housing`"
        )
        raise AssertionError("pytest.skip should stop execution") from exc


def test_california_housing_cached_regression_acceptance():
    """Regression should work on cached medium-size real-world housing data."""

    dataset = _load_cached_california_housing()
    X = dataset.data.astype(np.float32, copy=False)
    y = dataset.target.astype(np.float32, copy=False)
    X_train, X_test, y_train, y_test = sklearn_model_selection.train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=1820,
    )
    scaler = sklearn_preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32, copy=False)
    X_test = scaler.transform(X_test).astype(np.float32, copy=False)
    model = mb.GradientBoostingRegressor(
        n_estimators=48,
        learning_rate=0.08,
        max_depth=3,
        max_bins=128,
        min_samples_leaf=16,
        min_child_weight=0.0,
        reg_lambda=1.0,
        random_state=1820,
        device="cpu",
    ).fit(X_train, y_train)

    predictions = model.predict(X_test)
    r2 = float(sklearn_metrics.r2_score(y_test, predictions))

    assert predictions.shape == y_test.shape
    assert np.all(np.isfinite(predictions))
    assert r2 >= 0.45
