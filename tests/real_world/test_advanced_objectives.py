"""Real-world acceptance for advanced regression objectives."""

from __future__ import annotations

import numpy as np
import pytest

import mpsboost as mb

sklearn_datasets = pytest.importorskip("sklearn.datasets")
sklearn_model_selection = pytest.importorskip("sklearn.model_selection")


def test_diabetes_advanced_objectives_produce_finite_predictions():
    """Quantile, Poisson, and Tweedie objectives should run on real tabular data."""

    dataset = sklearn_datasets.load_diabetes()
    X = dataset.data.astype(np.float32, copy=False)
    y = (dataset.target.astype(np.float64, copy=False) / 50.0) + 0.1
    X_train, X_test, y_train, _ = sklearn_model_selection.train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=1863,
    )
    common = dict(
        n_estimators=8,
        learning_rate=0.08,
        max_depth=2,
        max_bins=64,
        min_samples_leaf=6,
        min_child_weight=0.0,
        random_state=1863,
        device="cpu",
    )
    estimators = (
        mb.GradientBoostingRegressor(
            loss="quantile", quantile_alpha=0.75, **common
        ),
        mb.GradientBoostingRegressor(loss="poisson", **common),
        mb.GradientBoostingRegressor(
            loss="tweedie", tweedie_variance_power=1.5, **common
        ),
    )

    for estimator in estimators:
        fitted = estimator.fit(X_train, y_train)
        predictions = fitted.predict(X_test)
        assert fitted.training_summary_["native_objective"] == estimator.loss
        assert predictions.shape == (X_test.shape[0],)
        assert np.all(np.isfinite(predictions))
        if estimator.loss in {"poisson", "tweedie"}:
            assert np.all(predictions > 0.0)
