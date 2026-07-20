"""CPU-oracle unit tests for the multi-round boosting state machine.

Tests use the real C++ trainer, real binning, and sole tree implementation rather
than rewriting boosting in Python. Small data uses independent properties and
reproducible results, avoiding expectations computed by the implementation under test.
"""

import numpy as np
import pytest

from mpsboost import MPSBoostRegressor


def test_cpu_boosting_reduces_squared_error_and_is_deterministic():
    """Multi-round models reduce training error and repeated training predicts identically."""

    X = np.arange(32, dtype=np.float32).reshape(-1, 1)
    y = np.where(X[:, 0] < 16, -2.0, 3.0).astype(np.float64)
    parameters = dict(
        n_estimators=4,
        learning_rate=0.5,
        max_depth=2,
        max_bins=16,
        min_samples_leaf=1,
        min_child_weight=1.0,
        reg_lambda=0.0,
        device="cpu",
    )
    first = MPSBoostRegressor(**parameters).fit(X, y)
    second = MPSBoostRegressor(**parameters).fit(X, y)
    baseline_error = np.mean((y - np.mean(y)) ** 2)
    model_error = np.mean((y - first.predict(X)) ** 2)
    assert model_error < baseline_error * 0.02
    np.testing.assert_array_equal(first.predict(X), second.predict(X))
    assert first.n_estimators_ == 4


def test_fixed_training_boundaries_are_used_for_new_data():
    """New-data prediction must apply the training schema rather than refitting boundaries."""

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([0.0, 0.0, 10.0, 10.0])
    model = MPSBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        max_bins=4,
        min_samples_leaf=1,
        min_child_weight=1.0,
        reg_lambda=0.0,
        device="cpu",
    ).fit(X, y)
    prediction = model.predict(np.array([[-100.0], [100.0]], dtype=np.float32))
    assert prediction[0] < prediction[1]


def test_advanced_regression_objectives_train_and_predict_on_native_path():
    """Quantile, Poisson, and Tweedie losses should use the shared native trainer."""

    X = np.asarray([[0.0], [0.1], [1.0], [1.1], [2.0], [2.1]], dtype=np.float32)
    y = np.asarray([1.0, 1.0, 2.0, 2.0, 5.0, 5.0], dtype=np.float64)
    common = dict(
        n_estimators=3,
        learning_rate=0.3,
        max_depth=1,
        max_bins=8,
        min_samples_leaf=1,
        min_child_weight=0.0,
        reg_lambda=1.0,
        device="cpu",
    )

    quantile = MPSBoostRegressor(loss="quantile", quantile_alpha=0.75, **common).fit(X, y)
    poisson = MPSBoostRegressor(loss="poisson", **common).fit(X, y)
    tweedie = MPSBoostRegressor(loss="tweedie", tweedie_variance_power=1.5, **common).fit(X, y)

    assert quantile.training_summary_["native_objective"] == "quantile"
    assert poisson.training_summary_["native_objective"] == "poisson"
    assert tweedie.training_summary_["native_objective"] == "tweedie"
    for model in (quantile, poisson, tweedie):
        predictions = model.predict(X)
        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))
    assert np.all(poisson.predict(X) > 0.0)
    assert np.all(tweedie.predict(X) > 0.0)


@pytest.mark.parametrize(
    "name,value,message",
    [
        ("n_estimators", 0, "n_estimators"),
        ("learning_rate", 0.0, "learning_rate"),
        ("max_depth", -1, "max_depth"),
        ("max_bins", 1, "max_bins"),
        ("min_samples_leaf", 0, "min_samples_leaf"),
        ("loss", "bad", "loss"),
        ("quantile_alpha", 1.0, "quantile_alpha"),
        ("tweedie_variance_power", 2.0, "tweedie_variance_power"),
    ],
)
def test_invalid_training_parameters_fail_before_model(name, value, message):
    """Public parameter errors must not create fitted state."""

    model = MPSBoostRegressor(device="cpu", **{name: value})
    with pytest.raises((TypeError, ValueError), match=message):
        model.fit(np.ones((2, 1), dtype=np.float32), np.ones(2))
    assert not hasattr(model, "model_")
