"""Real MPS parity tests for industrial tabular semantics.

These tests validate that high-level tabular features keep the same estimator
semantics on the CPU oracle and the real MPS training backend. They intentionally
exercise the public Python API instead of private kernels so the full adapter,
quantization, native training, and prediction path are covered together.
"""

import numpy as np
import pytest

from mpsboost import MPSBoostClassifier, MPSBoostRegressor

pytestmark = pytest.mark.gpu


def test_real_mps_regressor_matches_cpu_with_industrial_semantics():
    """CPU and MPS should agree with missing, categorical, constraints, and regularization."""

    X = np.asarray(
        [
            [0.0, "a", 2.0],
            [1.0, "a", 2.0],
            [2.0, "b", 1.0],
            [3.0, "b", 1.0],
            [4.0, "c", np.nan],
            [5.0, "c", np.nan],
            [6.0, "d", 0.0],
            [7.0, "d", 0.0],
        ],
        dtype=object,
    )
    y = np.asarray([0.0, 0.2, 1.0, 1.2, 2.0, 2.2, 3.0, 3.2], dtype=np.float32)
    parameters = dict(
        n_estimators=3,
        learning_rate=0.5,
        max_depth=2,
        max_bins=8,
        min_samples_leaf=1,
        min_child_weight=0.0,
        reg_lambda=0.0,
        reg_alpha=0.05,
        max_delta_step=1.0,
        categorical_features=[1],
        monotonic_constraints=[1, 0, 0],
        interaction_constraints=[[0, 1], [2]],
    )
    cpu = MPSBoostRegressor(device="cpu", **parameters).fit(X, y)
    mps = MPSBoostRegressor(device="mps", **parameters).fit(X, y)

    np.testing.assert_allclose(mps.predict(X), cpu.predict(X), rtol=2e-4, atol=2e-4)
    assert mps.device_ == "mps"


def test_real_mps_classifier_matches_cpu_with_missing_and_regularization():
    """Binary-logistic CPU and MPS training should agree under S15 regularization controls."""

    X = np.asarray(
        [
            [0.0, np.nan],
            [0.1, np.nan],
            [1.0, 1.0],
            [1.1, 1.0],
            [2.0, 0.0],
            [2.1, 0.0],
        ],
        dtype=np.float32,
    )
    y = np.asarray([0, 0, 1, 1, 1, 1], dtype=np.int64)
    parameters = dict(
        n_estimators=3,
        learning_rate=0.5,
        max_depth=2,
        max_bins=8,
        min_samples_leaf=1,
        min_child_weight=0.0,
        reg_lambda=0.0,
        reg_alpha=0.01,
        max_delta_step=1.0,
        monotonic_constraints=[1, 0],
        interaction_constraints=[[0, 1]],
    )
    cpu = MPSBoostClassifier(device="cpu", **parameters).fit(X, y)
    mps = MPSBoostClassifier(device="mps", **parameters).fit(X, y)

    np.testing.assert_allclose(
        mps.predict_proba(X),
        cpu.predict_proba(X),
        rtol=2e-4,
        atol=2e-4,
    )
    np.testing.assert_array_equal(mps.predict(X), cpu.predict(X))
    assert mps.device_ == "mps"
