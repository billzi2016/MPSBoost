"""End-to-end tests from Python estimators to real MPS training and prediction.

This file runs on self-hosted Apple Silicon hardware, covering Python to quantization
to GPU gradient/histogram to multi-round trees to prediction. Mocks, CPU fallback,
and internal-kernel-only checks are forbidden.
"""

import numpy as np
import pytest

from mpsboost import MPSBoostRegressor

pytestmark = pytest.mark.gpu


def test_real_mps_regressor_matches_cpu_model_on_stable_dataset():
    """Under frozen semantics, MPS and CPU oracle structure and predictions agree within tolerance."""

    X = np.arange(128, dtype=np.float32).reshape(64, 2)
    y = np.where(X[:, 0] < 64, -1.0, 2.0).astype(np.float64)
    parameters = dict(
        n_estimators=5,
        learning_rate=0.3,
        max_depth=3,
        max_bins=32,
        min_samples_leaf=2,
        min_child_weight=1.0,
        reg_lambda=1.0,
    )
    cpu = MPSBoostRegressor(device="cpu", **parameters).fit(X, y)
    mps = MPSBoostRegressor(device="mps", **parameters).fit(X, y)
    np.testing.assert_allclose(mps.predict(X), cpu.predict(X), rtol=2e-4, atol=2e-4)
    assert mps.device_ == "mps"
    assert mps.n_estimators_ == parameters["n_estimators"]
