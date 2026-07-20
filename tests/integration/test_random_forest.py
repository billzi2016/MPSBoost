"""End-to-end real MPS integration tests for random forest estimators.

These tests run only on a real Apple Silicon runner. They verify that the public random forest
estimators can drive independent native decision-tree training through the MPS backend and produce
the same aggregate predictions as the CPU path under deterministic sampling.
"""

import numpy as np
import pytest

from mpsboost import RandomForestClassifier, RandomForestRegressor

pytestmark = pytest.mark.gpu


def test_real_mps_random_forest_regressor_matches_cpu_model():
    """CPU and MPS forests should agree when every tree sees deterministic full data."""

    X = np.arange(128, dtype=np.float32).reshape(64, 2)
    y = np.where(X[:, 0] < 64, -1.0, 2.0).astype(np.float64)
    parameters = dict(
        n_estimators=3,
        max_depth=3,
        max_bins=32,
        min_samples_leaf=2,
        min_child_weight=1.0,
        reg_lambda=1.0,
        max_features=1.0,
        sample_fraction=1.0,
        bootstrap=False,
        random_state=31,
    )
    cpu = RandomForestRegressor(device="cpu", **parameters).fit(X, y)
    mps = RandomForestRegressor(device="mps", **parameters).fit(X, y)

    np.testing.assert_allclose(mps.predict(X), cpu.predict(X), rtol=2e-4, atol=2e-4)
    assert mps.device_ == "mps"
    assert mps.training_summary_["tree_devices"] == ["mps"] * parameters["n_estimators"]
    assert mps.training_summary_["scheduling"] == "serial"
    assert mps.n_estimators_ == parameters["n_estimators"]


def test_real_mps_random_forest_classifier_matches_cpu_model():
    """CPU and MPS classifier forests should agree on aggregate probabilities."""

    X = np.arange(128, dtype=np.float32).reshape(64, 2)
    y = (X[:, 0] >= 64).astype(np.int64)
    parameters = dict(
        n_estimators=3,
        max_depth=3,
        max_bins=32,
        min_samples_leaf=2,
        min_child_weight=1.0,
        reg_lambda=1.0,
        max_features=1.0,
        sample_fraction=1.0,
        bootstrap=False,
        random_state=37,
    )
    cpu = RandomForestClassifier(device="cpu", **parameters).fit(X, y)
    mps = RandomForestClassifier(device="mps", **parameters).fit(X, y)

    np.testing.assert_allclose(
        mps.predict_proba(X),
        cpu.predict_proba(X),
        rtol=2e-4,
        atol=2e-4,
    )
    np.testing.assert_array_equal(mps.predict(X), cpu.predict(X))
    assert mps.device_ == "mps"
    assert mps.training_summary_["tree_devices"] == ["mps"] * parameters["n_estimators"]
    assert mps.training_summary_["scheduling"] == "serial"
    assert mps.n_estimators_ == parameters["n_estimators"]
