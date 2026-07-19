"""End-to-end real MPS integration tests for ExtraTrees estimators.

These tests run only on a real Apple Silicon runner. They verify that ExtraTrees estimators drive
native random-threshold split training through the MPS backend and keep deterministic aggregate
predictions aligned with the CPU backend under identical sampling and seed settings.
"""

import numpy as np
import pytest

from mpsboost import ExtraTreesClassifier, ExtraTreesRegressor

pytestmark = pytest.mark.gpu


def test_real_mps_extra_trees_regressor_matches_cpu_model():
    """CPU and MPS ExtraTrees regressors should agree under deterministic full-data sampling."""

    X = np.arange(160, dtype=np.float32).reshape(80, 2)
    y = np.where(X[:, 0] < 80, -1.25, 2.5).astype(np.float64)
    parameters = dict(
        n_estimators=4,
        max_depth=3,
        max_bins=32,
        min_samples_leaf=2,
        min_child_weight=1.0,
        reg_lambda=1.0,
        max_features=1.0,
        sample_fraction=1.0,
        bootstrap=False,
        random_state=41,
    )
    cpu = ExtraTreesRegressor(device="cpu", **parameters).fit(X, y)
    mps = ExtraTreesRegressor(device="mps", **parameters).fit(X, y)

    np.testing.assert_allclose(mps.predict(X), cpu.predict(X), rtol=2e-4, atol=2e-4)
    assert mps.device_ == "mps"
    assert mps.n_estimators_ == parameters["n_estimators"]


def test_real_mps_extra_trees_classifier_matches_cpu_model():
    """CPU and MPS ExtraTrees classifiers should agree on probabilities and labels."""

    X = np.arange(160, dtype=np.float32).reshape(80, 2)
    y = (X[:, 0] >= 80).astype(np.int64)
    parameters = dict(
        n_estimators=4,
        max_depth=3,
        max_bins=32,
        min_samples_leaf=2,
        min_child_weight=1.0,
        reg_lambda=1.0,
        max_features=1.0,
        sample_fraction=1.0,
        bootstrap=False,
        random_state=43,
    )
    cpu = ExtraTreesClassifier(device="cpu", **parameters).fit(X, y)
    mps = ExtraTreesClassifier(device="mps", **parameters).fit(X, y)

    np.testing.assert_allclose(
        mps.predict_proba(X),
        cpu.predict_proba(X),
        rtol=2e-4,
        atol=2e-4,
    )
    np.testing.assert_array_equal(mps.predict(X), cpu.predict(X))
    assert mps.device_ == "mps"
    assert mps.n_estimators_ == parameters["n_estimators"]
