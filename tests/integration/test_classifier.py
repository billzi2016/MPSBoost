"""End-to-end real MPS integration tests for the binary classifier.

These tests must run on a real Apple Silicon runner. They cover the Python estimator, dense input
adaptation, quantization, binary-logistic objective routing, MPS histogram/split training, and
classifier probability conversion without mock devices or silent CPU fallback.
"""

import numpy as np
import pytest

from mpsboost import MPSBoostClassifier

pytestmark = pytest.mark.gpu


def test_real_mps_classifier_matches_cpu_model_on_stable_dataset():
    """MPS and CPU classifier training should agree on a deterministic separable dataset."""

    X = np.arange(128, dtype=np.float32).reshape(64, 2)
    y = (X[:, 0] >= 64).astype(np.int64)
    parameters = dict(
        n_estimators=5,
        learning_rate=0.3,
        max_depth=3,
        max_bins=32,
        min_samples_leaf=2,
        min_child_weight=1.0,
        reg_lambda=1.0,
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
    assert mps.training_summary_["device"] == "mps"
    assert mps.training_summary_["device_decision"]["selected"] == "mps"
    assert mps.training_summary_["fit_seconds"] > 0.0
    assert mps.n_estimators_ == parameters["n_estimators"]
