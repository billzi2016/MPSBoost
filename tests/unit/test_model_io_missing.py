"""Missing-value model persistence coverage.

This focused file keeps the general model I/O suite small while still testing
that native model bytes preserve the tree-level default direction chosen for
missing values.
"""

import numpy as np

from mpsboost import MPSBoostRegressor


def test_missing_default_direction_round_trip(tmp_path):
    """Model files should preserve missing-value default directions."""

    X = np.asarray([[0.0], [1.0], [2.0], [np.nan]], dtype=np.float32)
    y = np.asarray([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    model = MPSBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        reg_lambda=0.0,
        device="cpu",
    ).fit(X, y)
    path = tmp_path / "missing.mb"
    model.save_model(path)

    restored = MPSBoostRegressor(device="cpu").load_model(path)

    assert restored.model_.trees[0].nodes[0]["default_left"] is False
    np.testing.assert_allclose(restored.predict(X), model.predict(X))
    np.testing.assert_allclose(
        restored.predict(np.asarray([[np.nan], [0.0], [2.0]], dtype=np.float32)),
        np.asarray([10.0, 0.0, 10.0], dtype=np.float32),
    )
