"""Native missing-value split and default-direction tests."""

import numpy as np

from mpsboost import MPSBoostRegressor
from mpsboost import _native


def _train(matrix, labels, *, max_depth=1):
    dataset = _native._quantize_dense(np.asarray(matrix, dtype=np.float32), max_bins=4)
    tree = _native._train_single_tree_cpu(
        dataset,
        labels,
        [0.0] * len(labels),
        max_depth=max_depth,
        min_samples_leaf=1,
        min_child_weight=0.0,
        reg_lambda=0.0,
    )
    return dataset, tree


def test_missing_values_choose_and_store_default_direction():
    """Split search should choose the better missing default direction."""

    dataset, tree = _train(
        [[0.0], [1.0], [2.0], [np.nan]],
        [0.0, 0.0, 10.0, 10.0],
    )
    root = tree.nodes[0]

    assert root["is_leaf"] is False
    assert root["default_left"] is False
    assert tree.predict(dataset) == [0.0, 0.0, 10.0, 10.0]


def test_prediction_nan_uses_stored_default_direction():
    """Prediction-time NaN should follow the model's saved default direction."""

    train_dataset, tree = _train(
        [[0.0], [1.0], [2.0], [np.nan]],
        [0.0, 0.0, 10.0, 10.0],
    )

    assert train_dataset.missing == [[False, False, False, True]]
    assert tree.nodes[0]["default_left"] is False


def test_estimator_prediction_nan_uses_frozen_training_schema():
    """Estimator prediction should transform NaN with the fitted schema."""

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

    assert model.predict(np.asarray([[np.nan], [0.0], [2.0]], dtype=np.float32)).tolist() == [
        10.0,
        0.0,
        10.0,
    ]
