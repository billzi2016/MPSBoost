"""Monotonic-constraint coverage for native tree estimators."""

import numpy as np
import pytest

from mpsboost import MPSBoostRegressor, RandomForestRegressor, _native


def _dataset(values):
    """Quantize a one-feature matrix for direct native tree tests."""

    return _native._quantize_dense(np.asarray(values, dtype=np.float32), max_bins=8)


def test_native_positive_constraint_blocks_decreasing_root_split():
    """A positive constraint must reject splits whose left leaf exceeds the right leaf."""

    dataset = _dataset([[0.0], [1.0], [2.0], [3.0]])
    labels = [3.0, 2.0, 1.0, 0.0]
    constrained = _native._train_single_tree_cpu(
        dataset,
        labels,
        [0.0] * 4,
        max_depth=1,
        reg_lambda=0.0,
        monotonic_constraints=[1],
    )
    decreasing = _native._train_single_tree_cpu(
        dataset,
        labels,
        [0.0] * 4,
        max_depth=1,
        reg_lambda=0.0,
        monotonic_constraints=[-1],
    )

    assert len(constrained.nodes) == 1
    assert len(decreasing.nodes) == 3
    assert decreasing.predict(dataset)[0] > decreasing.predict(dataset)[-1]


def test_regressor_positive_constraint_keeps_predictions_nondecreasing():
    """Boosting should preserve nondecreasing predictions when every tree is constrained."""

    X = np.asarray([[float(value)] for value in range(8)], dtype=np.float32)
    y = np.asarray([0.0, 0.0, 1.0, 1.0, 3.0, 3.0, 4.0, 4.0], dtype=np.float32)
    model = MPSBoostRegressor(
        n_estimators=3,
        learning_rate=0.5,
        max_depth=2,
        min_samples_leaf=1,
        min_child_weight=0.0,
        monotonic_constraints=[1],
        device="cpu",
    ).fit(X, y)

    predictions = model.predict(X)

    assert model.training_summary_["monotonic_constraints"] == [1]
    assert np.all(np.diff(predictions) >= -1e-6)


def test_monotonic_constraints_validate_length_and_values():
    """Invalid constraint vectors should fail before native training finishes."""

    X = np.ones((3, 2), dtype=np.float32)
    y = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="length"):
        MPSBoostRegressor(monotonic_constraints=[1], device="cpu").fit(X, y)
    with pytest.raises(ValueError, match="-1, 0, or 1"):
        MPSBoostRegressor(monotonic_constraints=[2, 0], device="cpu").fit(X, y)
    with pytest.raises(TypeError, match="integers"):
        MPSBoostRegressor(monotonic_constraints=[1.0, 0], device="cpu").fit(X, y)


def test_forest_slices_monotonic_constraints_to_feature_subsets():
    """Feature-subsampled trees should receive local constraints for sampled columns."""

    X = np.asarray(
        [[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]],
        dtype=np.float32,
    )
    y = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    model = RandomForestRegressor(
        n_estimators=3,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        max_features=0.5,
        sample_fraction=1.0,
        bootstrap=False,
        monotonic_constraints=[1, -1],
        random_state=53,
        device="cpu",
    ).fit(X, y)

    assert model.training_summary_["monotonic_constraints"] == [1, -1]
    assert all(tree.training_summary_["monotonic_constraints"] in ([1], [-1]) for tree in model.estimators_)
