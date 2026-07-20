"""L1 and leaf-clipping regularization coverage for tree estimators."""

import numpy as np
import pytest

from mpsboost import MPSBoostRegressor, RandomForestRegressor, _native


def test_objective_l1_and_leaf_clipping_match_hand_values():
    """Native objective helpers should expose one L1 and clipping formula."""

    assert _native._node_score(-4.0, 2.0, 1.0, reg_alpha=1.0) == pytest.approx(3.0)
    assert _native._leaf_weight(-4.0, 2.0, 1.0, reg_alpha=1.0) == pytest.approx(1.0)
    assert _native._leaf_weight(-4.0, 2.0, 1.0, reg_alpha=1.0, max_delta_step=0.5) == 0.5
    assert _native._leaf_weight(0.5, 2.0, 1.0, reg_alpha=1.0) == 0.0
    assert _native._split_gain(0.0, 2.0, -4.0, 2.0, 0.0, 0.25, reg_alpha=1.0) == 0.875


def test_native_tree_l1_can_suppress_small_leaf_updates():
    """Large enough L1 should shrink constant-leaf updates to exactly zero."""

    dataset = _native._quantize_dense(np.ones((3, 1), dtype=np.float32), max_bins=4)
    tree = _native._train_single_tree_cpu(
        dataset,
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 0.0],
        max_depth=0,
        min_samples_leaf=1,
        min_child_weight=0.0,
        reg_lambda=0.0,
        reg_alpha=1.0,
    )

    assert tree.nodes[0]["leaf_value"] == 0.0
    assert tree.predict(dataset) == [0.0, 0.0, 0.0]


def test_native_tree_max_delta_step_clips_leaf_updates():
    """max_delta_step should clip native leaf values symmetrically."""

    dataset = _native._quantize_dense(np.ones((3, 1), dtype=np.float32), max_bins=4)
    tree = _native._train_single_tree_cpu(
        dataset,
        [10.0, 10.0, 10.0],
        [0.0, 0.0, 0.0],
        max_depth=0,
        min_samples_leaf=1,
        min_child_weight=0.0,
        reg_lambda=0.0,
        max_delta_step=2.0,
    )

    assert tree.nodes[0]["leaf_value"] == 2.0
    assert tree.predict(dataset) == [2.0, 2.0, 2.0]


def test_estimator_regularization_parameters_validate_and_train():
    """Public estimator parameters should validate and reach native training."""

    X = np.ones((3, 1), dtype=np.float32)
    y = np.asarray([0.0, 0.0, 0.5], dtype=np.float32)
    model = MPSBoostRegressor(
        n_estimators=1,
        max_depth=0,
        min_samples_leaf=1,
        min_child_weight=0.0,
        reg_lambda=0.0,
        reg_alpha=1.0,
        max_delta_step=0.25,
        device="cpu",
    ).fit(X, y)

    assert model.training_summary_["reg_alpha"] == 1.0
    assert model.training_summary_["max_delta_step"] == 0.25
    with pytest.raises(ValueError, match="reg_alpha"):
        MPSBoostRegressor(reg_alpha=-1.0, device="cpu").fit(X, y)
    with pytest.raises(ValueError, match="max_delta_step"):
        MPSBoostRegressor(max_delta_step=-1.0, device="cpu").fit(X, y)


def test_forest_forwards_regularization_to_each_tree():
    """Forest tree factories should not drop L1 or leaf-clipping controls."""

    X = np.ones((4, 1), dtype=np.float32)
    y = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    model = RandomForestRegressor(
        n_estimators=2,
        max_depth=0,
        min_samples_leaf=1,
        min_child_weight=0.0,
        sample_fraction=1.0,
        bootstrap=False,
        reg_alpha=0.5,
        max_delta_step=0.25,
        random_state=79,
        device="cpu",
    ).fit(X, y)

    assert model.training_summary_["reg_alpha"] == 0.5
    assert model.training_summary_["max_delta_step"] == 0.25
    for tree in model.estimators_:
        assert tree.training_summary_["reg_alpha"] == 0.5
        assert tree.training_summary_["max_delta_step"] == 0.25
