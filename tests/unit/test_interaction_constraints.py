"""Interaction-constraint coverage for native tree estimators."""

import numpy as np
import pytest

from mpsboost import MPSBoostRegressor, RandomForestRegressor, _native


def _train_tree(matrix, labels, constraints):
    """Train one direct native tree with interaction constraints."""

    dataset = _native._quantize_dense(np.asarray(matrix, dtype=np.float32), max_bins=8)
    tree = _native._train_single_tree_cpu(
        dataset,
        list(labels),
        [0.0] * len(labels),
        max_depth=2,
        min_samples_leaf=1,
        min_child_weight=0.0,
        reg_lambda=0.0,
        interaction_constraints=constraints,
    )
    return tree


def test_native_interaction_constraints_disallow_features_outside_groups():
    """A feature absent from every interaction group must never be selected."""

    tree = _train_tree(
        [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0]],
        [0.0, 1.0, 2.0, 3.0],
        [[0]],
    )

    assert all(node["is_leaf"] or node["feature_index"] == 0 for node in tree.nodes)


def test_native_interaction_constraints_allow_same_group_paths():
    """Features in one group may appear together along a root-to-leaf path."""

    tree = _train_tree(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        [0.0, 1.0, 1.0, 2.0],
        [[0, 1]],
    )

    split_features = [node["feature_index"] for node in tree.nodes if not node["is_leaf"]]
    assert set(split_features) <= {0, 1}
    assert len(split_features) >= 1


def test_regressor_validates_interaction_constraints():
    """Invalid interaction groups should fail before device-specific training."""

    X = np.ones((3, 2), dtype=np.float32)
    y = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="non-empty"):
        MPSBoostRegressor(interaction_constraints=[[]], device="cpu").fit(X, y)
    with pytest.raises(ValueError, match="out of range"):
        MPSBoostRegressor(interaction_constraints=[[2]], device="cpu").fit(X, y)
    with pytest.raises(ValueError, match="duplicates"):
        MPSBoostRegressor(interaction_constraints=[[0, 0]], device="cpu").fit(X, y)
    with pytest.raises(TypeError, match="integers"):
        MPSBoostRegressor(interaction_constraints=[["bad"]], device="cpu").fit(X, y)


def test_forest_maps_interaction_constraints_to_sampled_feature_subsets():
    """Feature-subsampled trees should receive local interaction group indices."""

    X = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    y = np.asarray([0.0, 1.0, 1.0, 2.0], dtype=np.float32)
    model = RandomForestRegressor(
        n_estimators=3,
        max_depth=2,
        min_samples_leaf=1,
        min_child_weight=0.0,
        max_features=0.67,
        sample_fraction=1.0,
        bootstrap=False,
        interaction_constraints=[[0, 1], [2]],
        random_state=67,
        device="cpu",
    ).fit(X, y)

    assert model.training_summary_["interaction_constraints"] == [[0, 1], [2]]
    for tree in model.estimators_:
        for group in tree.training_summary_["interaction_constraints"]:
            assert all(0 <= feature < tree.n_features_in_ for feature in group)
