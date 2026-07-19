"""Leaf-wise CPU tree growth tests.

These tests cover the controlled LightGBM-like growth strategy without
duplicating the native split-scoring implementation in Python.
"""

import numpy as np
import pytest

from mpsboost import _native


def _train(matrix, labels, *, max_depth, **parameters):
    """Quantize dense input and train one native CPU tree."""

    dataset = _native._quantize_dense(np.asarray(matrix, dtype=np.float32), max_bins=16)
    predictions = [0.0] * len(labels)
    tree = _native._train_single_tree_cpu(
        dataset, labels, predictions, max_depth=max_depth, reg_lambda=0.0, **parameters
    )
    return dataset, tree


def test_leaf_wise_growth_expands_best_gain_leaf_first():
    """Leaf-wise growth should expand the current best leaf before shallower siblings."""

    matrix = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]
    labels = [0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 11.0, 12.0]
    dataset, tree = _train(
        matrix,
        labels,
        max_depth=3,
        growth_strategy="leaf_wise",
        max_leaves=3,
    )

    nodes = tree.nodes
    assert len(nodes) == 5
    assert nodes[0]["threshold_bin"] == 2
    assert nodes[2]["is_leaf"] is False
    assert nodes[2]["threshold_bin"] == 5
    assert tree.predict(dataset) == pytest.approx(
        [0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 11.5, 11.5]
    )


def test_leaf_wise_growth_respects_max_leaves_limit():
    """max_leaves should cap leaf-wise growth independently from max_depth."""

    _, tree = _train(
        [[float(value)] for value in range(8)],
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        max_depth=4,
        growth_strategy="leaf_wise",
        max_leaves=2,
    )

    assert len(tree.nodes) == 3
    assert sum(1 for node in tree.nodes if node["is_leaf"]) == 2


def test_leaf_wise_growth_respects_active_leaf_queue_limit():
    """The active leaf queue limit should fail before unchecked leaf-wise growth."""

    dataset = _native._quantize_dense(
        np.asarray([[float(value)] for value in range(8)], dtype=np.float32),
        max_bins=8,
    )
    with pytest.raises(_native.TrainingError, match="max_active_leaves"):
        _native._train_single_tree_cpu(
            dataset,
            [0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 11.0, 12.0],
            [0.0] * 8,
            max_depth=3,
            min_samples_leaf=1,
            min_child_weight=0.0,
            reg_lambda=0.0,
            growth_strategy="leaf_wise",
            max_leaves=4,
            max_active_leaves=2,
        )


def test_min_gain_to_split_blocks_small_positive_gain():
    """min_gain_to_split should reject otherwise valid low-gain split candidates."""

    dataset, unrestricted = _train(
        [[0.0], [1.0], [2.0], [3.0]],
        [0.0, 0.0, 1.0, 1.0],
        max_depth=1,
    )
    _, blocked = _train(
        [[0.0], [1.0], [2.0], [3.0]],
        [0.0, 0.0, 1.0, 1.0],
        max_depth=1,
        min_gain_to_split=2.0,
    )

    assert len(unrestricted.nodes) == 3
    assert len(blocked.nodes) == 1
    assert blocked.predict(dataset) == [0.5, 0.5, 0.5, 0.5]
