"""CPU histogram oracle 与深度受限单树的真实手算测试。

本文件通过内部 native 入口运行同一 C++ 树生长和预测实现。期望树来自小数据手算，
不调用另一套参考训练器；重复运行用于冻结 feature/threshold tie-break 和节点顺序。
"""

import numpy as np
import pytest

from mpsboost import _native


def _train(matrix, labels, *, max_depth, **parameters):
    """量化并训练一棵内部 CPU oracle 树，集中测试装配而不复制算法。"""

    dataset = _native._quantize_dense(np.asarray(matrix, dtype=np.float32), max_bins=16)
    predictions = [0.0] * len(labels)
    tree = _native._train_single_tree_cpu(
        dataset, labels, predictions, max_depth=max_depth, reg_lambda=0.0, **parameters
    )
    return dataset, tree


def test_cpu_histogram_matches_hand_computed_count_gradient_and_hessian():
    """指定节点行集合的每个 bin 必须给出独立手算的 count/G/H。"""

    matrix = np.array(
        [[0.0, 10.0], [1.0, 10.0], [2.0, 20.0], [3.0, 20.0]],
        dtype=np.float32,
    )
    dataset = _native._quantize_dense(matrix, max_bins=16)
    histograms = _native._cpu_histograms(
        dataset,
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0, 0.0],
        [1, 2, 3],
    )
    assert histograms[0] == [
        {"count": 0, "gradient_sum": 0.0, "hessian_sum": 0.0},
        {"count": 1, "gradient_sum": -1.0, "hessian_sum": 1.0},
        {"count": 1, "gradient_sum": -2.0, "hessian_sum": 1.0},
        {"count": 1, "gradient_sum": -3.0, "hessian_sum": 1.0},
    ]
    assert histograms[1] == [
        {"count": 1, "gradient_sum": -1.0, "hessian_sum": 1.0},
        {"count": 2, "gradient_sum": -5.0, "hessian_sum": 2.0},
    ]


def test_hand_computed_root_split_leaf_values_and_prediction():
    """四行阶跃数据必须在 bin 1 切分，并产生手算叶值 0 和 2。"""

    dataset, tree = _train([[0.0], [1.0], [2.0], [3.0]], [0.0, 0.0, 2.0, 2.0], max_depth=1)
    nodes = tree.nodes
    assert len(nodes) == 3
    assert nodes[0] == {
        "is_leaf": False,
        "feature_index": 0,
        "threshold_bin": 1,
        "left_child": 1,
        "right_child": 2,
        "leaf_value": 0.0,
        "gain": 2.0,
        "default_left": True,
    }
    assert nodes[1]["is_leaf"] is True
    assert nodes[1]["leaf_value"] == 0.0
    assert nodes[2]["is_leaf"] is True
    assert nodes[2]["leaf_value"] == 2.0
    assert tree.predict(dataset) == [0.0, 0.0, 2.0, 2.0]


def test_level_wise_depth_two_tree_matches_every_hand_computed_node():
    """按层节点顺序、三个 split、四个叶值和逐样本预测必须全部一致。"""

    matrix = [[float(value)] for value in range(8)]
    labels = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    dataset, tree = _train(matrix, labels, max_depth=2)
    nodes = tree.nodes
    assert len(nodes) == 7
    assert [(node["feature_index"], node["threshold_bin"]) for node in nodes[:3]] == [
        (0, 3),
        (0, 1),
        (0, 5),
    ]
    assert [(node["left_child"], node["right_child"]) for node in nodes[:3]] == [
        (1, 2),
        (3, 4),
        (5, 6),
    ]
    assert [node["leaf_value"] for node in nodes[3:]] == [0.0, 1.0, 2.0, 3.0]
    assert tree.predict(dataset) == labels


def test_stable_tie_break_prefers_smaller_feature_then_threshold():
    """完全相同 feature gain 选 feature 0，同 feature 相同 gain 选较小 threshold。"""

    matrix = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    _, tree = _train(matrix, [0.0, 1.0, 1.0, 0.0], max_depth=1)
    root = tree.nodes[0]
    assert root["feature_index"] == 0
    assert root["threshold_bin"] == 0


def test_random_threshold_split_strategy_is_seeded_and_native():
    """Random-threshold trees must use native split selection with deterministic seeds."""

    dataset = _native._quantize_dense(
        np.asarray([[float(value)] for value in range(8)], dtype=np.float32),
        max_bins=8,
    )
    labels = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    first = _native._train_single_tree_cpu(
        dataset,
        labels,
        [0.0] * 8,
        max_depth=1,
        reg_lambda=0.0,
        split_strategy="random_threshold",
        random_seed=101,
    )
    second = _native._train_single_tree_cpu(
        dataset,
        labels,
        [0.0] * 8,
        max_depth=1,
        reg_lambda=0.0,
        split_strategy="random_threshold",
        random_seed=101,
    )
    different = _native._train_single_tree_cpu(
        dataset,
        labels,
        [0.0] * 8,
        max_depth=1,
        reg_lambda=0.0,
        split_strategy="random_threshold",
        random_seed=202,
    )

    assert first.nodes == second.nodes
    assert first.nodes[0]["threshold_bin"] != different.nodes[0]["threshold_bin"]


def test_depth_gamma_and_constant_feature_keep_valid_leaf():
    """禁止生长或无正增益时只保留根叶，叶值仍由全部样本统计计算。"""

    dataset, depth_zero = _train([[1.0], [1.0]], [2.0, 4.0], max_depth=0)
    assert len(depth_zero.nodes) == 1
    assert depth_zero.nodes[0]["leaf_value"] == 3.0
    assert depth_zero.predict(dataset) == [3.0, 3.0]

    _, blocked = _train(
        [[0.0], [1.0], [2.0], [3.0]],
        [0.0, 0.0, 2.0, 2.0],
        max_depth=2,
        gamma=3.0,
    )
    assert len(blocked.nodes) == 1
    assert blocked.nodes[0]["leaf_value"] == 1.0


def test_minimum_child_constraints_and_repeated_training_are_deterministic():
    """子节点约束必须参与候选过滤，同一输入重复训练节点字段完全一致。"""

    matrix = [[0.0], [1.0], [2.0], [3.0]]
    labels = [0.0, 0.0, 0.0, 10.0]
    dataset = _native._quantize_dense(np.asarray(matrix, dtype=np.float32), max_bins=4)
    arguments = (dataset, labels, [0.0] * 4)
    first = _native._train_single_tree_cpu(
        *arguments, max_depth=1, min_samples_leaf=2, min_child_weight=2.0, reg_lambda=0.0
    )
    second = _native._train_single_tree_cpu(
        *arguments, max_depth=1, min_samples_leaf=2, min_child_weight=2.0, reg_lambda=0.0
    )
    assert first.nodes == second.nodes
    assert first.nodes[0]["threshold_bin"] == 1
    assert first.predict(dataset) == second.predict(dataset)


def test_invalid_training_contract_and_prediction_shape_fail_explicitly():
    """长度、参数和预测特征数错误必须明确失败，不能导出或使用部分模型。"""

    one_feature = _native._quantize_dense(np.ones((2, 1), dtype=np.float32), 4)
    with pytest.raises(_native.TrainingError, match="长度不一致"):
        _native._train_single_tree_cpu(one_feature, [1.0], [0.0, 0.0], max_depth=1)
    with pytest.raises(_native.TrainingError, match="min_samples_leaf"):
        _native._train_single_tree_cpu(
            one_feature, [1.0, 2.0], [0.0, 0.0], max_depth=1, min_samples_leaf=0
        )
    with pytest.raises(_native.TrainingError, match="reg_lambda"):
        _native._train_single_tree_cpu(
            one_feature, [1.0, 2.0], [0.0, 0.0], max_depth=1, reg_lambda=-1.0
        )
    with pytest.raises(ValueError, match="growth strategy"):
        _native._train_single_tree_cpu(
            one_feature,
            [1.0, 2.0],
            [0.0, 0.0],
            max_depth=1,
            growth_strategy="unknown",
        )
    with pytest.raises(_native.TrainingError, match="max_leaves"):
        _native._train_single_tree_cpu(
            one_feature,
            [1.0, 2.0],
            [0.0, 0.0],
            max_depth=1,
            growth_strategy="leaf_wise",
            max_leaves=1,
        )
    with pytest.raises(_native.TrainingError, match="min_gain_to_split"):
        _native._train_single_tree_cpu(
            one_feature,
            [1.0, 2.0],
            [0.0, 0.0],
            max_depth=1,
            min_gain_to_split=-1.0,
        )

    tree = _native._train_single_tree_cpu(
        one_feature, [1.0, 2.0], [0.0, 0.0], max_depth=0, reg_lambda=0.0
    )
    two_features = _native._quantize_dense(np.ones((2, 2), dtype=np.float32), 4)
    with pytest.raises(_native.TrainingError, match="特征数量"):
        tree.predict(two_features)
