"""真实 MPS split scan、partition 和 L1 buffer pool 测试。

本文件只验证内部 GPU 热路径能力，不暴露公共 API。期望值来自 CPU oracle 的 histogram
和冻结公式；测试不得用 mock command 或 host 伪造设备结果。
"""

import numpy as np
import pytest

from mpsboost import _native
from mpsboost.diagnostics import _metallib_path

pytestmark = pytest.mark.gpu


def _best_candidates_from_cpu(histograms, reg_lambda=1.0, gamma=0.0):
    """用 CPU histogram 和公开内部公式生成逐 feature 最佳候选。"""

    candidates = []
    for feature, bins in enumerate(histograms):
        parent_gradient = sum(item["gradient_sum"] for item in bins)
        parent_hessian = sum(item["hessian_sum"] for item in bins)
        parent_score = _native._node_score(parent_gradient, parent_hessian, reg_lambda)
        best = {"valid": False, "feature": feature}
        left_count = 0
        left_gradient = 0.0
        left_hessian = 0.0
        for threshold, item in enumerate(bins[:-1]):
            left_count += item["count"]
            left_gradient += item["gradient_sum"]
            left_hessian += item["hessian_sum"]
            right_count = sum(bin_["count"] for bin_ in bins) - left_count
            right_gradient = parent_gradient - left_gradient
            right_hessian = parent_hessian - left_hessian
            if (
                left_count < 2
                or right_count < 2
                or left_hessian <= 0
                or right_hessian <= 0
            ):
                continue
            gain = (
                0.5
                * (
                    _native._node_score(left_gradient, left_hessian, reg_lambda)
                    + _native._node_score(right_gradient, right_hessian, reg_lambda)
                    - parent_score
                )
                - gamma
            )
            if gain <= 0:
                continue
            if not best["valid"] or gain > best["gain"]:
                best = {
                    "valid": True,
                    "feature": feature,
                    "threshold_bin": threshold,
                    "left_count": left_count,
                    "right_count": right_count,
                    "left_gradient_sum": left_gradient,
                    "left_hessian_sum": left_hessian,
                    "right_gradient_sum": right_gradient,
                    "right_hessian_sum": right_hessian,
                    "gain": gain,
                }
        candidates.append(best)
    return candidates


def test_split_scan_candidates_match_cpu_histogram_scan():
    """GPU split scan 必须逐 feature 对照 CPU histogram 前缀扫描。"""

    rows = 257
    X = np.column_stack(
        (
            np.arange(rows, dtype=np.float32) % 17,
            np.linspace(-3.0, 3.0, rows, dtype=np.float32),
            np.where(np.arange(rows) < 180, 1.0, 9.0).astype(np.float32),
        )
    )
    labels = np.where(X[:, 0] < 8, -2.0, 3.0).astype(np.float64).tolist()
    predictions = np.zeros(rows, dtype=np.float64).tolist()
    selected = list(range(1, rows, 3))
    dataset = _native._quantize_dense(X, 32)
    expected_histograms = _native._cpu_histograms(
        dataset, labels, predictions, selected
    )
    expected = _best_candidates_from_cpu(expected_histograms)
    with _metallib_path() as path:
        actual = _native._MpsBackend(path).split_candidates(
            dataset,
            labels,
            predictions,
            selected,
            min_samples_leaf=2,
            min_child_weight=0.0,
            reg_lambda=1.0,
            gamma=0.0,
        )

    for actual_item, expected_item in zip(actual, expected, strict=True):
        assert actual_item["valid"] == expected_item["valid"]
        if not expected_item["valid"]:
            continue
        assert actual_item["feature"] == expected_item["feature"]
        assert actual_item["threshold_bin"] == expected_item["threshold_bin"]
        assert actual_item["left_count"] == expected_item["left_count"]
        assert actual_item["right_count"] == expected_item["right_count"]
        assert actual_item["gain"] == pytest.approx(
            expected_item["gain"], rel=2e-5, abs=2e-5
        )


def test_partition_rows_matches_cpu_bin_rule_and_reuses_buffers():
    """GPU partition 必须满足 bin <= threshold 左分支规则，并复用临时 buffer。"""

    rows = 513
    X = np.column_stack(
        (
            np.arange(rows, dtype=np.float32) % 23,
            np.linspace(-1.0, 1.0, rows, dtype=np.float32),
        )
    )
    dataset = _native._quantize_dense(X, 32)
    selected = list(range(0, rows, 2))
    feature = 0
    threshold = 10
    expected_left = [row for row in selected if dataset.bins[feature][row] <= threshold]
    expected_right = [row for row in selected if dataset.bins[feature][row] > threshold]

    with _metallib_path() as path:
        backend = _native._MpsBackend(path)
        left, right = backend.partition_rows(dataset, selected, feature, threshold)
        backend.partition_rows(dataset, selected, feature, threshold)
        timing = backend.last_timing

    assert left == expected_left
    assert right == expected_right
    assert timing["pooled_buffer_reuse_count"] >= 2
