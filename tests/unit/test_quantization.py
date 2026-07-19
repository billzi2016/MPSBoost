"""确定性分箱、所有权和序列化的真实单元测试。

测试通过内部 native 入口直接执行唯一 C++ 分箱实现，不使用 Python 重写期望算法。
手算样例给出独立期望；随机/重复执行用于验证确定性和生命周期。
"""

import gc

import numpy as np
import pytest

from mpsboost import _native


def test_hand_computed_boundaries_and_lower_bound_semantics():
    """等于边界的值必须进入左 bin，且内部布局按特征返回。"""

    matrix = np.array([[1.0, 10.0], [2.0, 10.0], [3.0, 20.0], [4.0, 20.0]], dtype=np.float32)
    dataset = _native._quantize_dense(matrix, max_bins=2)
    assert dataset.boundaries == [[2.0], [10.0]]
    assert dataset.bins == [[0, 0, 1, 1], [0, 0, 1, 1]]
    assert dataset.bin_width == 8


def test_constant_and_duplicate_values_do_not_create_empty_right_bin():
    """常量特征只有一个 bin，重复候选边界只保留一次。"""

    matrix = np.array([[5.0, 1.0], [5.0, 1.0], [5.0, 2.0], [5.0, 2.0]], dtype=np.float64)
    dataset = _native._quantize_dense(matrix, max_bins=16)
    assert dataset.boundaries[0] == []
    assert dataset.bins[0] == [0, 0, 0, 0]
    assert dataset.boundaries[1] == [1.0]


def test_extreme_skew_is_deterministic():
    """热点重复值不会因排序实现或重复边界产生不稳定结果。"""

    matrix = np.concatenate(
        [np.zeros((999, 1), dtype=np.float32), np.ones((1, 1), dtype=np.float32)]
    )
    first = _native._quantize_dense(matrix, max_bins=256)
    second = _native._quantize_dense(matrix, max_bins=256)
    assert first.serialize() == second.serialize()
    assert first.boundaries == [[0.0]]
    assert first.bins[0].count(0) == 999
    assert first.bins[0].count(1) == 1


@pytest.mark.parametrize("max_bins, expected_width", [(256, 8), (257, 16), (65536, 16)])
def test_storage_width_follows_global_max_bins(max_bins, expected_width):
    """存储宽度只由冻结的 max_bins 规则决定，不能随数据内容漂移。"""

    dataset = _native._quantize_dense(np.arange(8, dtype=np.float32).reshape(4, 2), max_bins)
    assert dataset.bin_width == expected_width


def test_non_contiguous_positive_stride_is_read_without_full_copy():
    """正 stride 非连续 view 应直接读取，量化结果独立拥有内存。"""

    source = np.arange(24, dtype=np.float32).reshape(4, 6)
    view = source[:, ::2]
    dataset = _native._quantize_dense(view, max_bins=4)
    assert dataset.source_contiguous is False
    assert dataset.source_was_copied is False
    expected_bins = dataset.bins
    del view
    del source
    gc.collect()
    assert dataset.bins == expected_bins


@pytest.mark.parametrize(
    "matrix, message",
    [
        (np.empty((0, 2), dtype=np.float32), "至少包含一行"),
        (np.array([[np.nan]], dtype=np.float32), "NaN 或 Inf"),
        (np.array([[np.inf]], dtype=np.float64), "NaN 或 Inf"),
        (np.array([[np.finfo(np.float64).max]], dtype=np.float64), "float32 表示范围"),
    ],
)
def test_invalid_values_are_rejected_before_output(matrix, message):
    """非法形状和值必须明确失败，不生成部分数据集。"""

    with pytest.raises((ValueError, _native.DataError), match=message):
        _native._quantize_dense(matrix, max_bins=4)


def test_invalid_dtype_rank_stride_and_bin_count_are_rejected():
    """dtype、维度、负 stride 和 max_bins 违反契约时应早失败。"""

    with pytest.raises(TypeError, match="float32 或 float64"):
        _native._quantize_dense(np.array([[1]], dtype=np.int32), 4)
    with pytest.raises(ValueError, match="二维"):
        _native._quantize_dense(np.array([1.0], dtype=np.float32), 4)
    with pytest.raises(ValueError, match="负 stride"):
        _native._quantize_dense(np.arange(4, dtype=np.float32)[::-1].reshape(2, 2), 4)
    with pytest.raises(ValueError, match="max_bins"):
        _native._quantize_dense(np.ones((1, 1), dtype=np.float32), 1)


def test_shape_stride_and_offset_overflow_are_rejected_without_allocation():
    """超大元数据必须在读取指针或分配内存前被检查，不能依赖实际 OOM。"""

    maximum = 2**64 - 1
    with pytest.raises(ValueError, match="row stride.*溢出"):
        _native._validate_dense_view(maximum, 2, 8, 4, 4, 256)

    with pytest.raises(ValueError, match="元素数量.*溢出"):
        _native._validate_dense_view(2**33, 2**32 - 1, 4, 4, 4, 256)

    with pytest.raises(ValueError, match="最大 offset.*溢出|末端 offset.*溢出"):
        _native._validate_dense_view(2, 2, maximum - 3, 4, 4, 256)


def test_serialization_round_trip_and_corruption_rejection():
    """稳定字节表示必须 round-trip，截断和尾部垃圾必须拒绝。"""

    matrix = np.array([[1.0, 9.0], [2.0, 8.0], [3.0, 7.0]], dtype=np.float32)
    original = _native._quantize_dense(matrix, max_bins=257)
    serialized = original.serialize()
    restored = _native._deserialize_binned(serialized)
    assert restored.serialize() == serialized
    assert restored.boundaries == original.boundaries
    assert restored.bins == original.bins
    with pytest.raises(ValueError, match="截断"):
        _native._deserialize_binned(serialized[:-1])
    with pytest.raises(ValueError, match="尾部字节"):
        _native._deserialize_binned(serialized + b"x")
