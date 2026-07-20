"""Real unit tests for deterministic binning, ownership, and serialization.

Tests execute the sole C++ binning implementation through internal native entries
without rewriting the expected algorithm in Python. Hand-calculated cases provide
independent expectations; random/repeated runs validate determinism and lifetime.
"""

import gc

import numpy as np
import pytest

from mpsboost import _native


def test_hand_computed_boundaries_and_lower_bound_semantics():
    """Values equal to a boundary must enter the left bin in feature-major layout."""

    matrix = np.array([[1.0, 10.0], [2.0, 10.0], [3.0, 20.0], [4.0, 20.0]], dtype=np.float32)
    dataset = _native._quantize_dense(matrix, max_bins=2)
    assert dataset.boundaries == [[2.0], [10.0]]
    assert dataset.bins == [[0, 0, 1, 1], [0, 0, 1, 1]]
    assert dataset.bin_width == 8


def test_constant_and_duplicate_values_do_not_create_empty_right_bin():
    """Constant features have one bin and duplicate candidate boundaries occur once."""

    matrix = np.array([[5.0, 1.0], [5.0, 1.0], [5.0, 2.0], [5.0, 2.0]], dtype=np.float64)
    dataset = _native._quantize_dense(matrix, max_bins=16)
    assert dataset.boundaries[0] == []
    assert dataset.bins[0] == [0, 0, 0, 0]
    assert dataset.boundaries[1] == [1.0]


def test_nan_values_are_recorded_as_missing_without_changing_boundaries():
    """NaN values should be tracked in a missing mask and excluded from quantiles."""

    matrix = np.array(
        [[1.0, np.nan], [2.0, 10.0], [np.nan, 20.0], [4.0, np.nan]],
        dtype=np.float32,
    )
    dataset = _native._quantize_dense(matrix, max_bins=3)
    assert dataset.boundaries[0] == [1.0, 2.0]
    assert dataset.boundaries[1] == [10.0]
    assert dataset.missing == [[False, False, True, False], [True, False, False, True]]
    assert dataset.bins[0][2] == 0
    assert dataset.bins[1][0] == 0

    restored = _native._deserialize_binned(dataset.serialize())
    assert restored.boundaries == dataset.boundaries
    assert restored.bins == dataset.bins
    assert restored.missing == dataset.missing


def test_extreme_skew_is_deterministic():
    """Frequent repeated values must not become unstable from sorting or duplicate boundaries."""

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
    """Storage width depends only on frozen max_bins rules, never data contents."""

    dataset = _native._quantize_dense(np.arange(8, dtype=np.float32).reshape(4, 2), max_bins)
    assert dataset.bin_width == expected_width


def test_non_contiguous_positive_stride_is_read_without_full_copy():
    """Positive-stride noncontiguous views are read directly and quantized output owns memory."""

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
        (np.empty((0, 2), dtype=np.float32), "at least one row"),
        (np.array([[np.inf]], dtype=np.float64), "Inf"),
        (np.array([[np.finfo(np.float64).max]], dtype=np.float64), "finite float32 range"),
    ],
)
def test_invalid_values_are_rejected_before_output(matrix, message):
    """Invalid shapes and values must fail explicitly without partial datasets."""

    with pytest.raises((ValueError, _native.DataError), match=message):
        _native._quantize_dense(matrix, max_bins=4)


def test_invalid_dtype_rank_stride_and_bin_count_are_rejected():
    """dtype, dimensions, negative strides, and max_bins contract violations fail early."""

    with pytest.raises(TypeError, match="float32 or float64"):
        _native._quantize_dense(np.array([[1]], dtype=np.int32), 4)
    with pytest.raises(ValueError, match="two-dimensional"):
        _native._quantize_dense(np.array([1.0], dtype=np.float32), 4)
    with pytest.raises(ValueError, match="negative strides"):
        _native._quantize_dense(np.arange(4, dtype=np.float32)[::-1].reshape(2, 2), 4)
    with pytest.raises(ValueError, match="max_bins"):
        _native._quantize_dense(np.ones((1, 1), dtype=np.float32), 1)


def test_shape_stride_and_offset_overflow_are_rejected_without_allocation():
    """Oversized metadata must be checked before pointer reads or allocation, not real OOM."""

    maximum = 2**64 - 1
    with pytest.raises(ValueError, match="row stride.*overflow"):
        _native._validate_dense_view(maximum, 2, 8, 4, 4, 256)

    with pytest.raises(ValueError, match="element count.*overflow"):
        _native._validate_dense_view(2**33, 2**32 - 1, 4, 4, 4, 256)

    with pytest.raises(ValueError, match="maximum offset.*overflow|final offset.*overflow"):
        _native._validate_dense_view(2, 2, maximum - 3, 4, 4, 256)


def test_serialization_round_trip_and_corruption_rejection():
    """Stable byte representation must round-trip and reject truncation and trailing garbage."""

    matrix = np.array([[1.0, 9.0], [2.0, 8.0], [3.0, 7.0]], dtype=np.float32)
    original = _native._quantize_dense(matrix, max_bins=257)
    serialized = original.serialize()
    restored = _native._deserialize_binned(serialized)
    assert restored.serialize() == serialized
    assert restored.boundaries == original.boundaries
    assert restored.bins == original.bins
    with pytest.raises(ValueError, match="truncated"):
        _native._deserialize_binned(serialized[:-1])
    with pytest.raises(ValueError, match="trailing bytes"):
        _native._deserialize_binned(serialized + b"x")
