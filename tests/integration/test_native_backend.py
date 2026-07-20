"""Integration tests for the real MPS/Metal backend.

This file forbids mock devices and commands. Tests run on the real default Metal
device on Apple Silicon, validating the full path from Python and native bindings
through metallib and GPU buffers to synchronized readback.
"""

import pytest

import mpsboost as mb
from mpsboost.diagnostics import _run_vector_add_for_test

# This file covers one real-device pipeline. A module-level marker sends new cases
# to GPU jobs by default, preventing missed per-function marks and hardware coverage.
pytestmark = pytest.mark.gpu


def test_backend_reports_real_available_device():
    """A supported build must discover a real device and minimal non-sensitive capability."""

    assert mb.is_available() is True
    info = mb.system_info()
    assert info["backend"] == "mps"
    assert isinstance(info["device_name"], str)
    assert info["device_name"]
    assert info["has_unified_memory"] is True
    assert info["recommended_max_working_set_size"] > 0


@pytest.mark.parametrize("length", [1, 7, 257, 1025])
def test_real_gpu_vector_add_handles_partial_threadgroups(length):
    """Real GPU results must cover one element and non-threadgroup-multiple lengths."""

    left = [float(index) for index in range(length)]
    right = [float(index) * 0.5 for index in range(length)]
    actual = _run_vector_add_for_test(left, right)
    expected = [a + b for a, b in zip(left, right, strict=True)]
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_gpu_vector_add_rejects_mismatched_lengths():
    """Cross-language input-contract errors must fail before command submission."""

    with pytest.raises(mb._native.BackendError, match="lengths do not match"):
        _run_vector_add_for_test([1.0], [1.0, 2.0])
