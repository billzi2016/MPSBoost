"""真实 MPS/Metal 后端集成测试。

本文件禁止 mock 设备或 command。测试必须在 Apple Silicon 的真实默认 Metal 设备上运行，
验证从 Python、原生绑定、metallib、GPU buffer 到同步读回的完整链路。
"""

import pytest

import mpsboost as mb
from mpsboost.diagnostics import _run_vector_add_for_test

# 整个文件覆盖同一条真实设备链路。使用文件级标记可以保证新增用例默认进入 GPU 作业，
# 避免维护者遗漏逐函数标记后，误在无设备环境中运行或漏掉真实硬件验证。
pytestmark = pytest.mark.gpu


def test_backend_reports_real_available_device():
    """受支持构建必须发现真实设备并返回最小非敏感能力。"""

    assert mb.is_available() is True
    info = mb.system_info()
    assert info["backend"] == "mps"
    assert isinstance(info["device_name"], str)
    assert info["device_name"]
    assert info["has_unified_memory"] is True
    assert info["recommended_max_working_set_size"] > 0


@pytest.mark.parametrize("length", [1, 7, 257, 1025])
def test_real_gpu_vector_add_handles_partial_threadgroups(length):
    """真实 GPU 结果必须覆盖单元素和非线程组整倍数长度。"""

    left = [float(index) for index in range(length)]
    right = [float(index) * 0.5 for index in range(length)]
    actual = _run_vector_add_for_test(left, right)
    expected = [a + b for a, b in zip(left, right, strict=True)]
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_gpu_vector_add_rejects_mismatched_lengths():
    """跨语言输入契约错误必须在提交 command 前明确失败。"""

    with pytest.raises(mb._native.BackendError, match="长度不一致"):
        _run_vector_add_for_test([1.0], [1.0, 2.0])
