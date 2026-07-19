"""真实 MPS gradient 与两阶段 histogram 正确性测试。

所有用例在真实默认 Metal 设备执行并逐元素对照 CPU oracle；禁止 mock command、替换
metallib 或只检查 kernel 是否启动。覆盖非整组长度、偏斜 bin 和两种紧凑存储宽度。
"""

import numpy as np
import pytest

from mpsboost import _native
from mpsboost.diagnostics import _metallib_path

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("length", [1, 7, 257, 1025])
def test_mps_gradients_match_cpu_oracle(length):
    """平方误差 kernel 必须覆盖非线程组整倍数并保持冻结语义。"""

    labels = np.linspace(-3.0, 4.0, length, dtype=np.float64).tolist()
    predictions = np.linspace(1.0, -2.0, length, dtype=np.float64).tolist()
    expected = _native._squared_error_gradients(labels, predictions)
    with _metallib_path() as path:
        actual = _native._MpsBackend(path).gradients(labels, predictions)
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=2e-6, atol=2e-6
    )


@pytest.mark.parametrize("max_bins", [16, 257])
def test_two_stage_histogram_matches_every_cpu_bin(max_bins):
    """uint8/uint16 数据的 count/G/H 必须逐 feature、逐 bin 对照 CPU。"""

    rows = 1025
    matrix = np.column_stack(
        (
            np.arange(rows, dtype=np.float32) % 13,
            np.where(np.arange(rows) < 1000, 0.0, 1.0).astype(np.float32),
            np.linspace(-1.0, 1.0, rows, dtype=np.float32),
        )
    )
    labels = np.sin(np.arange(rows, dtype=np.float64) / 17.0).tolist()
    predictions = np.zeros(rows, dtype=np.float64).tolist()
    selected = list(range(3, rows, 2))
    dataset = _native._quantize_dense(matrix, max_bins)
    expected = _native._cpu_histograms(dataset, labels, predictions, selected)
    with _metallib_path() as path:
        backend = _native._MpsBackend(path)
        baseline = backend.baseline_histograms(
            dataset, labels, predictions, selected
        )
        result = backend.histograms(dataset, labels, predictions, selected)
    actual = result["histograms"]
    assert len(actual) == len(expected)
    for baseline_feature, actual_feature, expected_feature in zip(
        baseline, actual, expected, strict=True
    ):
        for baseline_bin, actual_bin, expected_bin in zip(
            baseline_feature, actual_feature, expected_feature, strict=True
        ):
            assert baseline_bin["count"] == actual_bin["count"]
            assert baseline_bin["gradient_sum"] == pytest.approx(
                actual_bin["gradient_sum"], rel=2e-5, abs=2e-5
            )
            assert baseline_bin["hessian_sum"] == pytest.approx(
                actual_bin["hessian_sum"], rel=1e-6, abs=1e-6
            )
            assert actual_bin["count"] == expected_bin["count"]
            assert actual_bin["gradient_sum"] == pytest.approx(
                expected_bin["gradient_sum"], rel=2e-5, abs=2e-5
            )
            assert actual_bin["hessian_sum"] == pytest.approx(
                expected_bin["hessian_sum"], rel=1e-6, abs=1e-6
            )
    assert result["encode_seconds"] >= 0.0
    assert result["command_seconds"] >= 0.0
