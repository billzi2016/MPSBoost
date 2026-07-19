"""MPSBoost 真实 CPU/MPS histogram benchmark runner。

runner 计量量化、CPU oracle、GPU 端到端调用和同步 command 分阶段时间，输出原始 JSON。
它不伪造缺失设备、不丢弃慢结果，也不参与测试或运行时包。
"""

from __future__ import annotations

import argparse
import json
import platform
from statistics import median
from time import perf_counter

import numpy as np

import mpsboost as mb
from mpsboost import _native
from mpsboost.diagnostics import _metallib_path

from datasets import histogram_scenarios, regressor_scenarios


def _elapsed(callable_):
    """执行一次真实调用并返回结果与端到端 wall time。"""

    started = perf_counter()
    result = callable_()
    return result, perf_counter() - started


def run(repeats: int) -> dict:
    """运行全部预登记场景并返回可序列化原始结果。

    每个场景先做一次明确记录的 warm-up，再保存全部重复值及中位数。GPU wall time包含
    buffer 创建、复制、command 和同步；阶段时间覆盖对应 command 的提交与等待。
    """

    if repeats < 3:
        raise ValueError("repeats 必须至少为 3，避免单次偶然值主导结论")
    if not mb.is_available():
        raise RuntimeError("真实 MPS 设备不可用，benchmark 不允许 CPU mock")
    report = {
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mpsboost": mb.__version__,
            "backend": mb.system_info(),
        },
        "repeats": repeats,
        "scenarios": [],
        "regressor_scenarios": [],
    }
    with _metallib_path() as path:
        backend = _native._MpsBackend(path)
        for scenario in histogram_scenarios():
            quantized, quantization_seconds = _elapsed(
                lambda: _native._quantize_dense(scenario.X, 256)
            )
            labels = scenario.y.tolist()
            predictions = np.zeros_like(scenario.y).tolist()
            rows = list(range(scenario.X.shape[0]))
            _native._cpu_histograms(quantized, labels, predictions, rows)
            backend.histograms(quantized, labels, predictions, rows)
            cpu_times = []
            gpu_wall_times = []
            gpu_encode_times = []
            gpu_command_times = []
            pool_reuse_counts = []
            pool_allocation_counts = []
            for _ in range(repeats):
                _, cpu_elapsed = _elapsed(
                    lambda: _native._cpu_histograms(
                        quantized, labels, predictions, rows
                    )
                )
                gpu_result, gpu_elapsed = _elapsed(
                    lambda: backend.histograms(quantized, labels, predictions, rows)
                )
                cpu_times.append(cpu_elapsed)
                gpu_wall_times.append(gpu_elapsed)
                gpu_encode_times.append(gpu_result["encode_seconds"])
                gpu_command_times.append(gpu_result["command_seconds"])
                timing = backend.last_timing
                pool_reuse_counts.append(timing["pooled_buffer_reuse_count"])
                pool_allocation_counts.append(timing["pooled_buffer_allocation_count"])
            cpu_median = median(cpu_times)
            gpu_median = median(gpu_wall_times)
            report["scenarios"].append(
                {
                    "name": scenario.name,
                    "rows": scenario.X.shape[0],
                    "features": scenario.X.shape[1],
                    "max_bins": 256,
                    "quantization_seconds": quantization_seconds,
                    "cpu_wall_seconds": cpu_times,
                    "gpu_wall_seconds": gpu_wall_times,
                    "gpu_encode_seconds": gpu_encode_times,
                    "gpu_command_seconds": gpu_command_times,
                    "pooled_buffer_reuse_count": pool_reuse_counts[-1],
                    "pooled_buffer_allocation_count": pool_allocation_counts[-1],
                    "cpu_median_seconds": cpu_median,
                    "gpu_median_seconds": gpu_median,
                    "wall_speedup": cpu_median / gpu_median,
                }
            )
    for scenario in regressor_scenarios():
        parameters = dict(
            n_estimators=8,
            learning_rate=0.2,
            max_depth=4,
            max_bins=256,
            min_samples_leaf=16,
            min_child_weight=1.0,
            reg_lambda=1.0,
        )
        mb.GradientBoostingRegressor(device="cpu", **parameters).fit(scenario.X, scenario.y)
        mb.GradientBoostingRegressor(device="mps", **parameters).fit(scenario.X, scenario.y)
        cpu_times = []
        mps_times = []
        max_prediction_differences = []
        for _ in range(repeats):
            cpu_model, cpu_elapsed = _elapsed(
                lambda: mb.GradientBoostingRegressor(device="cpu", **parameters).fit(
                    scenario.X, scenario.y
                )
            )
            mps_model, mps_elapsed = _elapsed(
                lambda: mb.GradientBoostingRegressor(device="mps", **parameters).fit(
                    scenario.X, scenario.y
                )
            )
            cpu_times.append(cpu_elapsed)
            mps_times.append(mps_elapsed)
            max_prediction_differences.append(
                float(
                    np.max(
                        np.abs(
                            cpu_model.predict(scenario.X)
                            - mps_model.predict(scenario.X)
                        )
                    )
                )
            )
        cpu_median = median(cpu_times)
        mps_median = median(mps_times)
        report["regressor_scenarios"].append(
            {
                "name": scenario.name,
                "rows": scenario.X.shape[0],
                "features": scenario.X.shape[1],
                "parameters": parameters,
                "cpu_fit_seconds": cpu_times,
                "mps_fit_seconds": mps_times,
                "cpu_median_seconds": cpu_median,
                "mps_median_seconds": mps_median,
                "wall_speedup": cpu_median / mps_median,
                "max_prediction_difference": max(max_prediction_differences),
            }
        )
    return report


def main() -> None:
    """解析 CLI 并把完整原始报告写到标准输出。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.repeats), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
