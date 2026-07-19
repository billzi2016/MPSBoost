"""MPSBoost 可复现 histogram benchmark 数据集定义。

本模块只生成固定种子的合成稠密数据与标签，不运行算法、不下载网络数据。所有规模和
分布显式命名，报告可以准确复现实验而不是挑选未记录的最快样本。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BenchmarkDataset:
    """一个包含名称、输入和标签的不可变 benchmark 场景。"""

    name: str
    X: NDArray[np.float32]
    y: NDArray[np.float64]


def histogram_scenarios() -> tuple[BenchmarkDataset, ...]:
    """返回预登记 Small/Medium/Large/Wide 合成场景。

    固定 RNG 和生成公式保证跨运行输入一致；偏斜列专门施压热点 bin，不能只测近似均匀
    分布后宣称 histogram 普遍加速。
    """

    definitions = (
        ("small", 4_096, 16),
        ("medium", 32_768, 32),
        ("large", 131_072, 64),
        ("wide", 32_768, 256),
    )
    scenarios = []
    for offset, (name, rows, features) in enumerate(definitions):
        generator = np.random.default_rng(20260719 + offset)
        matrix = generator.normal(size=(rows, features)).astype(np.float32)
        matrix[:, 0] = (generator.random(rows) > 0.98).astype(np.float32)
        labels = (
            matrix[:, 0].astype(np.float64) * 2.0
            + matrix[:, 1].astype(np.float64) * -0.5
        )
        scenarios.append(BenchmarkDataset(name, matrix, labels))
    return tuple(scenarios)


def regressor_scenarios() -> tuple[BenchmarkDataset, ...]:
    """返回预登记端到端 GBDT 训练场景。

    规模选择避免 benchmark 退化成纯启动开销，同时不会把常规验证拖成不可接受的长任务。
    标签包含非线性和阈值项，确保树模型需要真实 split，而不是只学习均值。
    """

    definitions = (
        ("gbdt-medium", 16_384, 32),
        ("gbdt-wide", 16_384, 128),
        ("gbdt-large-wide", 32_768, 256),
    )
    scenarios = []
    for offset, (name, rows, features) in enumerate(definitions):
        generator = np.random.default_rng(20260719 + 100 + offset)
        matrix = generator.normal(size=(rows, features)).astype(np.float32)
        matrix[:, 0] = (generator.random(rows) > 0.95).astype(np.float32)
        labels = (
            1.7 * matrix[:, 0].astype(np.float64)
            - 0.8 * matrix[:, 1].astype(np.float64)
            + np.where(matrix[:, 2] > 0.25, 2.0, -1.0).astype(np.float64)
        )
        scenarios.append(BenchmarkDataset(name, matrix, labels))
    return tuple(scenarios)
