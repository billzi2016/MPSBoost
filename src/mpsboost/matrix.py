"""MPSBoost Python 输入适配的唯一实现。

本模块只把稠密二维数值输入和一维标签整理成原生层支持的连续数组；完整尺寸、有限值、
stride 与溢出校验仍由 C++ 执行。这里不得实现分箱、目标函数或训练逻辑。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def as_dense_matrix(value: Any) -> NDArray[np.float32] | NDArray[np.float64]:
    """将用户输入转换为原生训练入口可借用的二维浮点数组。

    已经是 float32/float64 的正 stride 数组保持 dtype 和视图语义；其他实数 dtype 转为
    float32。对象、复数、布尔和非二维输入明确拒绝，不接受隐式稀疏展开。
    """

    if hasattr(value, "toarray") and not isinstance(value, np.ndarray):
        raise TypeError("当前版本不支持稀疏矩阵")
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError("X 必须是二维稠密数值数组")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("X 必须至少包含一行和一个特征")
    if array.dtype.kind not in "iuf":
        raise TypeError("X dtype 必须是实数数值类型")
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        array = array.astype(np.float32)
    if any(stride <= 0 for stride in array.strides):
        # 原生层拒绝负/零 stride；在用户 API 统一复制成 C 连续数组，避免相同输入在 fit
        # 和 predict 表现不同。复制事实会由 training_summary_ 明确报告。
        array = np.ascontiguousarray(array)
    return array


def as_labels(value: Any, expected_rows: int) -> NDArray[np.float64]:
    """返回连续的一维 float64 标签并验证行数。

    参数错误发生在设备初始化前；有限值和极端范围由 native 目标函数再次验证，防止绕过
    Python 入口的调用者形成第二套语义。
    """

    labels = np.asarray(value)
    if labels.ndim != 1:
        raise ValueError("y 必须是一维数值数组")
    if labels.shape[0] != expected_rows:
        raise ValueError("X 与 y 的样本数量不一致")
    if labels.dtype.kind not in "iuf":
        raise TypeError("y dtype 必须是实数数值类型")
    return np.ascontiguousarray(labels, dtype=np.float64)
