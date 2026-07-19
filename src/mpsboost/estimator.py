"""MPSBoost sklearn 风格回归 estimator。

本模块负责构造参数、输入适配、并发 fit 防护和拟合状态的原子替换。分箱、boosting、
Metal 调度和模型格式均委托各自唯一 native 实现，不在 Python 复制算法。
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import _native
from .diagnostics import _metallib_path, is_available
from .matrix import as_dense_matrix, as_labels


class NotFittedError(RuntimeError):
    """在 estimator 尚无完整模型时调用拟合后能力。"""


class MPSBoostRegressor:
    """使用真实 MPS histogram 热路径训练平方误差 GBDT 回归模型。

    构造函数只保存参数，不初始化设备、创建缓存或分配训练内存。``device="cpu"`` 是显式
    oracle/诊断模式；``device="mps"`` 不可用时明确失败，绝不静默回退。
    """

    _PARAMETER_NAMES = (
        "n_estimators",
        "learning_rate",
        "max_depth",
        "max_bins",
        "min_child_weight",
        "min_samples_leaf",
        "reg_lambda",
        "random_state",
        "device",
        "verbosity",
    )

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        max_bins: int = 256,
        min_child_weight: float = 1.0,
        min_samples_leaf: int = 20,
        reg_lambda: float = 1.0,
        random_state: int | None = None,
        device: str = "mps",
        verbosity: int = 1,
    ) -> None:
        """保存 estimator 参数，不执行验证之外的昂贵副作用。"""

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_bins = max_bins
        self.min_child_weight = min_child_weight
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.device = device
        self.verbosity = verbosity
        self._fit_lock = Lock()

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """返回全部构造参数，兼容 estimator 工具的参数发现协议。

        ``deep`` 为接口兼容参数；本 estimator 当前没有嵌套 estimator，因此不会改变结果。
        """

        del deep
        return {name: getattr(self, name) for name in self._PARAMETER_NAMES}

    def set_params(self, **parameters: Any) -> "MPSBoostRegressor":
        """设置已知构造参数并返回自身；未知参数立即失败。

        参数发生变化时清除旧拟合状态，防止 ``get_params`` 与现有模型语义不一致。
        """

        unknown = sorted(set(parameters) - set(self._PARAMETER_NAMES))
        if unknown:
            raise ValueError(f"未知参数：{', '.join(unknown)}")
        for name, value in parameters.items():
            setattr(self, name, value)
        if parameters:
            self._clear_fitted_state()
        return self

    def fit(self, X: Any, y: Any) -> "MPSBoostRegressor":
        """训练完整模型，并仅在成功后原子替换拟合状态。

        同一实例不支持并发 fit；失败时清除半成品，调用方不会得到部分树。长时间 native
        训练释放 GIL，其他 Python 线程仍可运行。
        """

        if not self._fit_lock.acquire(blocking=False):
            raise RuntimeError("同一 estimator 不支持并发 fit")
        try:
            self._validate_parameters()
            matrix = as_dense_matrix(X)
            labels = as_labels(y, matrix.shape[0])
            parameters = _native._TrainingParameters(
                self.n_estimators,
                self.learning_rate,
                self.max_bins,
                self.max_depth,
                self.min_samples_leaf,
                self.min_child_weight,
                self.reg_lambda,
            )
            started = perf_counter()
            if self.device == "mps":
                if not is_available():
                    raise _native.BackendError(
                        "MPS 后端不可用；请在受支持的 Apple Silicon Mac 上运行"
                    )
                with _metallib_path() as metallib_path:
                    candidate = _native._train_regressor_mps(
                        matrix, labels, parameters, metallib_path
                    )
            else:
                candidate = _native._train_regressor_cpu(matrix, labels, parameters)
            elapsed = perf_counter() - started

            # 所有赋值集中在成功路径末端，使设备异常、OOM 或输入错误不会留下半模型。
            self.model_ = candidate
            self.n_features_in_ = matrix.shape[1]
            self.device_ = self.device
            self.n_estimators_ = candidate.tree_count
            self.training_summary_ = {
                "fit_seconds": elapsed,
                "input_contiguous": bool(matrix.flags.c_contiguous),
                "device": self.device,
                "n_estimators": candidate.tree_count,
            }
            return self
        except Exception:
            self._clear_fitted_state()
            raise
        finally:
            self._fit_lock.release()

    def predict(self, X: Any) -> NDArray[np.float32]:
        """使用训练期冻结的分箱规则返回一维 float32 预测。"""

        model = self._require_model()
        matrix = as_dense_matrix(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("预测特征数量与训练数据不一致")
        return np.asarray(model.predict(matrix), dtype=np.float32)

    def save_model(self, path: str | Path) -> None:
        """以版本化格式原子保存模型，不写入训练数据、缓存或设备标识。"""

        self._require_model().save(str(path))

    def load_model(self, path: str | Path) -> "MPSBoostRegressor":
        """加载并验证模型，仅在完整成功后替换当前拟合状态。"""

        if not self._fit_lock.acquire(blocking=False):
            raise RuntimeError("模型训练或加载正在进行")
        try:
            candidate = _native._load_regression_model(str(path))
            self.model_ = candidate
            self.n_features_in_ = candidate.feature_count
            self.device_ = self.device
            self.n_estimators_ = candidate.tree_count
            self.training_summary_ = {"loaded": True, "device": self.device}
            return self
        finally:
            self._fit_lock.release()

    def _require_model(self) -> Any:
        """返回完整 native 模型；未拟合时给出稳定异常。"""

        if not hasattr(self, "model_"):
            raise NotFittedError("MPSBoostRegressor 尚未拟合或加载模型")
        return self.model_

    def _clear_fitted_state(self) -> None:
        """统一删除全部拟合后字段，避免失败路径遗漏某个状态。"""

        for name in (
            "model_",
            "n_features_in_",
            "device_",
            "n_estimators_",
            "training_summary_",
        ):
            self.__dict__.pop(name, None)

    def _validate_parameters(self) -> None:
        """在设备初始化前集中验证全部 Python 公共参数。"""

        integer_ranges = {
            "n_estimators": (self.n_estimators, 1, 2**32 - 1),
            "max_depth": (self.max_depth, 0, 31),
            "max_bins": (self.max_bins, 2, 65536),
            "min_samples_leaf": (self.min_samples_leaf, 1, 2**63 - 1),
            "verbosity": (self.verbosity, 0, 3),
        }
        for name, (value, minimum, maximum) in integer_ranges.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} 必须是整数")
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} 必须位于 [{minimum}, {maximum}]")
        for name, numeric_value, lower, upper, lower_inclusive in (
            ("learning_rate", self.learning_rate, 0.0, 1.0, False),
            ("min_child_weight", self.min_child_weight, 0.0, np.inf, True),
            ("reg_lambda", self.reg_lambda, 0.0, np.inf, True),
        ):
            if isinstance(numeric_value, bool) or not isinstance(
                numeric_value, (int, float)
            ):
                raise TypeError(f"{name} 必须是实数")
            numeric = float(numeric_value)
            valid_lower = numeric >= lower if lower_inclusive else numeric > lower
            if not np.isfinite(numeric) or not valid_lower or numeric > upper:
                bracket = "[" if lower_inclusive else "("
                raise ValueError(f"{name} 必须位于 {bracket}{lower}, {upper}]")
        if self.device not in {"mps", "cpu"}:
            raise ValueError("device 只能是 'mps' 或 'cpu'")
        if self.random_state is not None and (
            isinstance(self.random_state, bool) or not isinstance(self.random_state, int)
        ):
            raise TypeError("random_state 必须是整数或 None")
