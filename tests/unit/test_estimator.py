"""Estimator 参数、输入和状态原子性测试。

本文件只验证 Python 用户契约；训练正确性由 trainer 与真实 Metal 测试覆盖，不使用假
native handle 或 mock 设备让公共 fit 路径成功。
"""

import numpy as np
import pytest

from mpsboost import MPSBoostRegressor
from mpsboost.estimator import NotFittedError


def test_get_and_set_params_follow_estimator_protocol():
    """全部显式构造参数必须可发现，未知参数失败且 set_params 返回自身。"""

    model = MPSBoostRegressor(n_estimators=3, device="cpu")
    assert set(model.get_params()) == set(model._PARAMETER_NAMES)
    assert model.set_params(n_estimators=5) is model
    assert model.n_estimators == 5
    with pytest.raises(ValueError, match="未知参数"):
        model.set_params(unknown=1)


def test_unfitted_and_wrong_feature_prediction_fail_explicitly():
    """未拟合与特征数变化必须给出稳定异常。"""

    model = MPSBoostRegressor(
        n_estimators=1, max_depth=0, min_samples_leaf=1, device="cpu"
    )
    with pytest.raises(NotFittedError):
        model.predict(np.ones((1, 1), dtype=np.float32))
    model.fit(np.ones((2, 1), dtype=np.float32), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="特征数量"):
        model.predict(np.ones((2, 2), dtype=np.float32))


def test_failed_refit_clears_previous_model_instead_of_exposing_partial_state():
    """失败 refit 不得继续呈现与当前训练请求无关的旧模型。"""

    model = MPSBoostRegressor(
        n_estimators=1, max_depth=0, min_samples_leaf=1, device="cpu"
    ).fit(np.ones((2, 1), dtype=np.float32), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="样本数量"):
        model.fit(np.ones((3, 1), dtype=np.float32), np.array([1.0, 2.0]))
    with pytest.raises(NotFittedError):
        model.predict(np.ones((1, 1), dtype=np.float32))


def test_dense_adapter_rejects_sparse_complex_and_invalid_rank():
    """当前版本范围外输入不能被静默展开或截断。"""

    model = MPSBoostRegressor(device="cpu")
    with pytest.raises(TypeError, match="dtype"):
        model.fit(np.ones((2, 1), dtype=np.complex64), np.ones(2))
    with pytest.raises(ValueError, match="二维"):
        model.fit(np.ones(2), np.ones(2))
