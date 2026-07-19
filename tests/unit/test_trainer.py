"""多轮 boosting 状态机的 CPU oracle 单元测试。

测试使用真实 C++ trainer、真实分箱和唯一树实现，不在 Python 重写 boosting 算法；小型
数据只用独立性质和可复现结果断言，避免以被测实现计算自身期望。
"""

import numpy as np
import pytest

from mpsboost import MPSBoostRegressor


def test_cpu_boosting_reduces_squared_error_and_is_deterministic():
    """多轮模型应显著降低训练误差，重复训练预测必须完全一致。"""

    X = np.arange(32, dtype=np.float32).reshape(-1, 1)
    y = np.where(X[:, 0] < 16, -2.0, 3.0).astype(np.float64)
    parameters = dict(
        n_estimators=4,
        learning_rate=0.5,
        max_depth=2,
        max_bins=16,
        min_samples_leaf=1,
        min_child_weight=1.0,
        reg_lambda=0.0,
        device="cpu",
    )
    first = MPSBoostRegressor(**parameters).fit(X, y)
    second = MPSBoostRegressor(**parameters).fit(X, y)
    baseline_error = np.mean((y - np.mean(y)) ** 2)
    model_error = np.mean((y - first.predict(X)) ** 2)
    assert model_error < baseline_error * 0.02
    np.testing.assert_array_equal(first.predict(X), second.predict(X))
    assert first.n_estimators_ == 4


def test_fixed_training_boundaries_are_used_for_new_data():
    """预测新数据必须应用训练 schema，而不是根据预测批次重新拟合边界。"""

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([0.0, 0.0, 10.0, 10.0])
    model = MPSBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        max_bins=4,
        min_samples_leaf=1,
        min_child_weight=1.0,
        reg_lambda=0.0,
        device="cpu",
    ).fit(X, y)
    prediction = model.predict(np.array([[-100.0], [100.0]], dtype=np.float32))
    assert prediction[0] < prediction[1]


@pytest.mark.parametrize(
    "name,value,message",
    [
        ("n_estimators", 0, "n_estimators"),
        ("learning_rate", 0.0, "learning_rate"),
        ("max_depth", -1, "max_depth"),
        ("max_bins", 1, "max_bins"),
        ("min_samples_leaf", 0, "min_samples_leaf"),
    ],
)
def test_invalid_training_parameters_fail_before_model(name, value, message):
    """公共参数错误不得创建任何拟合后状态。"""

    model = MPSBoostRegressor(device="cpu", **{name: value})
    with pytest.raises((TypeError, ValueError), match=message):
        model.fit(np.ones((2, 1), dtype=np.float32), np.ones(2))
    assert not hasattr(model, "model_")
