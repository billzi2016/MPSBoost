"""平方误差与训练数学的手算领域测试。

测试直接调用唯一 C++ 实现，并用独立手算常量验证 gradient/Hessian、节点分数、叶值
和切分增益；禁止在 Python 中重写通用公式作为期望值。
"""

import math

import pytest

from mpsboost import _native


def test_squared_error_gradients_match_hand_computation():
    """平方误差必须产生 prediction-label 和恒为 1 的 Hessian。"""

    actual = _native._squared_error_gradients([1.0, -2.0, 4.0], [1.5, -3.0, 2.0])
    assert actual == [(0.5, 1.0), (-1.0, 1.0), (-2.0, 1.0)]


def test_score_weight_and_gain_match_hand_computation():
    """冻结 lambda 与 gamma 参与数学公式的位置，防止多个后端语义漂移。"""

    assert _native._node_score(-4.0, 2.0, 1.0) == pytest.approx(16.0 / 3.0)
    assert _native._leaf_weight(-4.0, 2.0, 1.0) == pytest.approx(4.0 / 3.0)
    # left=(0,2)、right=(-4,2)、lambda=0 时，父分数为 4、右分数为 8，
    # 因而未扣 gamma 的增益为 2；这里再显式扣除 0.25。
    assert _native._split_gain(0.0, 2.0, -4.0, 2.0, 0.0, 0.25) == 1.75


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: _native._squared_error_gradients([], []), "标签不能为空"),
        (lambda: _native._squared_error_gradients([1.0], []), "长度不一致"),
        (lambda: _native._squared_error_gradients([math.nan], [0.0]), "有限值"),
        (lambda: _native._node_score(1.0, -1.0, 1.0), "Hessian"),
        (lambda: _native._leaf_weight(1.0, 0.0, 0.0), "有限正数"),
        (lambda: _native._split_gain(0.0, 0.0, 1.0, 1.0, 1.0, 0.0), "严格为正"),
        (lambda: _native._split_gain(1.0, 1.0, 1.0, 1.0, 0.0, -1.0), "gamma"),
    ],
)
def test_invalid_objective_inputs_fail_before_model_construction(call, message):
    """非有限值、非法正则项和无效 Hessian 不得产生可继续训练的统计。"""

    with pytest.raises(_native.TrainingError, match=message):
        call()
