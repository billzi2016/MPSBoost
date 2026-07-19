"""公开 API 与版本单一来源测试。

未完成的训练 estimator 不得提前暴露；这样可以防止用户把开发占位误认为正式能力。
"""

from importlib.metadata import version

import mpsboost as mps


def test_version_is_injected_by_native_build():
    """native 顶层版本必须与安装包 metadata 的唯一版本来源一致。"""

    assert mps.__version__ == version("mpsboost")


def test_unfinished_estimators_are_not_public():
    """真实训练完成前不得发布 mock estimator。"""

    assert not hasattr(mps, "MPSBoostRegressor")
    assert not hasattr(mps, "MPSBoostClassifier")
