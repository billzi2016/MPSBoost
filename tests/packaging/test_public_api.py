"""公开 API、estimator 入口与版本单一来源测试。"""

from importlib.metadata import version

import mpsboost as mps


def test_version_is_injected_by_native_build():
    """native 顶层版本必须与安装包 metadata 的唯一版本来源一致。"""

    assert mps.__version__ == version("mpsboost")


def test_only_completed_regressor_is_public():
    """真实回归 estimator 必须公开，未实现分类器仍不得泄露。"""

    assert mps.MPSBoostRegressor.__name__ == "MPSBoostRegressor"
    assert not hasattr(mps, "MPSBoostClassifier")
    assert set(mps.__all__) == {
        "MPSBoostRegressor",
        "__version__",
        "cache_info",
        "clear_cache",
        "create_cache",
        "is_available",
        "system_info",
    }
