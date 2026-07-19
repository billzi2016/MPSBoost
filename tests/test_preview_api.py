"""0.1.0a0 公共占位 API 的行为测试。

这些测试重点防止技术预览误导用户：后端必须报告不可用，训练必须明确失败，查询
缓存路径不能产生文件系统副作用。真实训练实现加入后，应替换相应断言而不是简单删除。
"""

import mpsboost as mps


def test_preview_version_and_backend_status():
    """版本、后端名称与可用性必须准确描述当前 wheel。"""

    assert mps.__version__ == "0.1.0a0"
    assert mps.is_available() is False
    assert mps.system_info()["backend"] == "mps"


def test_regressor_has_sklearn_style_params():
    """回归器需要支持 sklearn 工具依赖的参数读取和更新约定。"""

    model = mps.MPSBoostRegressor(n_estimators=7, device="mps")
    assert model.get_params()["n_estimators"] == 7
    assert model.set_params(max_depth=3) is model
    assert model.get_params()["max_depth"] == 3


def test_training_fails_explicitly():
    """未实现的训练不得静默成功、返回假模型或偷偷切到 CPU。"""

    model = mps.MPSBoostRegressor()
    try:
        model.fit([[1.0]], [1.0])
    except mps.PreviewFeatureUnavailable as exc:
        assert "not implemented" in str(exc)
    else:
        raise AssertionError("preview training must not silently succeed")


def test_cache_lookup_does_not_create_directories(tmp_path, monkeypatch):
    """单纯查询缓存布局不得污染磁盘，尤其不能在 import 时创建目录。"""

    root = tmp_path / "cache"
    monkeypatch.setenv("MPSBOOST_CACHE_DIR", str(root))
    layout = mps.cache_layout()
    assert layout.root == root
    assert not root.exists()
