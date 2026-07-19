"""分层缓存路径的无副作用单元测试。"""

import mpsboost as mps


def test_cache_lookup_does_not_create_directories(tmp_path, monkeypatch):
    """查询 L2 布局不得创建目录或污染用户文件系统。"""

    root = tmp_path / "cache"
    monkeypatch.setenv("MPSBOOST_CACHE_DIR", str(root))
    layout = mps.cache_layout()
    assert layout.root == root
    assert layout.pipelines == root / "pipelines"
    assert layout.quantization == root / "quantization"
    assert layout.tuning == root / "tuning"
    assert not root.exists()
