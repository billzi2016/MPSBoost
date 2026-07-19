"""分层缓存服务的无副作用和安全性单元测试。"""

import pytest

from mpsboost import cache_info, clear_cache as public_clear_cache
from mpsboost import create_cache
from mpsboost._cache import (
    CacheKey,
    cache_layout,
    cache_path,
    clear_cache,
    ensure_cache_directories,
    read_cache_bytes,
    write_cache_bytes,
)


def test_cache_lookup_does_not_create_directories(tmp_path, monkeypatch):
    """查询 L2 布局不得创建目录或污染用户文件系统。"""

    root = tmp_path / "cache"
    monkeypatch.setenv("MPSBOOST_CACHE_DIR", str(root))
    # 缓存仍属于 S7 内部能力，不能为了测试方便提前污染 0.2.0 顶层公共 API。
    layout = cache_layout()
    assert layout.root == root
    assert layout.pipelines == root / "pipelines"
    assert layout.quantization == root / "quantization"
    assert layout.tuning == root / "tuning"
    assert not root.exists()

    public_info = cache_info()
    assert public_info["root"] == str(root)
    assert public_info["exists"] is False
    assert not root.exists()


def test_explicit_creation_uses_layered_directories(tmp_path, monkeypatch):
    """只有显式创建 API 才能创建 L2 目录。"""

    root = tmp_path / "cache"
    monkeypatch.setenv("MPSBOOST_CACHE_DIR", str(root))
    created = create_cache()
    layout = ensure_cache_directories()
    assert created["root"] == str(root)
    assert layout.root.is_dir()
    assert layout.pipelines.is_dir()
    assert layout.quantization.is_dir()
    assert layout.tuning.is_dir()


def test_cache_round_trip_and_version_invalidation(tmp_path, monkeypatch):
    """缓存 key 必须包含版本；旧版本不得读出新版本内容。"""

    monkeypatch.setenv("MPSBOOST_CACHE_DIR", str(tmp_path / "cache"))
    key = CacheKey("tuning", "v1", {"device": "test", "rows": 16})
    path = write_cache_bytes(key, b"payload")
    assert path == cache_path(key)
    assert read_cache_bytes(key) == b"payload"
    old_key = CacheKey("tuning", "v0", {"device": "test", "rows": 16})
    assert read_cache_bytes(old_key) is None


def test_corrupted_cache_is_ignored_and_removed(tmp_path, monkeypatch):
    """损坏缓存必须安全失效，不能把坏字节交给调用方。"""

    monkeypatch.setenv("MPSBOOST_CACHE_DIR", str(tmp_path / "cache"))
    key = CacheKey("quantization", "v1", {"schema": "abc"})
    path = write_cache_bytes(key, b"good")
    path.write_bytes(b"broken")
    assert read_cache_bytes(key) is None
    assert not path.exists()


def test_clear_cache_removes_only_valid_cache_root(tmp_path, monkeypatch):
    """清理 API 只操作显式缓存根目录，并返回移除文件数量。"""

    root = tmp_path / "cache"
    monkeypatch.setenv("MPSBOOST_CACHE_DIR", str(root))
    write_cache_bytes(CacheKey("pipelines", "v1", {"shape": [1, 2]}), b"x")
    assert public_clear_cache() == 1
    assert not root.exists()


def test_clear_cache_rejects_home_and_symlink(tmp_path, monkeypatch):
    """危险目标必须拒绝，避免误删用户目录或符号链接逃逸。"""

    monkeypatch.setenv("MPSBOOST_CACHE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="主目录|根目录"):
        # tmp_path 不是主目录；这里直接测试符号链接分支之外的危险路径需要真实 HOME。
        monkeypatch.setenv("MPSBOOST_CACHE_DIR", str(tmp_path.home()))
        clear_cache()

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("MPSBOOST_CACHE_DIR", str(link))
    with pytest.raises(ValueError, match="符号链接"):
        clear_cache()


def test_invalid_namespace_fails_before_directory_creation(tmp_path, monkeypatch):
    """未知 namespace 早失败，并且不得创建缓存根目录。"""

    root = tmp_path / "cache"
    monkeypatch.setenv("MPSBOOST_CACHE_DIR", str(root))
    with pytest.raises(ValueError, match="namespace"):
        write_cache_bytes(CacheKey("unknown", "v1", {}), b"x")
    assert not root.exists()
