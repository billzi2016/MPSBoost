"""Side-effect and safety unit tests for the layered cache service."""

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
    """Querying L2 layout must not create directories or pollute user filesystems."""

    root = tmp_path / "cache"
    monkeypatch.setenv("MPSBOOST_CACHE_DIR", str(root))
    # Caching remains an internal S7 capability and must not pollute the 0.2.0
    # top-level public API merely for test convenience.
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
    """Only the explicit creation API may create L2 directories."""

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
    """Cache keys must include versions; old versions must not read new content."""

    monkeypatch.setenv("MPSBOOST_CACHE_DIR", str(tmp_path / "cache"))
    key = CacheKey("tuning", "v1", {"device": "test", "rows": 16})
    path = write_cache_bytes(key, b"payload")
    assert path == cache_path(key)
    assert read_cache_bytes(key) == b"payload"
    old_key = CacheKey("tuning", "v0", {"device": "test", "rows": 16})
    assert read_cache_bytes(old_key) is None


def test_corrupted_cache_is_ignored_and_removed(tmp_path, monkeypatch):
    """Corrupt cache must fail safely and never return bad bytes to callers."""

    monkeypatch.setenv("MPSBOOST_CACHE_DIR", str(tmp_path / "cache"))
    key = CacheKey("quantization", "v1", {"schema": "abc"})
    path = write_cache_bytes(key, b"good")
    path.write_bytes(b"broken")
    assert read_cache_bytes(key) is None
    assert not path.exists()


def test_clear_cache_removes_only_valid_cache_root(tmp_path, monkeypatch):
    """Cleanup API operates only on the explicit cache root and returns removed-file count."""

    root = tmp_path / "cache"
    monkeypatch.setenv("MPSBOOST_CACHE_DIR", str(root))
    write_cache_bytes(CacheKey("pipelines", "v1", {"shape": [1, 2]}), b"x")
    assert public_clear_cache() == 1
    assert not root.exists()


def test_clear_cache_rejects_home_and_symlink(tmp_path, monkeypatch):
    """Dangerous targets must be rejected to avoid deleting user directories or escaping symlinks."""

    monkeypatch.setenv("MPSBOOST_CACHE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="home directory|filesystem root"):
        # tmp_path is not home; testing this dangerous path outside the symlink branch
        # requires the real HOME.
        monkeypatch.setenv("MPSBOOST_CACHE_DIR", str(tmp_path.home()))
        clear_cache()

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("MPSBOOST_CACHE_DIR", str(link))
    with pytest.raises(ValueError, match="symlinked"):
        clear_cache()


def test_invalid_namespace_fails_before_directory_creation(tmp_path, monkeypatch):
    """Unknown namespaces fail early and must not create the cache root."""

    root = tmp_path / "cache"
    monkeypatch.setenv("MPSBOOST_CACHE_DIR", str(root))
    with pytest.raises(ValueError, match="namespace"):
        write_cache_bytes(CacheKey("unknown", "v1", {}), b"x")
    assert not root.exists()
