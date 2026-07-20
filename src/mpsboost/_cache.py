"""MPSBoost layered cache service.

This module defines the sole runtime L2-cache paths, keys, read/write, validation,
and cleanup entries. Import and query paths create no directories; only explicit
writes or creation touch the file system. Caching improves speed only and never
becomes a prerequisite for correct training or prediction.

- ``import mpsboost`` has no file-system side effects;
- read-only environments can still import and query configuration;
- users choose permissions, location, and cleanup when caching is needed;
- ordinary imports never pollute the user home directory during tests.

Caching has three lifecycle layers: L1 is in-process objects; L2 is the
user-rebuildable cache described here; L3 is build/CI output managed by tooling
and excluded from the runtime API.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class CacheLayout:
    """User-level, safely rebuildable L2 cache paths.

    L1 lives in process memory and has no file-system path; L3 belongs to build and
    CI tooling and is intentionally absent from the runtime API. Each subdirectory
    uses its own format version and invalidation key.

    Attributes:
        root: MPSBoost user cache root.
        pipelines: Rebuildable compiled caches such as Metal pipeline/binary archives.
        quantization: Binned metadata cache saved only with explicit user permission.
        tuning: Autotuning results tied to device and software versions.
    """

    root: Path
    pipelines: Path
    quantization: Path
    tuning: Path


@dataclass(frozen=True, slots=True)
class CacheKey:
    """A normalized L2 cache key.

    namespace selects the cache subdirectory, version controls format invalidation,
    and parts holds non-sensitive semantic fields. Filenames use SHA-256 of canonical
    JSON, avoiding paths, usernames, or raw data.
    """

    namespace: str
    version: str
    parts: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CacheInfo:
    """Cache summary safe to display to users."""

    root: Path
    exists: bool
    pipelines: Path
    quantization: Path
    tuning: Path


_CACHE_MAGIC = b"MPSBOOST-CACHE\0"
_CACHE_CONTAINER_VERSION = 1
_ALLOWED_NAMESPACES = frozenset({"pipelines", "quantization", "tuning"})


def cache_layout() -> CacheLayout:
    """Compute and return L2 cache paths without creating directories.

    ``MPSBOOST_CACHE_DIR`` lets containers, CI, or advanced users override the root.
    Otherwise it follows macOS cache conventions. Version keys, atomic writes,
    validation, and corruption recovery precede real writes; this function plans paths only.
    """

    configured = os.environ.get("MPSBOOST_CACHE_DIR")
    if configured:
        # expanduser supports explicit ``~/...`` configuration. Do not call resolve
        # so nonexistent paths are not queried and symlink semantics stay unfrozen.
        root = Path(configured).expanduser()
    else:
        root = Path.home() / "Library" / "Caches" / "mpsboost"
    # Subdirectories are isolated by data semantics so they can be cleared and
    # upgraded independently without invalidating every cache type at once.
    return CacheLayout(
        root=root,
        pipelines=root / "pipelines",
        quantization=root / "quantization",
        tuning=root / "tuning",
    )


def cache_info() -> CacheInfo:
    """Return a cache-path summary without creating directories or reading files."""

    layout = cache_layout()
    return CacheInfo(
        root=layout.root,
        exists=layout.root.exists(),
        pipelines=layout.pipelines,
        quantization=layout.quantization,
        tuning=layout.tuning,
    )


def ensure_cache_directories() -> CacheLayout:
    """Explicitly create L2 cache directories and return the layout.

    Raises:
        ValueError: The cache root resolves to a dangerous path or symlink.
        OSError: The caller lacks permission to create directories.
    """

    layout = cache_layout()
    _validate_cache_root(layout.root)
    for path in (layout.root, layout.pipelines, layout.quantization, layout.tuning):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return layout


def cache_path(key: CacheKey) -> Path:
    """Compute a cache-file path for key without creating directories."""

    payload = _canonical_key_payload(key)
    digest = hashlib.sha256(payload).hexdigest()
    layout = cache_layout()
    namespace_root = _namespace_root(layout, key.namespace)
    return namespace_root / f"{digest}.bin"


def write_cache_bytes(key: CacheKey, payload: bytes) -> Path:
    """Atomically write cache bytes in a validated container.

    Writes use a same-directory temporary file followed by full fsync and ``replace``
    so readers never observe partial files. Callers ensure payload excludes raw
    training data, labels, paths, and credentials.
    """

    if not isinstance(payload, bytes):
        raise TypeError("Cache payload must be bytes")
    # Normalize the key before creating directories. Unknown namespaces, empty
    # versions, and non-serializable fields must fail without polluting cache root.
    _canonical_key_payload(key)
    layout = ensure_cache_directories()
    target = _namespace_root(layout, key.namespace) / cache_path(key).name
    record = _encode_record(key, payload)
    descriptor = -1
    temporary_name = ""
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(record)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(target)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
    return target


def read_cache_bytes(key: CacheKey) -> bytes | None:
    """Read and validate cache; missing, corrupt, or stale keys return ``None``.

    Corrupt files are removed on a best-effort basis. Removal failure never affects
    callers because caching is not a correctness prerequisite.
    """

    path = cache_path(key)
    try:
        record = path.read_bytes()
        return _decode_record(key, record)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def clear_cache() -> int:
    """Explicitly clear the entire L2 cache root.

    Returns:
        Estimated count of removed regular files.

    Raises:
        ValueError: The root is filesystem root, user home, a symlink, or another dangerous target.
    """

    layout = cache_layout()
    if not layout.root.exists():
        return 0
    _validate_cache_root(layout.root)
    removed_files = sum(1 for item in layout.root.rglob("*") if item.is_file())
    shutil.rmtree(layout.root)
    return removed_files


def _namespace_root(layout: CacheLayout, namespace: str) -> Path:
    if namespace == "pipelines":
        return layout.pipelines
    if namespace == "quantization":
        return layout.quantization
    if namespace == "tuning":
        return layout.tuning
    raise ValueError("Cache namespace is unsupported")


def _canonical_key_payload(key: CacheKey) -> bytes:
    if key.namespace not in _ALLOWED_NAMESPACES:
        raise ValueError("Cache namespace is unsupported")
    if not key.version:
        raise ValueError("Cache key version must not be empty")
    payload = {
        "namespace": key.namespace,
        "version": key.version,
        "parts": key.parts,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _encode_record(key: CacheKey, payload: bytes) -> bytes:
    key_payload = _canonical_key_payload(key)
    checksum = hashlib.sha256(key_payload + b"\0" + payload).hexdigest().encode("ascii")
    header = {
        "container": _CACHE_CONTAINER_VERSION,
        "key": json.loads(key_payload.decode("utf-8")),
        "checksum": checksum.decode("ascii"),
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return (
        _CACHE_MAGIC + len(header_bytes).to_bytes(4, "little") + header_bytes + payload
    )


def _decode_record(key: CacheKey, record: bytes) -> bytes:
    if not record.startswith(_CACHE_MAGIC):
        raise ValueError("Cache magic does not match")
    header_start = len(_CACHE_MAGIC) + 4
    if len(record) < header_start:
        raise ValueError("Cache header is truncated")
    header_size = int.from_bytes(record[len(_CACHE_MAGIC) : header_start], "little")
    header_end = header_start + header_size
    if header_end > len(record):
        raise ValueError("Cache header is out of bounds")
    header = json.loads(record[header_start:header_end].decode("utf-8"))
    if header.get("container") != _CACHE_CONTAINER_VERSION:
        raise ValueError("Cache container version does not match")
    expected_key = json.loads(_canonical_key_payload(key).decode("utf-8"))
    if header.get("key") != expected_key:
        raise ValueError("Cache key does not match")
    payload = record[header_end:]
    checksum = hashlib.sha256(_canonical_key_payload(key) + b"\0" + payload).hexdigest()
    if header.get("checksum") != checksum:
        raise ValueError("Cache checksum does not match")
    return payload


def _validate_cache_root(root: Path) -> None:
    expanded = root.expanduser()
    resolved = expanded.resolve(strict=False)
    home = Path.home().resolve(strict=False)
    if resolved == Path(resolved.anchor):
        raise ValueError("Refusing to use the filesystem root as the MPSBoost cache root")
    if resolved == home:
        raise ValueError("Refusing to use the user home directory as the MPSBoost cache root")
    if expanded.exists() and expanded.is_symlink():
        raise ValueError("Refusing to clear or create a symlinked MPSBoost cache root")
