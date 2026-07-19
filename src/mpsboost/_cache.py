"""MPSBoost 分层缓存服务。

本模块定义运行时 L2 缓存的唯一路径、key、读写、校验和清理入口。导入和查询路径不
创建目录；只有显式写入或显式创建时才触碰文件系统。缓存只能提升速度，不能成为训练
或预测正确性的前提，任何损坏、旧版本或无权限都必须安全失败或回到无缓存路径。

- ``import mpsboost`` 没有文件系统副作用；
- 只读环境仍可导入并查询配置；
- 用户可以在真正需要缓存时决定权限、位置和清理策略；
- 测试不会因一次普通导入污染用户主目录。

缓存按生命周期分为三层：L1 是进程内对象；L2 是本模块描述的用户可重建缓存；
L3 是构建/CI 产物，由构建系统管理，不应混入运行时 API。
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
    """用户级、可安全重建的 L2 缓存路径集合。

    L1 位于进程内存，没有文件系统路径；L3 由构建和 CI 工具拥有，故意不暴露在
    运行时 API 中。这里的每个子目录必须使用独立格式版本和失效 key，不能因为共享
    根目录就混用不同生命周期的数据。

    Attributes:
        root: MPSBoost 用户缓存根目录。
        pipelines: Metal pipeline/binary archive 等可重建编译缓存。
        quantization: 用户显式允许时保存的分箱元数据缓存。
        tuning: 与设备和软件版本绑定的自动调优结果。
    """

    root: Path
    pipelines: Path
    quantization: Path
    tuning: Path


@dataclass(frozen=True, slots=True)
class CacheKey:
    """一个已规范化的 L2 缓存 key。

    namespace 决定缓存子目录，version 决定格式失效，parts 保存不含敏感信息的语义字段。
    文件名由 canonical JSON 的 SHA-256 生成，避免把路径、用户名或原始数据写入文件名。
    """

    namespace: str
    version: str
    parts: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CacheInfo:
    """可安全展示给用户的缓存摘要。"""

    root: Path
    exists: bool
    pipelines: Path
    quantization: Path
    tuning: Path


_CACHE_MAGIC = b"MPSBOOST-CACHE\0"
_CACHE_CONTAINER_VERSION = 1
_ALLOWED_NAMESPACES = frozenset({"pipelines", "quantization", "tuning"})


def cache_layout() -> CacheLayout:
    """计算并返回 L2 缓存路径，但不创建任何目录。

    ``MPSBOOST_CACHE_DIR`` 用于容器、CI 或高级用户显式改写根目录。未设置时遵循
    macOS 用户缓存习惯，使用 ``~/Library/Caches/mpsboost``。正式写缓存前还必须
    增加版本 key、原子写入、校验和损坏恢复；本函数只负责无副作用的路径规划。
    """

    configured = os.environ.get("MPSBOOST_CACHE_DIR")
    if configured:
        # expanduser 支持用户显式配置 ``~/...``；不调用 resolve，避免查询不存在路径
        # 或把符号链接语义提前固化。
        root = Path(configured).expanduser()
    else:
        root = Path.home() / "Library" / "Caches" / "mpsboost"
    # 子目录按数据语义隔离，未来可以独立清理和升级，避免“一次缓存格式升级导致
    # 所有类型缓存同时失效”。
    return CacheLayout(
        root=root,
        pipelines=root / "pipelines",
        quantization=root / "quantization",
        tuning=root / "tuning",
    )


def cache_info() -> CacheInfo:
    """返回缓存路径摘要，不创建目录、不读取文件内容。"""

    layout = cache_layout()
    return CacheInfo(
        root=layout.root,
        exists=layout.root.exists(),
        pipelines=layout.pipelines,
        quantization=layout.quantization,
        tuning=layout.tuning,
    )


def ensure_cache_directories() -> CacheLayout:
    """显式创建 L2 缓存目录并返回布局。

    Raises:
        ValueError: 缓存根目录解析为危险路径或符号链接。
        OSError: 调用方没有权限创建目录。
    """

    layout = cache_layout()
    _validate_cache_root(layout.root)
    for path in (layout.root, layout.pipelines, layout.quantization, layout.tuning):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return layout


def cache_path(key: CacheKey) -> Path:
    """计算指定 key 的缓存文件路径，不创建目录。"""

    payload = _canonical_key_payload(key)
    digest = hashlib.sha256(payload).hexdigest()
    layout = cache_layout()
    namespace_root = _namespace_root(layout, key.namespace)
    return namespace_root / f"{digest}.bin"


def write_cache_bytes(key: CacheKey, payload: bytes) -> Path:
    """使用校验容器原子写入缓存字节。

    写入发生在同目录临时文件，完整 fsync 后再 ``replace``，避免读者看到半文件。payload
    由调用方负责保证不含原始训练数据、标签、路径或凭据。
    """

    if not isinstance(payload, bytes):
        raise TypeError("缓存 payload 必须是 bytes")
    # 先规范化 key，再创建目录；未知 namespace、空版本或不可序列化字段必须无副作用
    # 失败，避免错误调用污染用户 cache root。
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
    """读取并校验缓存；缺失、损坏或旧 key 返回 ``None``。

    损坏文件会被尽力删除。删除失败不影响调用方，因为缓存不是正确性前提。
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
    """显式清理整个 L2 缓存根目录。

    Returns:
        被移除的普通文件数量估计值。

    Raises:
        ValueError: 根目录是根目录、用户主目录、符号链接或其他危险目标。
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
    raise ValueError("缓存 namespace 不受支持")


def _canonical_key_payload(key: CacheKey) -> bytes:
    if key.namespace not in _ALLOWED_NAMESPACES:
        raise ValueError("缓存 namespace 不受支持")
    if not key.version:
        raise ValueError("缓存 key version 不能为空")
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
        raise ValueError("缓存 magic 不匹配")
    header_start = len(_CACHE_MAGIC) + 4
    if len(record) < header_start:
        raise ValueError("缓存 header 截断")
    header_size = int.from_bytes(record[len(_CACHE_MAGIC) : header_start], "little")
    header_end = header_start + header_size
    if header_end > len(record):
        raise ValueError("缓存 header 越界")
    header = json.loads(record[header_start:header_end].decode("utf-8"))
    if header.get("container") != _CACHE_CONTAINER_VERSION:
        raise ValueError("缓存容器版本不匹配")
    expected_key = json.loads(_canonical_key_payload(key).decode("utf-8"))
    if header.get("key") != expected_key:
        raise ValueError("缓存 key 不匹配")
    payload = record[header_end:]
    checksum = hashlib.sha256(_canonical_key_payload(key) + b"\0" + payload).hexdigest()
    if header.get("checksum") != checksum:
        raise ValueError("缓存 checksum 不匹配")
    return payload


def _validate_cache_root(root: Path) -> None:
    expanded = root.expanduser()
    resolved = expanded.resolve(strict=False)
    home = Path.home().resolve(strict=False)
    if resolved == Path(resolved.anchor):
        raise ValueError("拒绝把文件系统根目录作为 MPSBoost 缓存根目录")
    if resolved == home:
        raise ValueError("拒绝把用户主目录作为 MPSBoost 缓存根目录")
    if expanded.exists() and expanded.is_symlink():
        raise ValueError("拒绝清理或创建符号链接形式的 MPSBoost 缓存根目录")
