"""MPSBoost 分层缓存位置定义。

本模块只计算路径，不在导入或查询时创建目录。这样可以保证：

- ``import mpsboost`` 没有文件系统副作用；
- 只读环境仍可导入并查询配置；
- 用户可以在真正需要缓存时决定权限、位置和清理策略；
- 测试不会因一次普通导入污染用户主目录。

缓存按生命周期分为三层：L1 是进程内对象；L2 是本模块描述的用户可重建缓存；
L3 是构建/CI 产物，由构建系统管理，不应混入运行时 API。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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
