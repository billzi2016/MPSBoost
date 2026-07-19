"""MPSBoost 对外公开入口。

本文件只负责整理并导出稳定的公共符号，不承载训练逻辑。这样做有两个目的：

1. 用户始终可以通过 ``import mpsboost as mps`` 获得清晰、稳定的入口；
2. 内部模块可以持续重构，而不会意外把实现细节暴露为公共 API。

当前开发版本提供真实的 MPS/Metal 设备诊断和 GPU smoke kernel。训练 estimator 尚未
达到规格要求，因此不会提前导出占位类，也不会静默切换到未声明的 CPU 实现。
"""

from ._native import __version__
from ._cache import CacheLayout, cache_layout
from .diagnostics import is_available, system_info

# 明确维护 __all__，避免 ``from mpsboost import *`` 泄露内部辅助函数和模块。
__all__ = [
    "CacheLayout",
    "__version__",
    "cache_layout",
    "is_available",
    "system_info",
]
