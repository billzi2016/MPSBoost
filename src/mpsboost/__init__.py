"""MPSBoost 对外公开入口。

本文件只负责整理并导出稳定的公共符号，不承载训练逻辑。这样做有两个目的：

1. 用户始终可以通过 ``import mpsboost as mb`` 获得清晰、稳定的入口；
2. 内部模块可以持续重构，而不会意外把实现细节暴露为公共 API。

当前开发版本提供真实 MPS/Metal 训练、设备诊断和 estimator 风格回归入口。
"""

from ._native import __version__  # type: ignore[import-not-found]
from .diagnostics import (
    cache_info,
    clear_cache,
    create_cache,
    is_available,
    system_info,
)
from .estimator import MPSBoostRegressor

# 明确维护 __all__，避免 ``from mpsboost import *`` 泄露内部辅助函数和模块。
__all__ = [
    "MPSBoostRegressor",
    "__version__",
    "cache_info",
    "clear_cache",
    "create_cache",
    "is_available",
    "system_info",
]
