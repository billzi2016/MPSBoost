"""MPSBoost 对外公开入口。

本文件只负责整理并导出稳定的公共符号，不承载训练逻辑。这样做有两个目的：

1. 用户始终可以通过 ``import mpsboost as mps`` 获得清晰、稳定的入口；
2. 内部模块可以持续重构，而不会意外把实现细节暴露为公共 API。

当前版本是 0.1.0a0 技术预览，只用于锁定包名和 API 方向。训练相关入口会明确
抛出 :class:`PreviewFeatureUnavailable`，绝不会静默切换到一个未声明的 CPU 实现。
"""

from ._api import (
    Booster,
    MPSBoostClassifier,
    MPSBoostRegressor,
    MPSMatrix,
    PreviewFeatureUnavailable,
    is_available,
    system_info,
    train,
)
from ._cache import CacheLayout, cache_layout

# 版本同时写入 pyproject.toml。正式实现阶段应改为单一版本源，避免两处手工维护；
# 占位版保留显式常量，确保即使构建元数据不可访问也能可靠查询版本。
__version__ = "0.1.0a0"

# 明确维护 __all__，避免 ``from mpsboost import *`` 泄露内部辅助函数和模块。
__all__ = [
    "Booster",
    "CacheLayout",
    "MPSBoostClassifier",
    "MPSBoostRegressor",
    "MPSMatrix",
    "PreviewFeatureUnavailable",
    "__version__",
    "cache_layout",
    "is_available",
    "system_info",
    "train",
]
