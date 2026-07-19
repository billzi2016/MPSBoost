"""缓存删除与损坏不影响训练结果的集成测试。

当前生产训练路径不依赖 L2 磁盘缓存。本测试固定这一契约：即使缓存被创建、损坏或删除，
显式 CPU oracle 训练仍必须得到相同预测，后续 MPS 缓存接入也必须保持同一语义。
"""

import numpy as np

from mpsboost import MPSBoostRegressor
from mpsboost._cache import CacheKey, clear_cache, write_cache_bytes


def test_l2_cache_deletion_does_not_change_training_result(tmp_path, monkeypatch):
    """删除全部 L2 缓存后，训练输出必须保持一致。"""

    monkeypatch.setenv("MPSBOOST_CACHE_DIR", str(tmp_path / "cache"))
    X = np.arange(64, dtype=np.float32).reshape(32, 2)
    y = np.where(X[:, 0] < 32, -1.0, 2.0).astype(np.float64)
    parameters = dict(
        n_estimators=4,
        learning_rate=0.3,
        max_depth=3,
        max_bins=16,
        min_samples_leaf=2,
        min_child_weight=1.0,
        reg_lambda=1.0,
        device="cpu",
    )
    first = MPSBoostRegressor(**parameters).fit(X, y).predict(X)
    write_cache_bytes(CacheKey("tuning", "v1", {"case": "integration"}), b"bad-data")
    assert clear_cache() == 1
    second = MPSBoostRegressor(**parameters).fit(X, y).predict(X)
    np.testing.assert_array_equal(first, second)
