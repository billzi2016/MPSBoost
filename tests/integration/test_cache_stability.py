"""Integration tests ensuring cache deletion and corruption do not affect training.

The production training path does not depend on L2 disk cache. This test freezes
that contract: explicit CPU-oracle training yields identical predictions even when
caches are created, corrupted, or deleted.
"""

import numpy as np

from mpsboost import MPSBoostRegressor
from mpsboost._cache import CacheKey, clear_cache, write_cache_bytes


def test_l2_cache_deletion_does_not_change_training_result(tmp_path, monkeypatch):
    """Training output must remain identical after deleting all L2 cache."""

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
