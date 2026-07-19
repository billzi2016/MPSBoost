"""Python estimator 到真实 MPS 训练与模型预测的端到端测试。

该文件必须在自托管 Apple Silicon 真实设备运行，覆盖 Python→量化→GPU gradient/
histogram→多轮树→预测完整链路；禁止 mock、CPU 回退或只验证内部 kernel。
"""

import numpy as np
import pytest

from mpsboost import MPSBoostRegressor

pytestmark = pytest.mark.gpu


def test_real_mps_regressor_matches_cpu_model_on_stable_dataset():
    """同一冻结语义下，MPS 与 CPU oracle 的结构效果和预测应在累计容差内一致。"""

    X = np.arange(128, dtype=np.float32).reshape(64, 2)
    y = np.where(X[:, 0] < 64, -1.0, 2.0).astype(np.float64)
    parameters = dict(
        n_estimators=5,
        learning_rate=0.3,
        max_depth=3,
        max_bins=32,
        min_samples_leaf=2,
        min_child_weight=1.0,
        reg_lambda=1.0,
    )
    cpu = MPSBoostRegressor(device="cpu", **parameters).fit(X, y)
    mps = MPSBoostRegressor(device="mps", **parameters).fit(X, y)
    np.testing.assert_allclose(mps.predict(X), cpu.predict(X), rtol=2e-4, atol=2e-4)
    assert mps.device_ == "mps"
    assert mps.n_estimators_ == parameters["n_estimators"]
