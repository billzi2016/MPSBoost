# 后端选择策略

MPSBoost 当前有两个一等 native 后端：

- `device="cpu"`：使用项目内 CPU backend，也是 correctness oracle。
- `device="mps"`：请求 Apple GPU/MPS 后端。
- `device="auto"`：根据环境和工作量选择后端。

CPU backend 不是临时替代品。它用于正确性验证、小数据快速路径、环境不可用时的可运行路径，以及部分 CPU-suitable estimator。

## CPU-suitable estimator

`IsolationForest` / `MPSIsolationForest` 和 `LearningToRankRegressor` 属于 CPU-suitable workflow。它们的计算形态分支多、延迟敏感，当前预计 CPU 比 Apple GPU 更合适。

如果用户传入 `device="mps"`，库会：

- 发出 warning。
- 继续运行。
- 在 `training_summary_` 中记录 `requested_device`、实际 `device` 和原因。

文案使用产品化后端调度口径，不把 CPU 选择描述为实现缺失。
