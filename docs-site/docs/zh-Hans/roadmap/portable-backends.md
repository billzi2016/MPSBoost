# S22 Portable Backends

S22 目标是让同一套 MPSBoost estimator 接口可在更多机器上运行。

## 不替代 native CPU

MPSBoost native CPU backend 继续保留：

- correctness oracle
- 默认 CPU backend
- Apple Silicon 无 GPU 或小工作负载路径
- 测试基线

portable backend 不替代 native CPU，也不伪装成 native MPSBoost。

## 目标后端

后续规划：

- Apple Silicon：MPSBoost native CPU/MPS
- Linux CPU：MPSBoost native CPU 或 sklearn/XGBoost CPU adapter
- Linux CUDA：XGBoost GPU adapter
- Windows CPU：sklearn/XGBoost CPU adapter

所有外部后端必须写入 `training_summary_["backend"]`，用户可以明确看到实际运行路径。

## 依赖策略

外部后端使用 optional extras：

- `mpsboost[xgboost]`
- `mpsboost[sklearn]`
- `mpsboost[cuda]`

默认安装保持轻量。
