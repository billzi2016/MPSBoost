# S22 Portable Backends

S22 目标是让同一套 MPSBoost estimator 接口可在更多机器上运行。

## 不替代 native CPU

MPSBoost native CPU backend 继续保留：

- correctness oracle
- 默认 CPU backend
- Apple Silicon 无 GPU 或小工作负载路径
- 测试基线

portable backend 不替代 native CPU；运行摘要必须明确报告实际 backend，不改变当前 native CPU/MPS 默认路径。

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

## 当前基础

`optional_dependency_status()` 在不导入重依赖的情况下报告 optional extras。
`portable_setup_instructions()` 返回可复制安装命令，不使用交互输入。
`choose_portable_backend(...)` 记录选中的 policy 和实际 backend。`PortableEstimatorAdapter` 在 native path 上保持 `fit`、`predict`、`predict_proba`、`score`、`get_params` 和 `set_params`。当用户显式选择 external policy 但当前 estimator 没有激活对应 external runtime 时，adapter 会 warning，记录 requested/effective backend，并通过 native CPU compatibility path 保持 workflow 可运行。
