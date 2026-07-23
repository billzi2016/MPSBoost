# MPSBoost 0.5.0 Known-Issue 审计

本文件记录 0.5.0 加固门。它不是 `1.0.0` 稳定承诺；它是 0.x release line 当前 known issue 清理记录。

## Blocking Customer-Facing Issue

在 0.5.0 portable-backend fallback 加固后，仓库当前没有记录 blocking customer-facing issue。

## 显式边界

- Native CPU/MPS 仍然是 MPSBoost 实现和 correctness oracle。
- IsolationForest 和 pointwise ranking 是 CPU-suitable workflow；MPS 请求会 warning 并继续在 CPU 运行，因为这种 workload shape 更适合 CPU。
- 官方 SHAP TreeExplainer 执行仍然需要 semantic validation，MPSBoost 才能声明 official SHAP output。Approximate explanation 会继续明确标记为 approximate。
- Linux CPU 和 Linux CUDA 运行通过显式 external sklearn/XGBoost/CUDA policy。external runtime 内部失败属于外部依赖/环境问题；MPSBoost 在可能时保持 native CPU 可用。
- `1.0.0` 仍然要等最终客户承诺门完成。

## 客户侧 Fallback 检查

- 缺 MPS 环境：warning 并给出可复制 `xcode-select`、Metal toolchain、reinstall 和 diagnostic 命令。
- CPU-only worker 和 model selection：`MPSBOOST_SKIP_ENV_CHECK=1` 保持可用且非交互。
- 缺可选 SHAP/portable dependency：diagnostics 会报告可复制 extras 安装命令。
- 显式 external portable policy：adapter 记录 requested/effective backend，warning，并在 external runtime 未对当前 estimator 激活时通过 native CPU compatibility 继续运行。
