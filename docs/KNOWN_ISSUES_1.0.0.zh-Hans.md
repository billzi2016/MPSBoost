# MPSBoost 1.0.0 Known-Issue 审计

1.0.0 发布门没有记录 blocking customer-facing issue。

## 已接受边界

- HIGGS 保持为 opt-in local-file performance-boundary dataset。由于 raw dataset 对普通客户安装过大，它不会被打包或自动下载。
- Official SHAP execution 需要显式 `mpsboost[shap]` extra 和 semantic-validation path。Approximate explanation 继续标记为 approximate。
- Linux CPU/CUDA 运行使用显式 external sklearn/XGBoost/CUDA backend policy，并记录 requested/effective backend decision。
- 更适合 CPU 的 workload 在用户请求 MPS 时 warning 并继续使用 CPU。

## 客户体验要求

- 缺 Metal 或 MPS 环境：warning 并给出可复制 setup command。
- CPU-only worker：`MPSBOOST_SKIP_ENV_CHECK=1`。
- 缺 optional dependency：给出可复制 `pip install 'mpsboost[extra]'` command。
- External backend runtime issue：清晰归因到所选 external stack，同时在可能时保持 native CPU 可用。
