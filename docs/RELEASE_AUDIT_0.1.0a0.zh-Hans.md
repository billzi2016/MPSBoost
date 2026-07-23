# MPSBoost 0.1.0a0 发布审计

`0.1.0a0` 用于预留 `mpsboost` package namespace，并发布计划中的 public API surface。它是项目发现和依赖 metadata 用的 alpha 占位版本，不是功能性训练版本。

包含：

- package namespace 预留；
- 初始 sklearn/XGBoost 风格 API 方向；
- backend diagnostics 和 cache-path 规划；
- 对 alpha 范围外 training operation 明确失败。

不包含：

- 真实训练；
- 真实 MPS kernel；
- 模型持久化；
- 性能声明。
