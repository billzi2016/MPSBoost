# MPSBoost 0.2.0a0 发布审计

`0.2.0a0` 是第一个 functional MPS regressor alpha。它把纯规划 API 替换为 compiled native extension，并加入第一条真实 regression training path。

包含：

- compiled native package foundation；
- device diagnostics 和 MPS availability check；
- deterministic dense-data validation 和 quantization 基础；
- 第一个 functional regression estimator path。

不包含：

- 优化后的 GPU hot path；
- cache stability guarantee；
- 正式 0.2.0 发布验证；
- 早期 regressor 之外的 tree-family breadth。
