# 更新日志

本项目所有值得记录的变更都会写入本文件。

## 0.4.0 - 2026-07-23

- 增加显式 portable-backend 诊断、可选 `xgboost`、`sklearn`、`cuda` extras，以及可观测的 backend selection summary。
- 增加官方 SHAP 集成边界：`mpsboost[shap]`、native tree export payload，以及 TreeExplainer 语义验证门槛。
- 增加 MNIST、Titanic、Adult Income 和 HIGGS 的 opt-in 真实世界验收测试，下载产物只写入项目内已忽略 cache，不写入用户全局 cache。
- 增加 S18 真实世界验收报告，并加入文档站点导航。
- 保持 native CPU/MPS 作为默认实现和 correctness oracle；外部 backend 仍然显式、可观测、可选。

## 0.3.0 - 2026-07-20

- 增加 sklearn 风格二分类和多分类 classifier，包括 native CPU softmax 和显式 OvR compatibility。
- 在共享 native tree engine 上增加 decision tree、random forest、ExtraTrees 和 CatBoost-like numeric estimator family。
- 增加 quantile、Poisson 和 Tweedie 回归目标，并支持 model-format round trip。
- 增加 gain、split-count、permutation 和受控 SHAP-like explanation helper。
- 增加适合 CPU 的 isolation forest 和 pointwise learning-to-rank estimator，并提供可观测 backend routing。
- 增加导入期 MPS 环境提示，提供可复制 setup 命令，并为 CPU-only worker 提供 `MPSBOOST_SKIP_ENV_CHECK=1`。
- 扩展真实世界数据集 smoke 覆盖和 0.x release line 的 CPU baseline 记录。

## 0.2.0 - 2026-07-19

- 将 preview mock API 替换为真实编译的 MPS/Metal backend foundation。
- 增加 native device diagnostics 和真实 GPU vector smoke kernel。
- 在 package build time 构建 Metal shader，并打包生成的 library。
- 为 strided float32/float64 buffer 增加 deterministic native quantization。
- 增加 compact uint8/uint16 feature-major bins 和经过验证的 serialization。
- 拒绝 invalid values、unsupported layouts、corrupted data 和 size overflows。
- 增加真实 MPS split-scan 和 stable partition/compaction kernel。
- 增加 histogram subtraction 和 layer-aware training hook。
- 为 MPS histogram 与 hot-path workspace 增加 L1 temporary buffer pool。
- 增加 S6 end-to-end regressor benchmark，记录 small-data regressions 和 large-wide wins。
- 增加 versioned L2 cache keys、atomic writes、checksum validation 和 damaged-cache invalidation。
- 增加 public cache diagnostics、explicit cache directory creation 和 safe cache clearing。
- 收紧 CI cache 边界，确保 self-hosted GPU job 不写入或上传 user-level pip cache。

## 0.2.0rc0 - 2026-07-19

- 发布 cache and stability release candidate。

## 0.2.0b0 - 2026-07-19

- 发布 GPU hot-path beta milestone。

## 0.2.0a0 - 2026-07-19

- 发布第一个 functional MPS regressor alpha。

## 0.1.0a0 - 2026-07-19

- 预留 `mpsboost` package namespace。
- 发布计划中的 sklearn/XGBoost-style Python API surface。
- 增加 backend diagnostics 和 layered cache-path definitions。
- 对尚未交付的 training operation 明确失败。
