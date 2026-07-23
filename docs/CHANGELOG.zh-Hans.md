# 更新日志

本项目所有值得记录的变更都会写入本文件。

## 1.0.0 - 2026-07-23

- 发布 stable customer-commitment release，承接 0.x 的功能、验证和加固线。
- 冻结公开范围：native CPU/MPS tree estimator、显式可选 backend 诊断、项目内真实世界数据 cache，以及客户侧环境提示。
- 增加最终 release、known-issue 和 performance report，并区分 current HEAD 证据与 v2/v3 前历史 benchmark baseline。
- HIGGS 保持为显式 large local-file performance-boundary dataset，不把 multi-gigabyte raw data 打包或自动下载进用户环境。
- 关闭 release audit 前必须完成 fresh PyPI install verification 和 versioned artifact hash 记录。

## 0.5.0 - 2026-07-23

- 加固 portable backend 行为：用户显式选择 external backend policy 时，adapter warning 后通过 native CPU compatibility path 继续运行，而不是中断 workflow。
- 在 adapter training summary 中同时记录 requested 和 effective portable backend decision。
- 增加到 `0.5.0` 的版本化 release audit 和 release page，同时保留更早 release history。
- 将长文档移动到 `docs/`，AI 使用规范移动到 `ai-skills/`，根目录保留 README 作为入口。
- 增加 0.5.0 known-issue audit，作为未来 `1.0.0` 承诺前的交接门。

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
