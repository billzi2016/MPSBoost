# MPSBoost 0.3.0 发布审计

本文件记录 `0.3.0` v2 arboretum 里程碑的发布门。

## 范围

`0.3.0` 支持：

- dense numeric regression
- binary and multiclass classification
- squared-error objective
- quantile、Poisson、Tweedie regression objectives
- deterministic quantization
- depth-limited histogram trees
- DecisionTree、RandomForest、ExtraTrees、CatBoost-like numeric estimators
- native CPU multiclass softmax with explicit OvR compatibility
- CPU-suitable isolation forest anomaly scoring
- pointwise learning-to-rank scoring with query-group validation
- real MPS gradient、histogram、split-scan、partition 和 buffer-pool paths
- explicit CPU oracle mode
- model save/load
- feature importance、permutation importance 和 controlled SHAP-like explanations
- import-time MPS environment guidance
- cache diagnostics、explicit cache creation 和 safe cache clearing

不包含：

- sparse matrices
- native MPS multiclass softmax
- official third-party SHAP TreeExplainer integration
- categorical model persistence
- public GPU prediction
- full third-party API compatibility

## Artifact

- wheel: `mpsboost-0.3.0-cp313-cp313-macosx_13_0_arm64.whl`
- sdist: `mpsboost-0.3.0.tar.gz`

发布前必须完成 `twine check`、wheel 内容审计、动态链接审计、临时环境安装 smoke 和正式 PyPI 安装复验。
