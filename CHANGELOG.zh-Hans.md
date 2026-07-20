# 更新日志

## 0.3.0 - 2026-07-20

- 增加 sklearn 风格二分类和多分类 classifier，包括 native CPU softmax 与显式 OvR compatibility。
- 增加 DecisionTree、RandomForest、ExtraTrees 和 CatBoost-like numeric estimator。
- 增加 quantile、Poisson、Tweedie 回归目标，并支持模型格式 round trip。
- 增加 gain、split-count、permutation 和受控 SHAP-like 解释 helper。
- 增加 CPU-suitable IsolationForest 和 pointwise LearningToRankRegressor，并记录可观测后端选择。
- 增加导入期 MPS 环境提示，提供可复制安装命令和 `MPSBOOST_SKIP_ENV_CHECK=1`。
- 扩展真实世界数据 smoke 覆盖和 CPU baseline 记录。

## 0.2.0 - 2026-07-19

- 将 preview mock API 替换为真实编译的 MPS/Metal 后端基础。
- 增加 native 设备诊断和真实 GPU vector smoke kernel。
- 构建期编译 Metal shader 并打包 metallib。
- 增加 deterministic native quantization。
- 增加 compact uint8/uint16 feature-major bins 和序列化校验。
- 增加真实 MPS split-scan、partition/compaction、histogram subtraction 和 buffer pool。
- 增加 S6 benchmark，记录小数据退化和 large-wide 加速。
- 增加 L2 cache key、atomic write、checksum validation 和 cache 管理 API。
