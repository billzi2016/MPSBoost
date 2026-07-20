# 特征重要性与解释

MPSBoost 提供多层解释能力：

- gain importance
- split-count importance
- permutation importance
- controlled SHAP-like approximate explanations

## 设计原则

- 不复制 prediction 或 scoring 逻辑。
- 不把近似解释声明为官方 SHAP。
- 官方 SHAP TreeExplainer adapter 作为单独任务推进。

## 使用方向

科研或报告中需要官方 SHAP 语义时，应等待 S16.5a / S23 文档补齐 adapter 说明。当前可使用 `permutation_importance` 作为模型无关解释。
