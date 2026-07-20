# CatBoost-like Numeric Estimators

MPSBoost 提供 CatBoost-like numeric estimator 入口：

- `CatBoostRegressor`
- `CatBoostClassifier`

当前重点是 ordered-boosting 风格的 sklearn-compatible public API 和 numeric feature 工作流。

支持能力：

- ordered boosting permutation semantics
- numeric feature training
- categorical feature 参数兼容入口
- shared native tree engine
- classifier / regressor API
- sample weight
- regularization controls

限制：

- categorical model persistence 仍由后续模型格式任务处理。
- 不调用 CatBoost 作为隐藏训练引擎。
