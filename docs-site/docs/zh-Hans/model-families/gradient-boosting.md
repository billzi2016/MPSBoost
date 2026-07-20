# Gradient Boosting

MPSBoost 的梯度提升模型使用项目内 native tree engine。CPU backend 是 correctness oracle，MPS backend 用于适合 Apple GPU 的 histogram 热路径。

## 回归

主要入口：

- `GradientBoostingRegressor`
- `MPSBoostRegressor`

支持能力：

- squared error
- quantile
- Poisson
- Tweedie
- sample weight
- monotonic constraints
- interaction constraints
- L1/L2 正则
- leaf-wise / level-wise growth
- model save/load

## 分类

主要入口：

- `GradientBoostingClassifier`
- `MPSBoostClassifier`

支持能力：

- binary logistic
- native CPU multiclass softmax
- OvR compatibility strategy
- `predict_proba`
- `decision_function`
- sklearn model selection

多分类默认在 CPU 可用时选择 native softmax；MPS 请求会使用可观测 compatibility strategy，避免把 CPU softmax 伪装为 MPS native softmax。
