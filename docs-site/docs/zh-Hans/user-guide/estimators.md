# Estimator API

当前 `0.5.0` 公开的 estimator 覆盖梯度提升、单棵决策树、随机森林、ExtraTrees、CatBoost-like numeric estimator、IsolationForest 和 learning-to-rank 入口。它们都遵循 sklearn 风格协议，因此常规训练、预测、评分和参数搜索可以使用同一套方法。

公开名称包括：

- `GradientBoostingRegressor`
- `GradientBoostingClassifier`
- `MPSBoostRegressor`
- `MPSBoostClassifier`
- `DecisionTreeRegressor`
- `DecisionTreeClassifier`
- `RandomForestRegressor`
- `RandomForestClassifier`
- `ExtraTreesRegressor`
- `ExtraTreesClassifier`
- `ExtraTreeRegressor`
- `ExtraTreeClassifier`
- `CatBoostRegressor`
- `CatBoostClassifier`
- `IsolationForest`
- `MPSIsolationForest`
- `LearningToRankRegressor`

这些 estimator 遵循以下 sklearn 风格入口：

- `fit`
- `predict`
- `predict_proba`，适用于 classifier
- `score`
- `get_params`
- `set_params`

高级回归目标通过 `loss` 参数选择：

- `loss="squared_error"`
- `loss="quantile"`
- `loss="poisson"`
- `loss="tweedie"`

Classifier 额外支持概率输出。多分类默认走 native softmax 可用路径；OvR 只作为显式兼容策略或阶段性 fallback。`device="auto"` 会根据环境和工作负载选择 CPU 或 MPS，training summary 会记录实际后端，用户不需要猜当前运行在哪条路径上。
