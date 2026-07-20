# Estimator API

当前 `0.3.0` 公开 estimator 包括：

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

这些 estimator 遵循 sklearn 风格入口：

- `fit`
- `predict`
- `predict_proba`，适用于 classifier
- `score`
- `get_params`
- `set_params`

高级回归目标通过参数选择：

- `loss="squared_error"`
- `loss="quantile"`
- `loss="poisson"`
- `loss="tweedie"`
