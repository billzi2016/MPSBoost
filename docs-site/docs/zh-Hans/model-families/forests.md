# Random Forest、ExtraTrees 与单树

MPSBoost 的 forest 和 single-tree estimator 共享 native tree format、采样语义和预测聚合逻辑。

## 单树

- `DecisionTreeRegressor`
- `DecisionTreeClassifier`

单树用于调试、基线、轻量任务和 forest 组合。

## Random Forest

- `RandomForestRegressor`
- `RandomForestClassifier`

核心语义：

- bootstrap / sample fraction
- feature subsampling
- independent native trees
- regression mean aggregation
- classification vote/probability aggregation
- deterministic `random_state`
- deterministic `n_jobs` scheduling

## ExtraTrees

- `ExtraTreesRegressor`
- `ExtraTreesClassifier`
- `ExtraTreeRegressor`
- `ExtraTreeClassifier`

ExtraTrees 使用随机 threshold candidates，并复用 native split、tree、prediction 和 forest container 逻辑。
