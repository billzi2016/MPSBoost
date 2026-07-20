# 异常检测与排序学习

MPSBoost `0.3.0` 包含 CPU-suitable 的异常检测和排序学习入口。

## Isolation Forest

入口：

- `IsolationForest`
- `MPSIsolationForest`

语义：

- random isolation trees
- path length
- anomaly score
- `score_samples`
- `decision_function`
- `predict`

该工作负载分支较多，当前预计 CPU 比 Apple GPU 更合适。用户传入 `device="mps"` 时，MPSBoost 会 warning 并继续使用 CPU backend，同时在 `training_summary_` 中记录原因。

## Learning to Rank

入口：

- `LearningToRankRegressor`

语义：

- group/query 输入契约
- pointwise ranking scorer
- full-list NDCG score
- sklearn-style parameter protocol

该工作流延迟敏感，当前预计 CPU 比 Apple GPU 更合适。请求 `device="mps"` 时保持可运行，并记录 CPU backend decision。
