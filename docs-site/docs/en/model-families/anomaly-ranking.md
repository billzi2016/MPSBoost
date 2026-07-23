# Anomaly Detection and Ranking

MPSBoost `0.4.0` includes CPU-suitable anomaly detection and ranking entries.

## Isolation Forest

Entries:

- `IsolationForest`
- `MPSIsolationForest`

Semantics:

- random isolation trees
- path length
- anomaly score
- `score_samples`
- `decision_function`
- `predict`

This workload is branch-heavy, so CPU is currently expected to be more suitable than Apple GPU. If
the user passes `device="mps"`, MPSBoost warns, continues with the CPU backend, and records the
reason in `training_summary_`.

## Learning to Rank

Entry:

- `LearningToRankRegressor`

Semantics:

- group/query input contract
- pointwise ranking scorer
- full-list NDCG score
- sklearn-style parameter protocol

This workflow is latency-sensitive, so CPU is currently expected to be more suitable than Apple
GPU. A `device="mps"` request remains runnable and records the CPU backend decision.
