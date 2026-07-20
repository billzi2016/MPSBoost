# Anomaly Detection and Ranking

`IsolationForest` / `MPSIsolationForest` provide random isolation trees, path length,
anomaly scores, `score_samples`, `decision_function`, and `predict`. `LearningToRankRegressor`
supports group/query contracts, pointwise ranking, NDCG, and sklearn parameters.
Both are currently CPU-suitable; an MPS request warns and records the CPU decision.
