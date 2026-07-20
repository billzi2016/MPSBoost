# Estimator API

Public `0.3.0` estimators include gradient boosting, decision trees, random forests,
ExtraTrees, CatBoost-like numeric estimators, IsolationForest, and learning to rank.
All follow the sklearn-style `fit`, `predict`, `score`, `get_params`, and `set_params`
protocol; classifiers also support `predict_proba`. Regression objectives use
`loss="squared_error"`, `"quantile"`, `"poisson"`, or `"tweedie"`.
