# CatBoost-like Numeric Estimators

`CatBoostRegressor` and `CatBoostClassifier` provide ordered-boosting-style,
sklearn-compatible numeric workflows on the shared native tree engine. Categorical
feature parameters are compatibility entries; categorical model persistence remains
future work. MPSBoost does not call CatBoost as a hidden training engine.
