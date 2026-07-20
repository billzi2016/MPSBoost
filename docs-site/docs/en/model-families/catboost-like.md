# CatBoost-like Numeric Estimators

MPSBoost provides CatBoost-like numeric estimator entries:

- `CatBoostRegressor`
- `CatBoostClassifier`

The current focus is an ordered-boosting-style sklearn-compatible public API and numeric feature
workflow.

Supported capabilities:

- ordered boosting permutation semantics
- numeric feature training
- categorical feature parameter compatibility entries
- shared native tree engine
- classifier / regressor API
- sample weight
- regularization controls

Limitations:

- Categorical model persistence remains future model-format work.
- MPSBoost does not call CatBoost as a hidden training engine.
