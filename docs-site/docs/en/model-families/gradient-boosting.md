# Gradient Boosting

MPSBoost gradient boosting models use the in-project native tree engine. The CPU backend is the
correctness oracle, and the MPS backend targets histogram hot paths that suit Apple GPU execution.

## Regression

Primary entries:

- `GradientBoostingRegressor`
- `MPSBoostRegressor`

Supported capabilities:

- squared error
- quantile
- Poisson
- Tweedie
- sample weight
- monotonic constraints
- interaction constraints
- L1/L2 regularization
- leaf-wise / level-wise growth
- model save/load

## Classification

Primary entries:

- `GradientBoostingClassifier`
- `MPSBoostClassifier`

Supported capabilities:

- binary logistic
- native CPU multiclass softmax
- OvR compatibility strategy
- `predict_proba`
- `decision_function`
- sklearn model selection

Multiclass defaults to native softmax when it is available on CPU. MPS requests use the observable
compatibility strategy and report the actual strategy in the training summary.
