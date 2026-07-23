# Estimator API

Public `0.4.0` estimators cover gradient boosting, single decision trees, random forests,
ExtraTrees, CatBoost-like numeric estimators, IsolationForest, and learning-to-rank entries. They
follow the sklearn-style protocol, so normal training, prediction, scoring, and parameter search
can use one method set.

Public names include:

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

These estimators follow these sklearn-style entries:

- `fit`
- `predict`
- `predict_proba`, for classifiers
- `score`
- `get_params`
- `set_params`

Advanced regression objectives are selected through `loss`:

- `loss="squared_error"`
- `loss="quantile"`
- `loss="poisson"`
- `loss="tweedie"`

Classifiers additionally support probability outputs. Multiclass defaults to the native softmax
path when available; OvR is only an explicit compatibility strategy or staged fallback.
`device="auto"` chooses CPU or MPS according to environment and workload, and the training summary
records the actual backend so users do not need to guess which path ran.
