# Forests and Single Trees

MPSBoost forest and single-tree estimators share the native tree format, sampling semantics, and
prediction aggregation logic.

## Single trees

- `DecisionTreeRegressor`
- `DecisionTreeClassifier`

Single trees are useful for debugging, baselines, lightweight tasks, and forest composition.

## Random Forest

- `RandomForestRegressor`
- `RandomForestClassifier`

Core semantics:

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

ExtraTrees uses random threshold candidates and reuses native split, tree, prediction, and forest
container logic.
