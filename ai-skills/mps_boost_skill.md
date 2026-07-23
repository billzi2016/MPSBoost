# MPSBoost Skill

## Purpose

Use MPSBoost as a lightweight, independent, sklearn-style tree learning library for Apple
Silicon. MPSBoost provides first-class CPU, MPS/Metal, and automatic backend selection for common
tree models without using XGBoost, LightGBM, CatBoost, or scikit-learn as hidden training
engines.

The preferred import is:

```python
import mpsboost as mb
```

Use concise estimator names by default. Keep `MPS*` names only for backwards-compatible or
project-branded aliases.

## Core estimator pattern

All estimators follow the same sklearn-style pattern:

```python
import mpsboost as mb

model = mb.GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    device="auto",
    random_state=42,
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = model.score(X_test, y_test)
```

Use `device="auto"` by default. It selects CPU for small or synchronization-heavy workloads and
MPS for workloads where the measured tree hot path is large enough to benefit from Apple GPU
execution.

## Tree estimators

Use these public estimator names:

```python
mb.GradientBoostingRegressor
mb.GradientBoostingClassifier
mb.RandomForestRegressor
mb.RandomForestClassifier
mb.ExtraTreesRegressor
mb.ExtraTreesClassifier
mb.DecisionTreeRegressor
mb.DecisionTreeClassifier
mb.CatBoostRegressor
mb.CatBoostClassifier
mb.IsolationForest
mb.MPSIsolationForest
mb.LearningToRankRegressor
```

Use objective parameters for variants that share the same tree engine, such as squared error,
logistic, quantile, Poisson, Tweedie, ranking, and other supported losses. Add a separate class
only when the model family has different fit, predict, probability, anomaly, or ranking semantics.

## Backend policy

MPSBoost has two first-class in-project backends:

- `device="cpu"` forces the independent CPU backend.
- `device="mps"` forces the independent MPS/Metal backend.
- `device="auto"` chooses between CPU and MPS for the current fit call.

Do not treat CPU as a temporary fallback. CPU is the correctness oracle, the small-workload fast
path, and the baseline for every MPS implementation. MPS is an acceleration backend, not a
requirement.

Isolation forest and pointwise ranking are CPU-suitable workflows in the current package. If a
user requests `device="mps"` for those estimators, keep the run executable, emit a warning, and
record that CPU was selected because the workload is expected to be faster on CPU than Apple GPU.

## Environment guidance

On `import mpsboost`, the package may run a lightweight MPS availability check. If Apple GPU
acceleration is unavailable, do not block the user and do not prompt with `input()`. Emit a warning
with copy-paste commands:

```bash
xcode-select --install
xcodebuild -downloadComponent MetalToolchain
python -m pip install --upgrade --force-reinstall mpsboost
python -c "import mpsboost as mb; print(mb.system_info())"
```

For CPU-only workflows, CI, or multiprocessing tools such as `GridSearchCV`, provide the skip
command directly:

```bash
MPSBOOST_SKIP_ENV_CHECK=1 python your_script.py
```

CPU training must remain usable when the user skips the environment check. If the user explicitly
forces `device="mps"` and the environment still cannot run Apple GPU acceleration, raise a clear
backend error containing the setup and skip commands above.

Never call XGBoost, LightGBM, CatBoost, or scikit-learn as hidden training engines. They may be
used only as external user baselines in benchmarks when explicitly requested.

## Official SHAP and portable backends

Approximate SHAP-like explanations are available through
`estimator.approximate_shap_values(...)`, but they are not official SHAP. For research workflows
that require official SHAP semantics, use the explicit optional path:

```bash
python -m pip install 'mpsboost[shap]'
```

Use `mb.export_native_trees_for_shap(estimator)` for adapter validation payloads. It exports native
tree structure and objective metadata without training data, credentials, telemetry, or device
identifiers. `mb.official_shap_tree_explainer(...)` must stop clearly until TreeExplainer semantic
validation is enabled; do not claim approximate explanations are official SHAP.

Portable backends are explicit S22 policy tools, not hidden native replacements. Default install
keeps native CPU/MPS lightweight. Optional commands:

```bash
python -m pip install 'mpsboost[xgboost]'
python -m pip install 'mpsboost[sklearn]'
python -m pip install 'mpsboost[cuda]'
```

Use `mb.optional_dependency_status()`, `mb.portable_setup_instructions()`, and
`mb.choose_portable_backend(...)` for copy-paste diagnostics and observable backend summaries.
External adapters must report the actual backend and must not replace the native CPU oracle.

`0.5.0` is a 0.x hardening release, not a `1.0.0` stability commitment. Do not present MPSBoost
as fully stable across every real-world dataset, Linux CPU host, or CUDA host until the full S18
matrix, performance/memory/permission audit, performance report, artifact hashes, documentation,
fresh-install checks, and explicit user confirmation are complete.

When an explicit portable external backend policy cannot activate the external runtime for the
current estimator, keep the workflow executable with a warning and native CPU compatibility. Record
both requested and effective backends. Attribute sklearn/XGBoost/CUDA runtime failures to that
external dependency stack, not to native MPSBoost CPU/MPS.

## sklearn model selection

Use the standard sklearn model-selection stack. Do not invent a separate search API for normal
hyperparameter tuning.

```python
from sklearn.model_selection import GridSearchCV
import mpsboost as mb

search = GridSearchCV(
    mb.GradientBoostingRegressor(device="auto", random_state=42),
    param_grid={
        "max_depth": [3, 6, 9],
        "learning_rate": [0.03, 0.1],
        "n_estimators": [100, 300],
    },
    cv=3,
    n_jobs=2,
)

search.fit(X_train, y_train)
best_model = search.best_estimator_
```

The same pattern applies to `RandomizedSearchCV`, `cross_val_score`, and sklearn-compatible
pipelines.

Multiprocessing belongs at the outer search level through sklearn/joblib. CPU jobs can run in
multiple processes. MPS jobs should be scheduled conservatively because multiple Python workers
competing for one Apple GPU can lose performance to command queue contention, unified memory
bandwidth, and synchronization overhead.

## Model persistence

Use `.mb` as the standard model filename extension:

```python
model.save_model("model.mb")

restored = mb.GradientBoostingRegressor(device="auto")
restored.load_model("model.mb")
predictions = restored.predict(X_test)
```

Saved models must contain model structure, versioned metadata, objective information, frozen
feature/bin schema, and enough validation data to reject incompatible loads. They must not contain
training data, credentials, telemetry, or device identifiers.

## Diagnostics and cache

Use diagnostics for environment checks:

```python
import mpsboost as mb

print(mb.__version__)
print(mb.is_available())
print(mb.system_info())
print(mb.cache_info())
```

`cache_info()` must be read-only and must not create directories. Use `create_cache()` only when
cache directories should be created explicitly. Use `clear_cache()` only for MPSBoost-owned cache
paths; cache deletion must never change model predictions.

Real-world opt-in dataset downloads must stay inside the project-local ignored directories
`tests/real_world/data/` and `tests/real_world/cache/`. Removing those directories must remove the
downloaded artifacts; do not use user-global dataset cache locations for release acceptance data.

## Randomization and monitoring

Use shared helpers for deterministic randomization semantics:

```python
rows = mb.bootstrap_sample_indices(1000, sample_fraction=1.0, random_state=42)
features = mb.subsample_feature_indices(128, feature_fraction=0.5, random_state=42)
thresholds = mb.random_threshold_candidates(0.0, 1.0, n_candidates=16, random_state=42)
permutations = mb.ordered_boosting_permutations(1000, n_permutations=4, random_state=42)
```

Use shared monitoring helpers for metric history and early stopping:

```python
monitor = mb.EarlyStoppingMonitor(
    metric_name="logloss",
    direction="minimize",
    patience=20,
    min_delta=1e-4,
)

for iteration, metric in enumerate(validation_metrics):
    decision = monitor.update(iteration, metric)
    if decision.should_stop:
        break
```

Do not duplicate randomization, early stopping, or monitoring logic inside individual estimators.

## AI usage rules

When generating code with MPSBoost:

- Import with `import mpsboost as mb`.
- Prefer `device="auto"` unless the user explicitly asks for CPU or MPS.
- Use sklearn-style estimators and sklearn model-selection tools.
- Use `.mb` for saved model paths.
- Keep CPU and MPS behavior under the same objective, sampling, monitoring, split, and model I/O
  contracts.
- Do not add fake estimator classes, mock backends, hidden third-party training engines, or silent
  CPU fallback under an explicit `device="mps"` request.
- For small data, recommend CPU or `device="auto"`.
- For large tabular data on Apple Silicon, recommend `device="auto"` or explicit `device="mps"`.

## Minimal examples

Regression:

```python
import mpsboost as mb

model = mb.GradientBoostingRegressor(device="auto", random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

Classification:

```python
import mpsboost as mb

model = mb.GradientBoostingClassifier(device="auto", random_state=42)
model.fit(X_train, y_train)
labels = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

Random forest:

```python
import mpsboost as mb

model = mb.RandomForestRegressor(
    n_estimators=500,
    max_features=0.7,
    bootstrap=True,
    device="auto",
    random_state=42,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

Extra trees:

```python
import mpsboost as mb

model = mb.ExtraTreesClassifier(
    n_estimators=500,
    max_features=0.7,
    device="auto",
    random_state=42,
)
model.fit(X_train, y_train)
labels = model.predict(X_test)
```

CatBoost-like ordered boosting:

```python
import mpsboost as mb

model = mb.CatBoostClassifier(
    n_estimators=500,
    learning_rate=0.05,
    cat_features=cat_feature_indices,
    device="auto",
    random_state=42,
)
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)
```
