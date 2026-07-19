# MPSBoost

[![PyPI](https://img.shields.io/pypi/v/mpsboost)](https://pypi.org/project/mpsboost/)
[![Python](https://img.shields.io/pypi/pyversions/mpsboost)](https://pypi.org/project/mpsboost/)

MPSBoost is an early-stage gradient boosting project for Apple Silicon. Its current accelerated backend uses custom Metal compute kernels for squared-error gradients and two-stage histogram construction while keeping one deterministic tree-building implementation shared with the CPU oracle.

> **Development status:** `0.2.0` is the first stable 0.x MPS histogram engine release. It supports dense numeric regression with a real MPS training path, split-scan and partition kernels, histogram subtraction, buffer reuse, explicit cache management, and documented performance boundaries.

## Project origin

MPSBoost was started by a Purdue CS PhD student working across algorithms, systems, AI, compilers, and formal verification. The immediate motivation is practical: Apple Silicon has a strong GPU stack, but common tree-based machine learning workloads still lack a simple, fast, low-permission MPS-accelerated path.

The project is built with an SDD workflow. Work is split into clear stages: first validate real MPS/Metal kernels and runtime behavior, then lock the specs and product requirements, then settle the technical stack, and finally execute the task list against those specs. Specs are treated as the project contract, not as after-the-fact notes.

For AI agents and automation, [mps_boost_skill.md](mps_boost_skill.md) is the canonical usage
entry point. It describes the complete target usage pattern, import style, estimator names,
backend policy, sklearn model selection, model persistence, diagnostics, and implementation
constraints.

## Installation

```bash
python -m pip install mpsboost
```

Accelerated releases provide prebuilt Apple Silicon wheels; normal users will not need a heavyweight framework, package manager, CMake, or a local Metal shader compiler.

## Estimator-style API

```python
import mpsboost as mb

model = mb.GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    device="mps",
)

model.fit(X_train, y_train)
prediction = model.predict(X_test)
importance = model.feature_importances_
model.save_model("model.mb")

restored = mb.GradientBoostingRegressor(device="mps")
restored.load_model("model.mb")
```

`MPSBoostRegressor` remains available as a backwards-compatible project-branded alias for
the same implementation.

## sklearn model selection

MPSBoost estimators are designed to follow the sklearn estimator protocol, so users should be
able to use the standard sklearn model-selection stack instead of learning a project-specific
search API.

```python
from sklearn.model_selection import GridSearchCV
import mpsboost as mb

search = GridSearchCV(
    mb.GradientBoostingRegressor(device="auto"),
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

The same direction applies to `RandomizedSearchCV`, `cross_val_score`, and future classifier
estimators. The current regressor exposes `get_params()`, `set_params()`, `fit()`, `predict()`,
and the default regression `score()` method needed by standard sklearn search utilities.

Multiprocessing is supported through sklearn/joblib at the outer search level. CPU jobs can run
in multiple processes. MPS jobs should be scheduled more conservatively: several Python workers
competing for one Apple GPU can be slower than one well-sized GPU worker because command queues,
unified memory bandwidth, and synchronization overhead become the bottleneck. The intended policy
is to let `device="auto"` choose CPU for small search jobs and reserve MPS for workloads where the
measured tree hot path is large enough.

## Tree estimator names

The primary public API uses concise sklearn-style estimator names, so users can usually switch
libraries by changing the import and keeping familiar model names.

| Model family | Primary names | Status |
| --- | --- | --- |
| Histogram gradient boosting | `GradientBoostingRegressor` | Available |
| Histogram gradient boosting classification | `GradientBoostingClassifier` | Available for strict binary 0/1 labels |
| Random forest | `RandomForestRegressor`, `RandomForestClassifier` | Planned |
| Extra trees | `ExtraTreesRegressor`, `ExtraTreesClassifier` | Planned |
| Single decision tree | `DecisionTreeRegressor`, `DecisionTreeClassifier` | Planned |
| CatBoost-like ordered boosting | `CatBoostRegressor`, `CatBoostClassifier` | Planned |
| Isolation forest | `IsolationForest` | Planned |
| Ranking trees | `LearningToRankRegressor` | Planned |

Objective variants such as quantile, Poisson, Tweedie, logistic, and ranking losses should be
selected through estimator parameters when they share the same tree engine. Separate class names
are added only when the model family has different fit/predict semantics.

The same status information is available from Python:

```python
import mpsboost as mb

print(mb.available_estimators())
print(mb.planned_estimators())
mb.require_estimator_supported("RandomForestRegressor")  # raises NotImplementedError today
```

The `0.3.0` milestone is reserved for the v2 arboretum foundation: one shared tree-family
registry, unified semantics for boosting/bagging/random-split models, planned estimator names,
and early-failure behavior for models that are not implemented yet. It should not be released as
a pile of placeholder classes.

Random forest row sampling, feature subsampling, ExtraTrees random thresholds, and CatBoost-like
ordered permutations share one deterministic randomization contract. This keeps planned model
families aligned before their MPS training kernels are exposed as public estimators.

Validation metric history and early stopping also share one estimator-independent monitoring
contract, so future classifiers and tree ensembles do not duplicate callback semantics.

## CPU and MPS backends

MPSBoost treats CPU as a first-class backend, not as a temporary fallback. The CPU oracle/backend
is implemented inside this project and shares the same quantization, objective, sampling,
monitoring, split-gain, and model-format contracts as the MPS backend. MPSBoost does not call
XGBoost, LightGBM, CatBoost, or scikit-learn as hidden training engines.

The intended long-term policy is:

- `device="cpu"` forces the in-project CPU backend.
- `device="mps"` forces the in-project MPS/Metal backend.
- `device="auto"` chooses CPU for small or synchronization-heavy workloads and MPS for
  workloads where the measured tree hot path can dominate transfer and launch overhead.

MPS is an acceleration backend, not a requirement. If CPU is faster or more stable for a given
workload, the project should say so and use CPU under `auto`.

## Backend diagnostics

The native backend exposes non-sensitive device and cache diagnostics:

```python
import mpsboost as mb

print(mb.__version__)
print(mb.is_available())
print(mb.system_info())
print(mb.cache_info())
```

`cache_info()` only reports paths and existence; it does not create directories. `create_cache()`
explicitly creates the L2 cache directories, and `clear_cache()` safely removes the MPSBoost cache
root after rejecting dangerous targets such as the filesystem root, the user home directory, or
symlinks. Cache deletion never changes model results.

## Project principles

- Familiar XGBoost/scikit-learn-style Python entry points.
- `device="mps"` as the stable user-facing Apple GPU backend name.
- Custom Metal kernels for tree-specific irregular computation.
- Prebuilt wheels and no heavyweight Python runtime dependency.
- Explicit errors instead of silent CPU fallback.
- End-to-end benchmarks, including preprocessing and synchronization.

## Status

The public API currently includes `GradientBoostingRegressor`, `GradientBoostingClassifier`,
their backwards-compatible project-branded aliases, the estimator capability registry,
deterministic randomization and monitoring helpers, cache diagnostics and management helpers,
`is_available`, `system_info`, and `__version__`. Training supports dense finite
`float32`/`float64`-compatible data, squared error regression, strict binary-logistic
classification for labels `0` and `1`, deterministic quantization, depth-limited histogram trees,
sklearn-compatible `score()`, model save/load, gain/split feature importance, explicit
`device="mps"`, explicit `device="cpu"`, and initial `device="auto"` selection.

The checked-in S6 benchmark records both regressions and wins. On the M2 Ultra validation machine, small end-to-end training remains slower on MPS, while the `gbdt-large-wide` scenario reached a 1.629x median speedup with maximum prediction difference around `5.4e-6` versus the CPU oracle.

Classification, missing values, sparse matrices, categorical features, sampling, early stopping, public GPU prediction, and full third-party API compatibility are not implemented in this milestone. Small datasets are expected to be slower on the GPU because fixed device setup and synchronization costs dominate; the checked-in benchmark report preserves this regression region alongside larger wins.

## Release audits

The `0.2.0` release gate includes:

- CPU, packaging, integration, and real Metal GPU tests on Python 3.10 and 3.13.
- Wheel content checks excluding specs, tests, caches, and build artifacts.
- Dynamic-link checks for the native extension.
- Apache-2.0 project licensing and runtime dependency review.
- Fresh PyPI installation and real MPS smoke validation.

## Independence notice

MPSBoost is an independent open-source project. It is not affiliated with, endorsed by, or
sponsored by Apple Inc., the XGBoost project, the LightGBM project, the CatBoost project, or the
scikit-learn project. The MPS/Metal backend is an independent implementation based on public
papers, public API documentation, and original engineering work; it is not derived from those
libraries. Apple, Metal, and Metal Performance Shaders may be trademarks of Apple Inc.

## License

Apache-2.0
