# MPSBoost

[![PyPI](https://img.shields.io/pypi/v/mpsboost)](https://pypi.org/project/mpsboost/)
[![Python](https://img.shields.io/pypi/pyversions/mpsboost)](https://pypi.org/project/mpsboost/)

MPSBoost is an early-stage gradient boosting project for Apple Silicon. Its execution backend combines reusable Metal Performance Shaders capabilities with custom Metal compute kernels for histogram construction, split evaluation, row partitioning, and prediction.

> **Development status:** the published `0.1.0a0` release reserves the package namespace. The current `0.2.0a0` source tree contains the first real native backend foundation; model training will only be exposed after it satisfies the correctness and packaging requirements.

## Installation

```bash
python -m pip install mpsboost
```

Accelerated releases provide prebuilt Apple Silicon wheels; normal users will not need a heavyweight framework, package manager, CMake, or a local Metal shader compiler.

## Target estimator-style API

```python
import mpsboost as mps

model = mps.MPSBoostRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    device="mps",
)

model.fit(X_train, y_train)
prediction = model.predict(X_test)
```

## Backend diagnostics

The native backend exposes non-sensitive device diagnostics:

```python
import mpsboost as mps

print(mps.__version__)
print(mps.is_available())
print(mps.system_info())
```

## Project principles

- Familiar XGBoost/scikit-learn-style Python entry points.
- `device="mps"` as the stable user-facing Apple GPU backend name.
- Custom Metal kernels for tree-specific irregular computation.
- Prebuilt wheels and no heavyweight Python runtime dependency.
- Explicit errors instead of silent CPU fallback.
- End-to-end benchmarks, including preprocessing and synchronization.

## Status

Model training is not yet part of the public API. The repository does not ship mock estimators or pretend that unfinished training is available.

## Independence notice

MPSBoost is an independent open-source project. It is not affiliated with, endorsed by, or sponsored by Apple Inc., the XGBoost project, or the scikit-learn project. Apple, Metal, and Metal Performance Shaders may be trademarks of Apple Inc.

## License

Apache-2.0
