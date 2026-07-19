# MPSBoost

[![PyPI](https://img.shields.io/pypi/v/mpsboost)](https://pypi.org/project/mpsboost/)
[![Python](https://img.shields.io/pypi/pyversions/mpsboost)](https://pypi.org/project/mpsboost/)

MPSBoost is an early-stage gradient boosting project for Apple Silicon. Its planned execution backend combines reusable Metal Performance Shaders capabilities with custom Metal compute kernels for histogram construction, split evaluation, row partitioning, and prediction.

> **Technology preview:** version `0.1.0a0` reserves the package namespace and exposes the planned Python API. Model training is not implemented in this release. Calls to `fit()` and `train()` fail explicitly instead of silently running a different backend.

## Installation

```bash
python -m pip install mpsboost
```

The placeholder release is pure Python and has no runtime dependencies. Future accelerated releases will provide prebuilt Apple Silicon wheels; normal users will not need PyTorch, Homebrew, CMake, or a local Metal shader compiler.

## Planned sklearn-style API

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

## Planned native API

```python
import mpsboost as mps

dtrain = mps.MPSMatrix(X_train, label=y_train)
booster = mps.train(
    params={"objective": "reg:squarederror", "device": "mps"},
    dtrain=dtrain,
    num_boost_round=200,
)
```

## Preview diagnostics

The placeholder package can be used to inspect its public version and planned backend status:

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

The current release is a namespace and API placeholder. Do not use it for model training. Follow the repository for implementation milestones.

## Independence notice

MPSBoost is an independent open-source project. It is not affiliated with, endorsed by, or sponsored by Apple Inc., the XGBoost project, or the scikit-learn project. Apple, Metal, and Metal Performance Shaders may be trademarks of Apple Inc.

## License

Apache-2.0
