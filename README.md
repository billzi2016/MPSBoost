# MPSBoost

[![PyPI](https://img.shields.io/pypi/v/mpsboost)](https://pypi.org/project/mpsboost/)
[![Python](https://img.shields.io/pypi/pyversions/mpsboost)](https://pypi.org/project/mpsboost/)

MPSBoost is an early-stage gradient boosting project for Apple Silicon. Its current accelerated backend uses custom Metal compute kernels for squared-error gradients and two-stage histogram construction while keeping one deterministic tree-building implementation shared with the CPU oracle.

> **Development status:** `0.2.0a0` is the first functional regressor milestone. It supports dense numeric regression with a real MPS training path and is still a pre-alpha release with a deliberately narrow feature set.

## Installation

```bash
python -m pip install mpsboost
```

Accelerated releases provide prebuilt Apple Silicon wheels; normal users will not need a heavyweight framework, package manager, CMake, or a local Metal shader compiler.

## Estimator-style API

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
model.save_model("model.mpsb")

restored = mps.MPSBoostRegressor(device="mps")
restored.load_model("model.mpsb")
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

The public API currently includes `MPSBoostRegressor`, `is_available`, `system_info`, and `__version__`. Training supports dense finite `float32`/`float64`-compatible data, squared error, deterministic quantization, depth-limited histogram trees, model save/load, and explicit `device="mps"` or diagnostic `device="cpu"` selection.

Classification, missing values, sparse matrices, categorical features, sampling, early stopping, GPU split scanning, GPU row partitioning, and GPU prediction are not implemented in this milestone. Small datasets are expected to be slower on the GPU because fixed device setup and synchronization costs dominate; the checked-in benchmark report preserves this regression region alongside larger wins.

## Independence notice

MPSBoost is an independent open-source project. It is not affiliated with, endorsed by, or sponsored by Apple Inc., the XGBoost project, or the scikit-learn project. Apple, Metal, and Metal Performance Shaders may be trademarks of Apple Inc.

## License

Apache-2.0
