# MPSBoost

[![PyPI](https://img.shields.io/pypi/v/mpsboost)](https://pypi.org/project/mpsboost/)
[![Python](https://img.shields.io/pypi/pyversions/mpsboost)](https://pypi.org/project/mpsboost/)

MPSBoost is an early-stage gradient boosting project for Apple Silicon. Its current accelerated backend uses custom Metal compute kernels for squared-error gradients and two-stage histogram construction while keeping one deterministic tree-building implementation shared with the CPU oracle.

> **Development status:** `0.2.0` is the first stable 0.x MPS histogram engine release. It supports dense numeric regression with a real MPS training path, split-scan and partition kernels, histogram subtraction, buffer reuse, explicit cache management, and documented performance boundaries.

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

The native backend exposes non-sensitive device and cache diagnostics:

```python
import mpsboost as mps

print(mps.__version__)
print(mps.is_available())
print(mps.system_info())
print(mps.cache_info())
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

The public API currently includes `MPSBoostRegressor`, cache diagnostics and management helpers, `is_available`, `system_info`, and `__version__`. Training supports dense finite `float32`/`float64`-compatible data, squared error, deterministic quantization, depth-limited histogram trees, model save/load, and explicit `device="mps"` or diagnostic `device="cpu"` selection.

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

MPSBoost is an independent open-source project. It is not affiliated with, endorsed by, or sponsored by Apple Inc., the XGBoost project, or the scikit-learn project. Apple, Metal, and Metal Performance Shaders may be trademarks of Apple Inc.

## License

Apache-2.0
