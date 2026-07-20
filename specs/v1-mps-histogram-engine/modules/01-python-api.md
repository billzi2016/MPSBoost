# Module Design: Python API

## 1. Responsibility

Provide a simple, stable estimator-style entry point for parameter validation, input
adaptation, exception conversion, and result presentation. This layer must not
implement binning, tree growth, or GPU hot paths.

## 2. Public Entry Point

```python
import mpsboost as mb

model = mb.GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    max_bins=256,
    min_child_weight=1.0,
    min_samples_leaf=20,
    reg_lambda=1.0,
    random_state=None,
    device="mps",
    verbosity=1,
)
model.fit(X, y)
prediction = model.predict(X_test)
```

Public 0.2.x symbols include only completed capabilities:
`GradientBoostingRegressor`, `MPSBoostRegressor`, `is_available`,
`system_info`, `__version__`, and cache diagnostic/management functions.
Unimplemented classification and low-level training APIs must not enter formal public
entry points.

## 3. Constructor Contract

- The constructor stores parameters only and performs no device initialization, file
  writing, or data allocation.
- Explicit parameters take precedence over `**kwargs`; Python naturally rejects unknown parameters.
- `get_params()` returns all constructor parameters; `set_params()` accepts only
  known names and returns `self`.
- Parameter semantics are defined in one validator, not copied among the estimator,
  binding, and C++.

## 4. `fit` Contract

### Inputs

- `X`: a two-dimensional numeric dense array;
- `y`: a one-dimensional numeric array whose length equals the row count;
- Version 0.2.0 accepts no NaN, Inf, sparse matrices, categorical dtypes, or
  multi-output values.

### Behavior

1. The Python layer performs lightweight type and structural checks.
2. The native boundary performs complete size, overflow, and memory-layout checks.
3. Create a training session and release the GIL during long computations.
4. On success, atomically replace estimator model state; on failure, retain the
   unfitted state.
5. Return `self`.

### Exceptions

- Input and parameter errors: `ValueError` or `TypeError`;
- unavailable MPS: `MPSBackendUnavailable`;
- insufficient memory budget: `MPSBoostMemoryError`;
- failed device command: `MPSExecutionError`;
- prediction before fitting: `NotFittedError`.

Exceptions are created by a unified conversion layer. Messages include the failing
stage and suggested resolution and do not expose meaningless error codes from the
implementation stack.

## 5. `predict` Contract

- Input feature count must match training;
- Return a one-dimensional `float32` or the dtype frozen by documentation;
- CPU and MPS inference read the same flat model;
- Prediction must not mutate the model or input;
- The small-batch strategy must be unique; opaque multiple prediction paths are forbidden.

## 6. Post-Fit State

- `n_features_in_`
- `device_`
- `n_estimators_`
- `model_` (a private native handle, not serializable Python state)
- `training_summary_` (a non-sensitive time and memory summary)

## 7. Threads and Ownership

- Concurrent `fit` on the same estimator is unsupported and explicitly rejected;
- Multiple estimators may share a read-only pipeline cache but not mutable training state;
- The native model uses an explicit ownership object; Python destruction must not
  release resources while GPU work is unsynchronized.

## 8. Comments and Tests

- Every public method must have a Chinese contract docstring;
- Test `get/set_params`, repeated `fit`, failure atomicity, unfitted use, input
  lifetime, and GIL behavior;
- Do not use fake native handles or mock GPUs to accept public training flows.
