# Backend Policy

MPSBoost currently has two first-class native backends:

- `device="cpu"`: the in-project CPU backend and correctness oracle.
- `device="mps"`: requests the Apple GPU/MPS backend.
- `device="auto"`: selects a backend from the environment and workload.

The CPU backend is not a temporary substitute. It supports correctness validation,
small-data fast paths, usable execution when MPS is unavailable, and selected
CPU-suitable estimators.

## CPU-Suitable Estimators

`IsolationForest` / `MPSIsolationForest` and `LearningToRankRegressor` are
CPU-suitable workflows. Their branch-heavy, latency-sensitive computation is
currently expected to suit CPU execution better than Apple GPU execution.

When a user requests `device="mps"`, the library warns, continues execution, and
records the requested device, actual device, and reason in `training_summary_`.
This is product backend scheduling, not an implementation-missing fallback.
