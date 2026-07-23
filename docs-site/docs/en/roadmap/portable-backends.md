# S22 Portable Backends

S22 aims to make the same MPSBoost estimator interface run on more machines.

## Does not replace native CPU

The MPSBoost native CPU backend remains:

- the correctness oracle
- the default CPU backend
- the Apple Silicon path when no GPU is available or the workload is small
- the test baseline

Portable backends do not replace native CPU. Runtime summaries must report the actual backend and
must not change the current native CPU/MPS default path.

## Target backends

Planned paths:

- Apple Silicon: MPSBoost native CPU/MPS
- Linux CPU: MPSBoost native CPU or sklearn/XGBoost CPU adapter
- Linux CUDA: XGBoost GPU adapter
- Windows CPU: sklearn/XGBoost CPU adapter

All external backends must write `training_summary_["backend"]`, so users can see the actual
runtime path.

## Dependency policy

External backends use optional extras:

- `mpsboost[xgboost]`
- `mpsboost[sklearn]`
- `mpsboost[cuda]`

Default installs remain lightweight.

## Current foundation

`optional_dependency_status()` reports optional extras without importing heavy dependencies.
`portable_setup_instructions()` returns copy-paste installation commands without interactive input.
`choose_portable_backend(...)` records the selected policy and actual backend. The
`PortableEstimatorAdapter` preserves `fit`, `predict`, `predict_proba`, `score`, `get_params`, and
`set_params` for the native path, while external adapters stop until their backend mappings are
validated.
