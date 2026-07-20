# Product Requirements Document (PRD)

## 1. Product Vision
MPSBoost provides efficient, stable, easy-to-install gradient-boosted decision-tree
training and inference for local tabular learning on Apple Silicon. Users choose
`device="mps"` through a familiar estimator-style Python API without needing to
understand which operations use optimized primitives or custom GPU kernels.

## 2. User Pain Points
- Local devices have powerful GPUs and unified memory, but conventional tree training
  usually cannot use those resources.
- Remote GPUs add cost, environment differences, data transfer, and privacy risks.
- General tensor interfaces do not suit histogram aggregation, discrete splits, and
  dynamic sample partitioning.
- Source compilation, complex toolchains, and heavy dependencies block ordinary Python users.
- Showing kernel acceleration while ignoring preprocessing and synchronization creates false performance expectations.

## 3. Target Users
- Data developers performing tabular regression and classification on Apple Silicon Macs;
- Teams needing local, offline, or privacy-friendly training;
- Users needing estimator-style interfaces integrated with existing Python workflows;
- Systems developers researching unified memory and irregular GPU algorithms.

## 4. Product Principles
1. Correctness, stability, and easy installation take priority over feature count.
2. The user entry point is uniformly `mps`; internal details do not leak into public parameters.
3. No silent fallback, unknown parameters, or fabricated features or performance.
4. CPUs may be faster for small data; the product must publish its applicability boundary.
5. Caching improves only speed and does not change training results.

## 5. Version Scope
### 0.2.0 Must Deliver
- Prebuilt wheels for Apple Silicon arm64 macOS;
- an `MPSBoostRegressor` estimator;
- numerical dense two-dimensional inputs;
- squared-error objective;
- deterministic binning;
- depth-limited histogram GBDT;
- real MPS execution of gradient and histogram hot paths;
- a CPU oracle for tests, not silent fallback for `device="mps"`;
- `fit`, `predict`, `get_params`, and `set_params`;
- model save and load;
- `is_available()` and `system_info()`;
- clear errors, Chinese code comments, complete tests, and reproducible benchmarks.

### Candidate Later Versions
- Binary classification, probability prediction, and early stopping;
- missing-value default directions;
- row/column sampling and additional regularization;
- GPU split scan, partition, and full training/prediction hot paths;
- categorical features, multiclass classification, ranking, and explainability.

Candidate capabilities must not be exposed as available APIs before entering their
corresponding approved tasks.

## 6. Core User Flow
```python
import mpsboost as mb

model = mb.GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    max_bins=256,
    reg_lambda=1.0,
    device="mps",
    random_state=42,
)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
model.save_model("model.mb")
```

## 7. Functional Requirements
### FR-01 Inputs
- Accept dense arrays of two-dimensional finite numeric values and one-dimensional labels.
- Validate shape, dtype, contiguity, overflow, and lifetime.
- Whether input conversion copies must be diagnosable.
### FR-02 Parameters
- Constructor parameters use common estimator naming.
- Unknown parameters fail immediately; conflicting parameters state their conflict reason.
- Constructors have no costly side effects; device initialization occurs in `fit()`.
### FR-03 Training
- Use unified binned data and second-order gradient statistics.
- Tree structure, leaf values, and predictions agree with the CPU oracle within stated tolerances.
- Device failure must not return a partially built model.
### FR-04 Prediction
- Support batch prediction after training.
- Predictions after loading match those before saving.
- Calling before fitting must fail explicitly.
### FR-05 Device
- `device="mps"` requires a real Apple GPU backend and otherwise fails early.
- `device="cpu"` is only for reference and diagnostics.
- `device="auto"` has initial selection rules and observability: use CPU for small
  work or unavailable MPS; use MPS when MPS is available and estimated hot paths are sufficiently large.
### FR-06 Diagnostics
- Return package version, backend availability, device name, runtime mode, and key timings.
- Do not output user names, home directories, training data, or credentials.
### FR-07 Cache
- Strictly separate L1 process cache, L2 user-rebuildable cache, and L3 build cache.
- Caches have version keys, validation, atomic writes, and safe invalidation.
- Importing the package and querying paths do not create cache directories.
### FR-08 Model
- Model format is versioned, length-verifiable, and contains no raw training data.
- CPU and MPS inference read the same format.
- Unknown newer versions are rejected by default; do not guess parsing.

## 8. Non-Functional Requirements
### NFR-01 Installation
- Install on supported platforms with `python -m pip install mpsboost`.
- Wheels include native extensions and shader resources.
- Users need no heavy runtime, package manager, or compiler.
### NFR-02 Performance
- On large preregistered data, GPU histograms target at least 2x the project CPU oracle.
- At least one preregistered end-to-end scenario exceeds a strong CPU baseline, targeting at least 1.3x.
- All performance claims include preprocessing, synchronization, device, data, and model quality.
### NFR-03 Memory
- `max_bins <= 256` uses `uint8` by default.
- Reuse long-lived buffers; forbid accidental allocations proportional to `rows × features × bins`.
- Check before out-of-memory and provide estimates and guidance.
### NFR-04 Stability
- Check completion status for every GPU command.
- Fail early on shader/native ABI mismatches.
- Repeated training does not grow memory linearly.
- Cache corruption causes rebuilding only, not an incorrect model.
### NFR-05 Maintainability
- Follow SOLID/DRY and Chinese comment rules.
- Protect public interfaces, model format, and key data layouts with tests.
- Do not allow two parameter semantics or duplicated mathematical formulas.

## 9. Explicit Non-Goals
- Version 0.2.0 does not implement multi-machine, multi-GPU, other operating systems,
  or non-Apple Silicon.
- Version 0.2.0 does not implement complete categorical features, multiclass,
  ranking, or explainability.
- Do not start from external project source code or copy third-party implementations.
- Do not develop a graphical interface or App Store product.

## 10. Definition of Done for 0.2.0
Release is allowed only when all of the following hold:
1. Real regression training and prediction are complete, with no mock or placeholder-success path.
2. CPU/GPU correctness, boundary, stability, and installation tests all pass.
3. Supported-environment wheels work on clean machines without compilation.
4. End-to-end benchmarks honestly record successful and degraded ranges.
5. Model save/load, cache invalidation, and error diagnostics pass tests.
6. All 0.2.0 release gates in `tasks.md` are checked.
