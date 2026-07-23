# Changelog

All notable changes to this project will be documented in this file.

## 1.0.0 - 2026-07-23

- Publish the stable customer-commitment release after the 0.x feature, validation, and hardening
  line.
- Freeze the public scope around native CPU/MPS tree estimators, explicit optional backend
  diagnostics, project-local real-world dataset caches, and customer-facing environment guidance.
- Add final release, known-issue, and performance reports that distinguish current HEAD evidence
  from historical pre-v2/v3 benchmark baselines.
- Keep HIGGS as an explicit large local-file performance-boundary dataset rather than packaging or
  automatically downloading multi-gigabyte raw data into user environments.
- Require fresh PyPI install verification and versioned artifact hashes before closing the release
  audit.

## 0.5.0 - 2026-07-23

- Harden portable backend behavior so explicit external backend policies warn and continue through
  the native CPU compatibility path instead of stopping user workflows.
- Record both requested and effective portable backend decisions in adapter training summaries.
- Add versioned release audits and release pages through `0.5.0` while preserving earlier release
  history.
- Move long-form project documentation into `docs/` and AI usage guidance into `ai-skills/`,
  leaving README as the root entry point.
- Add the 0.5.0 known-issue audit as the handoff gate before any future `1.0.0` commitment.

## 0.4.0 - 2026-07-23

- Add explicit portable-backend diagnostics, optional `xgboost`, `sklearn`, and `cuda` extras, and observable backend selection summaries.
- Add the official SHAP integration boundary with `mpsboost[shap]`, native tree export payloads, and a clear semantic-validation gate for TreeExplainer.
- Add opt-in real-world acceptance tests for MNIST, Titanic, Adult Income, and HIGGS using project-local ignored caches instead of user-global cache paths.
- Add the S18 real-world acceptance report and documentation-site navigation for the report.
- Keep native CPU/MPS as the default implementation and correctness oracle; external backends remain explicit, observable, and optional.

## 0.3.0 - 2026-07-20

- Add sklearn-style binary and multiclass classifiers, including native CPU softmax and explicit
  OvR compatibility.
- Add decision tree, random forest, ExtraTrees, and CatBoost-like numeric estimator families on
  the shared native tree engine.
- Add quantile, Poisson, and Tweedie regression objectives with model-format round trips.
- Add gain, split-count, permutation, and controlled SHAP-like explanation helpers.
- Add CPU-suitable isolation forest and pointwise learning-to-rank estimators with observable
  backend routing.
- Add import-time MPS environment guidance with copy-paste setup commands and
  `MPSBOOST_SKIP_ENV_CHECK=1` for CPU-only workers.
- Expand real-world dataset smoke coverage and CPU baseline records for the 0.x release line.

## 0.2.0 - 2026-07-19

- Replace preview mock APIs with a real compiled MPS/Metal backend foundation.
- Add native device diagnostics and a real GPU vector smoke kernel.
- Build Metal shaders at package build time and bundle the resulting library.
- Add deterministic native quantization for strided float32/float64 buffers.
- Add compact uint8/uint16 feature-major bins and validated serialization.
- Reject invalid values, unsupported layouts, corrupted data, and size overflows.
- Add real MPS split-scan and stable partition/compaction kernels.
- Add histogram subtraction and layer-aware training hooks.
- Add an L1 temporary buffer pool for MPS histogram and hot-path workspaces.
- Add S6 end-to-end regressor benchmarks with documented small-data regressions and large-wide wins.
- Add versioned L2 cache keys, atomic writes, checksum validation, and damaged-cache invalidation.
- Add public cache diagnostics, explicit cache directory creation, and safe cache clearing.
- Tighten CI cache boundaries so self-hosted GPU jobs do not write or upload user-level pip caches.

## 0.2.0rc0 - 2026-07-19

- Publish the cache and stability release candidate.

## 0.2.0b0 - 2026-07-19

- Publish the GPU hot-path beta milestone.

## 0.2.0a0 - 2026-07-19

- Publish the first functional MPS regressor alpha.

## 0.1.0a0 - 2026-07-19

- Reserve the `mpsboost` package namespace.
- Publish the planned sklearn/XGBoost-style Python API surface.
- Add backend diagnostics and layered cache-path definitions.
- Fail explicitly for training operations that are not implemented yet.
