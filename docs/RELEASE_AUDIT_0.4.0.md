# MPSBoost 0.4.0 Release Audit

This document records the release gate for `0.4.0`: the 0.x release after the large
real-world acceptance pass. It does not rewrite the `0.3.0` audit; `0.3.0` remains the all-trees
feature milestone before the large-scale S18 validation pass.

## Scope

`0.4.0` includes the `0.3.0` all-trees scope plus:

- optional official-SHAP integration diagnostics with native tree export payloads;
- explicit portable-backend diagnostics and optional extras for XGBoost, sklearn, and CUDA;
- project-local opt-in real-world dataset caches for MNIST, Titanic, Adult Income, Covertype, and HIGGS;
- the S18 real-world acceptance report with success, degradation, and boundary notes;
- documentation that `1.0.0` remains blocked until final stable-release gates are complete.

## Boundaries

Native CPU/MPS remains the MPSBoost implementation and correctness oracle. Linux CPU and Linux
CUDA execution are exposed through explicit external backend policy and adapters; failures inside
the external sklearn/XGBoost/CUDA runtime are not represented as native MPSBoost backend failures.

`1.0.0` remains blocked until model-quality, end-to-end performance, peak-memory, wheel/model-size,
permission, artifact-hash, and explicit user-confirmation gates are complete.

## Validation Matrix

Required before publishing `0.4.0`:

- local non-GPU unit, integration, packaging, and real-world acceptance tests;
- opt-in external real-world dataset acceptance from project-local caches;
- GitHub hosted CPU/package tests for Python 3.10 and 3.13;
- self-hosted real Metal GPU tests for Python 3.10 and 3.13;
- `twine check` for the exact uploaded wheels;
- fresh wheel install and import/training smoke verification.

## Artifact Rules

The release wheel must not contain `specs/`, `tests/`, `benchmarks/`, `.github/`, build
directories, cache directories, raw datasets, credentials, or runner files.
