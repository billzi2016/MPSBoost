# MPSBoost 0.4.0 Release Audit

This document records the release gate for the `0.4.0` 0.x finishing feature milestone.

## Scope

`0.4.0` supports:

- dense numeric regression;
- binary and multiclass classification;
- squared-error objective;
- quantile, Poisson, and Tweedie regression objectives;
- deterministic quantization;
- depth-limited histogram trees;
- decision trees, random forests, ExtraTrees, and CatBoost-like numeric estimators;
- native CPU multiclass softmax with explicit OvR compatibility;
- CPU-suitable isolation forest anomaly scoring;
- pointwise learning-to-rank scoring with query-group validation;
- real MPS gradient, histogram, split-scan, partition, and buffer-pool paths;
- explicit CPU oracle mode;
- model save/load;
- feature importance, permutation importance, and controlled SHAP-like explanations;
- import-time MPS environment guidance with copy-paste setup and skip commands;
- cache diagnostics, explicit cache creation, and safe cache clearing;
- optional official-SHAP integration diagnostics with native tree export payloads;
- explicit portable-backend diagnostics and optional extras for XGBoost, sklearn, and CUDA;
- project-local opt-in real-world dataset caches for MNIST, Titanic, Adult Income, Covertype, and HIGGS;
- the S18 real-world acceptance report with open 1.0 release gates.

Not included:

- sparse matrices;
- native MPS multiclass softmax;
- validated official third-party SHAP TreeExplainer execution;
- categorical model persistence;
- public GPU prediction;
- full third-party API compatibility;
- full Linux CPU/CUDA smoke matrix;
- `1.0.0` stable-release commitment.

## License

- Project license: Apache-2.0.
- Runtime dependency: NumPy, with permissive license expression reported by
  package metadata.
- Build/test-only dependencies are not bundled into runtime wheels.
- The wheel must include the project `LICENSE` file.

## Wheel Content Rules

The release wheel must contain only runtime package files:

- Python package files;
- the native extension;
- the compiled Metal shader library;
- package metadata and license metadata.

The release wheel must not contain:

- `specs/`;
- `tests/`;
- `benchmarks/`;
- `.github/`;
- build directories;
- cache directories;
- raw `.metal`, `.air`, or temporary shader files;
- credentials or runner files.

## Dynamic Link Rules

The native extension may link to macOS system libraries and frameworks required
for Python, C++, Objective-C runtime, Foundation, CoreFoundation, and Metal.
It must not link to heavyweight ML runtimes or private project-local absolute
paths.

## Validation Matrix

Required before publishing `0.4.0`:

- local full test suite;
- GitHub hosted CPU/package tests for Python 3.10 and 3.13;
- self-hosted real Metal GPU tests for Python 3.10 and 3.13;
- `twine check` for the exact uploaded wheels;
- fresh PyPI install and real MPS smoke test.

`1.0.0` remains blocked until the complete S18 real-world matrix, model-quality audit,
end-to-end performance audit, peak-memory audit, wheel/model-size audit, permission audit,
artifact hashes, and explicit user confirmation are complete.

## Benchmark Evidence

Checked-in benchmark results:

- `benchmarks/results/s4-m2-ultra-py313.json`
- `benchmarks/results/s4-m2-ultra-py313.md`
- `benchmarks/results/s6-m2-ultra-py313.json`
- `benchmarks/results/s6-m2-ultra-py313.md`

The S6 report records both GPU wins and small-data regressions. The
`gbdt-large-wide` end-to-end scenario reached a 1.629x median speedup on the
recorded M2 Ultra validation machine.
