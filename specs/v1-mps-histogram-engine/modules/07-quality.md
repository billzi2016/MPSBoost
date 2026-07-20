# Module Design: Tests, Benchmarks, and Quality

## 1. Test Pyramid
- Unit: mathematics, binning, parameters, and model format;
- CPU oracle: hand-calculated trees and determinism;
- MPS kernels: buffer-by-buffer comparison on real devices;
- Integration: Python → native → MPS → model;
- Installation: smoke `fit/predict` in a clean wheel environment;
- Stability: repeated training, exceptions, cache corruption, and memory pressure;
- Benchmarks: separate processes with fixed data and parameters.

## 2. Prohibitions
- Do not mock GPUs to accept real-device functionality;
- do not delete or skip failing tests to hide defects;
- do not compute expected values with the implementation itself;
- do not arbitrarily relax floating-point tolerances to pass tests;
- do not report only the fastest run or exclude preprocessing.

## 3. Numerical Comparison
Use `abs(actual-expected) <= atol + rtol*abs(expected)`; freeze tolerances separately
by kernel, accumulation scale, and model metric. Tolerance changes require error
distributions and root causes.

Compare, in order: per-bin `count/G/H`, split feature/bin/gain, tree structure/leaf
values, per-sample predictions, and final metrics.

## 4. Benchmarks
Preregister Small, Medium, Large, and Wide synthetic datasets plus at least two real
datasets that may legally be distributed/downloaded. Record:
- device, system, version, data hash, and parameters;
- CPU threads;
- cold/warm cache;
- input conversion, binning, initialization, kernels, synchronization, and total fit;
- peak RSS and device memory;
- model quality.

Public conclusions must show small-data degradation ranges and must not replace full
results with “up to” claims.

## 5. Code Quality
- Enable strict C++ warnings and treat project warnings as errors;
- run Python linting, type checking, and tests;
- audit shader compilation warnings;
- include CPU/native paths that can run under sanitizers in CI;
- review Chinese file headers, function contracts, and key-point comments.

## 6. Security
Fuzz model loaders and size calculations; validate cache-cleanup paths; audit wheels
for secrets, absolute paths, and non-distributable content; validate that runtime has
no implicit network access or extra permissions.

## 7. Completion Gate
Check a task only after module tests, cross-module integration, installation smoke,
performance sanity, and comment review all pass. Record blocked test environments as
incomplete.
