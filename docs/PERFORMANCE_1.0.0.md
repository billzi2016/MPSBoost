# MPSBoost 1.0.0 Performance Report

This report records the final 1.0.0 performance and size evidence available on the release machine.
It separates current-HEAD measurements from historical pre-v2/v3 benchmark baselines.

## Release Machine

- Date: 2026-07-23
- Platform: Apple Silicon macOS
- Python used for local audit: Python 3.13
- Dataset cache policy: project-local ignored `tests/real_world/data/` and `tests/real_world/cache/`
- Wheel artifacts checked locally:
  `dist/mpsboost-1.0.0-cp310-cp310-macosx_13_0_arm64.whl` at 280K and
  `dist/mpsboost-1.0.0-cp313-cp313-macosx_13_0_arm64.whl` at 284K

## Current-HEAD CPU Audit

These measurements use current source before the final `1.0.0` wheel rebuild. Peak RSS is converted
from the process maximum resident set size and rounded to MiB.

| Dataset | Rows | Features | Estimator | Train (s) | Predict (s) | Metric | Score | Model size |
| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| Iris | 150 | 4 | GradientBoostingClassifier | 0.000965 | 0.000044 | accuracy | 0.973684 | 9,460 B |
| Breast Cancer | 569 | 30 | GradientBoostingClassifier | 0.006503 | 0.000106 | accuracy | 0.930070 | 11,048 B |
| Diabetes | 442 | 10 | GradientBoostingRegressor | 0.003520 | 0.000048 | R2 | 0.424435 | 11,156 B |
| Digits | 1,797 | 64 | GradientBoostingClassifier | 0.106488 | 0.001003 | accuracy | 0.740000 | 49,528 B |
| California Housing | 20,640 | 8 | GradientBoostingRegressor | 0.079778 | 0.003057 | R2 | 0.726251 | 33,000 B |
| Covertype subset | 30,000 | 54 | GradientBoostingClassifier | 0.423550 | 0.006479 | accuracy | 0.473067 | 19,604 B |

Peak RSS during this audit stayed under 495 MiB. The largest observed value was from the Covertype
subset run after loading and splitting the cached dataset.

## Historical MPS Baselines

The checked-in S4/S6 results remain useful for workload-shape guidance but are not final 1.0.0
speed claims. They were recorded before later v2/v3 implementation and cleanup work. Their role is
to show the stable boundary: small or synchronization-heavy jobs can be slower on MPS, while wider
and larger histogram workloads can benefit from Apple GPU execution.

The historical S6 large-wide regressor case reached 1.629x median speedup over the CPU oracle on
the recorded M2 Ultra machine.

## HIGGS Boundary

HIGGS is intentionally a local-file, opt-in performance-boundary dataset. The raw official dataset
is multi-gigabyte scale and is not automatically downloaded into user environments or packaged in
the wheel. The executable test remains in `tests/real_world/test_external_opt_in_datasets.py` and
states the exact local file path required to run it.

## 1.0.0 Policy

- Use `device="auto"` by default.
- CPU is the correctness oracle and the preferred path for small, branch-heavy, and
  latency-sensitive workloads.
- MPS is an acceleration backend for workloads large enough to amortize launch, transfer, and
  synchronization overhead.
- Linux CPU/CUDA performance belongs to the selected external sklearn/XGBoost/CUDA stack and is
  reported as such.
