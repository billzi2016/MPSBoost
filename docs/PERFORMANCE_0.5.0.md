# MPSBoost 0.5.0 Performance Report

This report summarizes performance evidence for the 0.x hardening line. It is deliberately
conservative: checked-in S4/S6 numbers are historical pre-v2/v3 optimization baselines, while
`0.5.0` focuses on executable customer workflows, transparent backend selection, and documented
performance boundaries.

## Evidence Available

- S4 histogram benchmark on Apple M2 Ultra, Python 3.13.5, MPSBoost `0.2.0a0`.
- S6 end-to-end regressor benchmark on Apple M2 Ultra, Python 3.13.5, before later v2/v3 cleanup.
- Real-world acceptance reports for built-in, cached, and opt-in datasets.
- CI package tests on hosted CPU runners and self-hosted real Metal GPU runners.

## Historical MPS Boundary

The checked-in S6 report shows the expected shape:

| Scenario | Rows | Features | CPU median (s) | MPS median (s) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gbdt-medium` | 16,384 | 32 | 0.067630 | 0.089597 | 0.755x |
| `gbdt-wide` | 16,384 | 128 | 0.283176 | 0.249502 | 1.135x |
| `gbdt-large-wide` | 32,768 | 256 | 1.031002 | 0.633006 | 1.629x |

These are not final `0.5.0` speed claims. They are retained to show the workload boundary:
small or synchronization-heavy work can be slower on GPU, while wider/larger histogram workloads
can benefit from MPS.

## Current 0.5.0 Policy

- Use `device="auto"` for ordinary users.
- CPU is the correctness oracle and the preferred path for small data, branch-heavy anomaly
  detection, and latency-sensitive ranking.
- MPS is an acceleration backend for workloads large enough to amortize launch, transfer, and
  synchronization overhead.
- Explicit external portable policies record requested and effective backends. If the external
  runtime is not active for the estimator, MPSBoost warns and continues through native CPU
  compatibility rather than stopping the workflow.
- Linux CPU/CUDA performance depends on the selected external sklearn/XGBoost/CUDA stack and is
  reported as that external backend, not native MPSBoost CPU/MPS.

## Before 1.0.0

The final stable release needs a new full performance audit on current HEAD:

- train time and predict time across built-in, cached, and opt-in real-world datasets;
- peak memory;
- model size;
- wheel size;
- CPU/MPS applicability boundary;
- external backend attribution for Linux CPU/CUDA paths;
- raw command lines, machine details, package versions, and artifact hashes.
