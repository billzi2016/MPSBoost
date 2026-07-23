# MPSBoost 0.5.0 Known-Issue Audit

This file records the 0.5.0 hardening gate. It is not a `1.0.0` stability promise; it is the
current known-issue clearing record for the 0.x release line.

## Blocking Customer-Facing Issues

No blocking customer-facing issue is recorded in the repository at this gate after the 0.5.0
portable-backend fallback hardening.

## Explicit Boundaries

- Native CPU/MPS remains the MPSBoost implementation and correctness oracle.
- IsolationForest and pointwise ranking are CPU-suitable workflows; MPS requests warn and continue
  on CPU because that workload shape is better suited to CPU execution.
- Official SHAP TreeExplainer execution still requires semantic validation before MPSBoost claims
  official SHAP output. Approximate explanations remain clearly labeled approximate.
- Linux CPU and Linux CUDA operation go through explicit external sklearn/XGBoost/CUDA policy. A
  failure inside that external runtime is an external dependency/environment issue, while MPSBoost
  keeps native CPU usable when possible.
- `1.0.0` remains blocked until final customer-commitment gates are complete.

## Customer-Facing Fallback Checks

- Missing MPS environment: warning plus copy-paste `xcode-select`, Metal toolchain, reinstall, and
  diagnostic commands.
- CPU-only workers and model selection: `MPSBOOST_SKIP_ENV_CHECK=1` remains available and
  non-interactive.
- Missing optional SHAP/portable dependencies: copy-paste extras are reported by diagnostics.
- Explicit external portable policy: adapter records requested and effective backend, warns, and
  continues through native CPU compatibility when the external runtime is not active for the
  estimator.
