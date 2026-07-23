# MPSBoost 1.0.0 Known-Issue Audit

No blocking customer-facing issue is recorded for the 1.0.0 release gate.

## Accepted Boundaries

- HIGGS remains an opt-in local-file performance-boundary dataset. It is not packaged or
  automatically downloaded because the raw dataset is too large for ordinary customer installs.
- Official SHAP execution requires the explicit `mpsboost[shap]` extra and semantic-validation
  path. Approximate explanations remain labeled approximate.
- Linux CPU/CUDA operation uses explicit external sklearn/XGBoost/CUDA backend policy and records
  requested/effective backend decisions.
- Workloads that are more suitable for CPU warn and continue on CPU when users request MPS.

## Customer Experience Requirements

- Missing Metal or MPS environment: warning plus copy-paste setup commands.
- CPU-only workers: `MPSBOOST_SKIP_ENV_CHECK=1`.
- Missing optional dependency: copy-paste `pip install 'mpsboost[extra]'` command.
- External backend runtime issue: clear attribution to the selected external stack while native CPU
  remains usable when possible.
