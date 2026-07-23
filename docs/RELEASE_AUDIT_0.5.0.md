# MPSBoost 0.5.0 Release Audit

`0.5.0` is the zero-known-blocking-issue hardening release for the 0.x line. It does not replace
the `0.4.0` large-scale validation release; it adds customer-facing fallback behavior and a
known-issue audit before any future `1.0.0` commitment.

## Scope

`0.5.0` includes the `0.4.0` scope plus:

- portable backend fallback hardening so explicit external policies warn and keep workflows
  executable through native CPU compatibility when an external runtime is not active for the
  estimator;
- requested/effective portable backend records in training summaries;
- versioned known-issue audit;
- versioned performance report with historical baseline caveats and the required `1.0.0` audit scope;
- append-only release documentation through `0.5.0`.

## Validation Required Before Publishing

- portable backend unit tests;
- diagnostics tests;
- packaging public API tests;
- real-world opt-in skip matrix;
- MkDocs strict build and symlink check;
- wheel build, `twine check`, and fresh wheel install smoke verification;
- GitHub CI and Docs success after push.

See `docs/PERFORMANCE_0.5.0.md` for performance boundaries. Historical S4/S6 results remain
pre-v2/v3 baselines and are not presented as final current-HEAD speed claims.

## 1.0 Boundary

`1.0.0` remains blocked until all planned customer-facing failure paths, full real-world matrix
gates, performance/memory/permission audits, artifact hashes, and explicit final user confirmation
are complete.
