# MPSBoost 0.1.0a0 Release Audit

`0.1.0a0` reserved the `mpsboost` package namespace and published the planned public API surface.
It was an alpha placeholder release for project discovery and dependency metadata, not a functional
training release.

Included:

- package namespace reservation;
- initial sklearn/XGBoost-style API direction;
- backend diagnostics and cache-path planning;
- explicit failure for training operations outside the alpha scope.

Not included:

- real training;
- real MPS kernels;
- model persistence;
- performance claims.
