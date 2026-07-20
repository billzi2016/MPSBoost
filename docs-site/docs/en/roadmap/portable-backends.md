# S22 Portable Backends

S22 extends one estimator interface to more machines without replacing native CPU,
which remains the correctness oracle and default CPU backend. Planned external
adapters are observable through `training_summary_["backend"]` and use optional
extras: `mpsboost[xgboost]`, `mpsboost[sklearn]`, and `mpsboost[cuda]`.
