# MPSBoost 0.2.0 Release Audit

`0.2.0` was the first formal MPS histogram engine release. It established the real compiled
Apple Silicon backend, deterministic data/binning path, regression GBDT, model save/load, cache
safety, and release verification workflow.

Included:

- real compiled MPS/Metal backend foundation;
- native device diagnostics and GPU smoke kernel;
- deterministic quantization and compact binned representation;
- real regression GBDT training and prediction;
- model save/load;
- S6 benchmark evidence for both MPS wins and small-data regressions;
- cache diagnostics, explicit cache creation, and safe cache clearing;
- wheel, license, dynamic-link, and PyPI verification.

Not included:

- classifiers;
- forest/ExtraTrees/CatBoost-like families;
- advanced objectives;
- anomaly/ranking estimators;
- large real-world dataset matrix.
