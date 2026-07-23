# MPSBoost 1.0.0 Release Audit

`1.0.0` is the stable customer-commitment release for the native CPU/MPS MPSBoost scope.

## Scope

- Native CPU and MPS tree-learning backends.
- sklearn-style gradient boosting, decision tree, random forest, ExtraTrees, and CatBoost-like
  numeric estimators.
- Binary and multiclass classification with native CPU softmax and explicit OvR compatibility.
- Quantile, Poisson, Tweedie, feature importance, permutation importance, and approximate
  SHAP-like explanations.
- CPU-suitable anomaly detection and pointwise ranking with warning-based MPS routing.
- Explicit optional SHAP and portable backend diagnostics.
- Project-local real-world dataset cache policy and documented HIGGS local-file boundary.
- Customer-facing setup, skip, warning, and external-dependency attribution paths.

## Validation

- `PYTHONPATH=src pytest -q tests/real_world -rs`: 21 passed, 1 HIGGS local-file boundary skip.
- `PYTHONPATH=src pytest -q tests/unit/test_portable_backend.py tests/unit/test_diagnostics.py tests/packaging/test_public_api.py::test_completed_estimators_are_public tests/packaging/test_public_api.py::test_estimator_capability_registry_reports_available_and_planned_models -rs`:
  12 passed.
- Fresh Python 3.13 wheel environment,
  `/tmp/mpsboost-release-check-100-py313/bin/python -m pytest -q tests/unit tests/integration tests/packaging -m "not gpu" -rs`:
  220 passed.
- GitHub CI and Docs passed on the 0.5.0 publication record.
- `mkdocs build --strict`: passed after the 1.0.0 documentation updates and zh-Hans navigation
  translations.
- `python -m twine check dist/mpsboost-1.0.0-*.whl`: passed.
- Fresh local wheel smoke passed on Python 3.10 and Python 3.13: import, version `1.0.0`,
  multiclass `GradientBoostingClassifier.fit`, `predict_proba`, and `system_info()`.

## Artifacts

- `dist/mpsboost-1.0.0-cp310-cp310-macosx_13_0_arm64.whl`: 280K,
  SHA256 `0ef03786ba5c1511cca88b126d5632fca8ee5f49cf8cc80dafd9499f39b6e1fa`.
- `dist/mpsboost-1.0.0-cp313-cp313-macosx_13_0_arm64.whl`: 284K,
  SHA256 `61df4018d7f936a7b4814485b22a0ed755ef529bb359117aafc15e8cd972c4b5`.
- Wheel tags are `macosx_13_0_arm64` for Apple Silicon compatibility across supported modern
  macOS releases.

## Artifact Rules

The wheel must not contain raw datasets, test caches, `specs/`, `tests/`, `benchmarks/`, `.github/`,
build directories, credentials, or runner files.

## PyPI

- PyPI URL: https://pypi.org/project/mpsboost/1.0.0/
- Upload command:
  `python -m twine upload dist/mpsboost-1.0.0-cp310-cp310-macosx_13_0_arm64.whl dist/mpsboost-1.0.0-cp313-cp313-macosx_13_0_arm64.whl`
- Fresh formal PyPI install smoke passed on Python 3.10 and Python 3.13:
  `python -m pip install --no-cache-dir mpsboost==1.0.0`, import, version `1.0.0`,
  multiclass `GradientBoostingClassifier.fit`, `predict_proba`, and `system_info()`.
