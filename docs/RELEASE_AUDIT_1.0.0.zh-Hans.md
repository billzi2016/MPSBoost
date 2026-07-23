# MPSBoost 1.0.0 发布审计

`1.0.0` 是 native CPU/MPS MPSBoost 范围的 stable customer-commitment release。

## 范围

- Native CPU 和 MPS tree-learning backend。
- sklearn 风格 gradient boosting、decision tree、random forest、ExtraTrees 和 CatBoost-like numeric estimator。
- 二分类和多分类 classification，包含 native CPU softmax 和显式 OvR compatibility。
- Quantile、Poisson、Tweedie、feature importance、permutation importance 和 approximate SHAP-like explanation。
- CPU-suitable anomaly detection 和 pointwise ranking，并对 MPS routing 使用 warning。
- 显式 optional SHAP 和 portable backend diagnostics。
- 项目内真实世界数据 cache policy，以及 documented HIGGS local-file boundary。
- 客户侧 setup、skip、warning 和 external-dependency attribution path。

## 验证

- `PYTHONPATH=src pytest -q tests/real_world -rs`：21 passed，1 HIGGS local-file boundary skip。
- `PYTHONPATH=src pytest -q tests/unit/test_portable_backend.py tests/unit/test_diagnostics.py tests/packaging/test_public_api.py::test_completed_estimators_are_public tests/packaging/test_public_api.py::test_estimator_capability_registry_reports_available_and_planned_models -rs`：12 passed。
- Fresh Python 3.13 wheel environment，
  `/tmp/mpsboost-release-check-100-py313/bin/python -m pytest -q tests/unit tests/integration tests/packaging -m "not gpu" -rs`：
  220 passed。
- 0.5.0 publication record 上 GitHub CI 和 Docs passed。
- 1.0.0 文档更新和 zh-Hans navigation translation 后，`mkdocs build --strict` passed。
- `python -m twine check dist/mpsboost-1.0.0-*.whl`：passed。
- Python 3.10 和 Python 3.13 fresh local wheel smoke passed：import、version `1.0.0`、multiclass
  `GradientBoostingClassifier.fit`、`predict_proba` 和 `system_info()`。

## Artifacts

- `dist/mpsboost-1.0.0-cp310-cp310-macosx_13_0_arm64.whl`：280K，
  SHA256 `0ef03786ba5c1511cca88b126d5632fca8ee5f49cf8cc80dafd9499f39b6e1fa`。
- `dist/mpsboost-1.0.0-cp313-cp313-macosx_13_0_arm64.whl`：284K，
  SHA256 `61df4018d7f936a7b4814485b22a0ed755ef529bb359117aafc15e8cd972c4b5`。
- Wheel tag 是 `macosx_13_0_arm64`，用于覆盖 Apple Silicon 上受支持的现代 macOS release。

## Artifact 规则

Wheel 不得包含 raw dataset、test cache、`specs/`、`tests/`、`benchmarks/`、`.github/`、build directory、credential 或 runner file。

## PyPI

- PyPI URL：https://pypi.org/project/mpsboost/1.0.0/
- Upload command：
  `python -m twine upload dist/mpsboost-1.0.0-cp310-cp310-macosx_13_0_arm64.whl dist/mpsboost-1.0.0-cp313-cp313-macosx_13_0_arm64.whl`
- Python 3.10 和 Python 3.13 formal PyPI fresh install smoke passed：
  `python -m pip install --no-cache-dir mpsboost==1.0.0`、import、version `1.0.0`、
  multiclass `GradientBoostingClassifier.fit`、`predict_proba` 和 `system_info()`。
