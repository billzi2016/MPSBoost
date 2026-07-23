# Feature Importance and Explanations

MPSBoost provides multiple explanation layers:

- gain importance
- split-count importance
- permutation importance
- controlled SHAP-like approximate explanations

## Design principles

- Do not duplicate prediction or scoring logic.
- Do not present approximate explanations as official SHAP.
- Advance the official SHAP TreeExplainer adapter as a separate task.

## Usage direction

When research or reporting needs official SHAP semantics, wait for the S16.5a / S23 adapter
documentation. For the current package, use `permutation_importance` as model-agnostic
interpretation.

## Official SHAP path

Official SHAP is optional and explicit:

```bash
python -m pip install 'mpsboost[shap]'
```

`export_native_trees_for_shap(estimator)` exports native tree structure for TreeExplainer adapter
validation without training data, credentials, telemetry, or device identifiers.
`official_shap_tree_explainer(...)` stops clearly until semantic validation is enabled, so
approximate explanations are not presented as official SHAP output.
