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
