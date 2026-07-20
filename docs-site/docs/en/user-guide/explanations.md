# Feature Importance and Explanations

MPSBoost provides gain, split-count, and permutation importance, plus controlled
SHAP-like approximate explanations. Approximate explanations are not presented as
official SHAP. Use `permutation_importance` for model-agnostic interpretation; an
official TreeExplainer adapter remains a separate planned task.
