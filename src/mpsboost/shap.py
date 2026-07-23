"""Optional official SHAP integration helpers.

MPSBoost exposes an approximate SHAP-like method on fitted estimators, but that method is not
official SHAP. This module provides the explicit S16.5a boundary: optional dependency checks,
native tree export for adapter work, and clear guidance when ``shap`` is not installed.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any


def shap_setup_instructions() -> str:
    """Return copy-paste commands for enabling the optional official SHAP path."""

    return (
        "Official SHAP integration is optional and is not installed by default. Install it with:\n"
        "  python -m pip install 'mpsboost[shap]'\n"
        "Approximate SHAP-like explanations remain available through "
        "estimator.approximate_shap_values(...), but they must not be described as official SHAP."
    )


def is_shap_available() -> bool:
    """Return whether the optional ``shap`` dependency can be imported."""

    return find_spec("shap") is not None


def export_native_trees_for_shap(estimator: Any) -> dict[str, Any]:
    """Export fitted native tree structure for future SHAP TreeExplainer adapter validation.

    The payload is intentionally plain Python data. It contains model structure and objective
    metadata, but no training data, credentials, telemetry, or device identifiers.
    """

    if hasattr(estimator, "_require_model"):
        model = estimator._require_model()
        return _export_native_model(model, estimator=type(estimator).__name__)
    if hasattr(estimator, "estimators_"):
        return {
            "format": "mpsboost-shap-export",
            "version": 1,
            "estimator": type(estimator).__name__,
            "kind": "forest",
            "trees": [
                _export_native_model(tree._require_model(), estimator=type(tree).__name__)
                for tree in estimator.estimators_
            ],
        }
    raise TypeError("estimator must be a fitted MPSBoost estimator")


def official_shap_tree_explainer(estimator: Any, **kwargs: Any) -> Any:
    """Create the optional official SHAP TreeExplainer for an exported native model.

    This function currently stops before claiming full semantic compatibility. It verifies the
    optional dependency and export path, then raises a clear message describing the remaining
    adapter contract. That avoids presenting approximate explanations as official SHAP.
    """

    if not is_shap_available():
        raise ImportError(shap_setup_instructions())
    export_native_trees_for_shap(estimator)
    del kwargs
    raise RuntimeError(
        "Official SHAP TreeExplainer adapter validation is not enabled for this release. "
        "Use estimator.approximate_shap_values(...) for controlled approximate explanations, "
        "or track S16.5a for the validated official adapter."
    )


def _export_native_model(model: Any, *, estimator: str) -> dict[str, Any]:
    """Convert one native model binding object into a stable adapter payload."""

    trees = []
    for tree in model.trees:
        nodes = []
        for node in tree.nodes:
            nodes.append(
                {
                    "is_leaf": bool(node["is_leaf"]),
                    "feature_index": int(node["feature_index"]),
                    "threshold_bin": int(node["threshold_bin"]),
                    "left_child": int(node["left_child"]),
                    "right_child": int(node["right_child"]),
                    "leaf_value": float(node["leaf_value"]),
                    "gain": float(node["gain"]),
                    "default_left": bool(node["default_left"]),
                }
            )
        trees.append({"nodes": nodes})
    return {
        "format": "mpsboost-shap-export",
        "version": 1,
        "estimator": estimator,
        "kind": "native_model",
        "objective": str(model.objective),
        "feature_count": int(model.feature_count),
        "tree_count": int(model.tree_count),
        "trees": trees,
    }
