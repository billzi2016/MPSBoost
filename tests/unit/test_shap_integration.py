"""Official SHAP integration boundary tests."""

import numpy as np
import pytest

import mpsboost as mb


def _fitted_regressor():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0.0, 0.0, 3.0, 3.0], dtype=np.float32)
    return mb.GradientBoostingRegressor(
        n_estimators=1,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        device="cpu",
    ).fit(X, y)


def test_shap_setup_instructions_are_explicit():
    """Official SHAP guidance should use optional extras and separate approximate explanations."""

    text = mb.shap_setup_instructions()

    assert "python -m pip install 'mpsboost[shap]'" in text
    assert "approximate_shap_values" in text
    assert "official SHAP" in text
    assert "input(" not in text


def test_native_tree_export_for_shap_contains_structure_without_training_data():
    """SHAP export should expose native model structure without private payloads."""

    export = mb.export_native_trees_for_shap(_fitted_regressor())

    assert export["format"] == "mpsboost-shap-export"
    assert export["kind"] == "native_model"
    assert export["feature_count"] == 2
    assert export["tree_count"] == 1
    assert export["trees"][0]["nodes"]
    assert "leaf_value" in export["trees"][0]["nodes"][0]
    assert "default_left" in export["trees"][0]["nodes"][0]
    assert "training_data" not in export
    assert "device_identifier" not in export


def test_official_shap_tree_explainer_does_not_claim_unvalidated_support(monkeypatch):
    """The official adapter should stop clearly instead of presenting approximate SHAP as official."""

    monkeypatch.setattr(mb, "is_shap_available", lambda: True)
    import mpsboost.shap as shap_module

    monkeypatch.setattr(shap_module, "is_shap_available", lambda: True)

    with pytest.raises(RuntimeError, match="adapter validation is not enabled"):
        mb.official_shap_tree_explainer(_fitted_regressor())
