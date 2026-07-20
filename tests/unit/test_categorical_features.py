"""Categorical feature semantics for shared tree estimators."""

import numpy as np
import pytest

from mpsboost import (
    CatBoostClassifier,
    CatBoostRegressor,
    DecisionTreeRegressor,
    MPSBoostRegressor,
    RandomForestRegressor,
)


def test_regressor_orders_categorical_feature_by_weighted_target_mean():
    """Categorical columns should train through deterministic ordered native splits."""

    X = np.asarray(
        [["low"], ["low"], ["mid"], ["mid"], ["high"], ["high"]],
        dtype=object,
    )
    y = np.asarray([0.0, 0.0, 5.0, 5.0, 10.0, 10.0], dtype=np.float32)
    model = MPSBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=2,
        min_samples_leaf=1,
        min_child_weight=0.0,
        categorical_features=[0],
        device="cpu",
    ).fit(X, y)

    predictions = model.predict(np.asarray([["low"], ["mid"], ["high"]], dtype=object))

    assert model.training_summary_["categorical_features"] == [0]
    assert model.categorical_metadata_.mappings[0].categories == ("low", "mid", "high")
    assert predictions[0] < predictions[1] < predictions[2]


def test_unknown_category_uses_native_missing_default_direction():
    """Unknown prediction categories should encode as NaN and reuse default directions."""

    X = np.asarray([["left"], ["left"], ["right"], ["right"]], dtype=object)
    y = np.asarray([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    model = DecisionTreeRegressor(
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        categorical_features=[0],
        device="cpu",
    ).fit(X, y)

    prediction = model.predict(np.asarray([["unseen"]], dtype=object))

    assert np.isfinite(prediction[0])


def test_categorical_model_save_fails_until_metadata_format_exists(tmp_path):
    """Categorical mappings must not be silently dropped by native-only model files."""

    X = np.asarray([["a"], ["a"], ["b"], ["b"]], dtype=object)
    y = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    model = MPSBoostRegressor(
        n_estimators=1,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        categorical_features=[0],
        device="cpu",
    ).fit(X, y)

    with pytest.raises(NotImplementedError, match="categorical model persistence"):
        model.save_model(tmp_path / "categorical.mb")


def test_categorical_features_validate_indices_and_numeric_columns():
    """Invalid categorical declarations should fail before native training."""

    X = np.asarray([["a", "not-number"], ["b", "also-not-number"]], dtype=object)
    y = np.asarray([0.0, 1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="out of range"):
        MPSBoostRegressor(categorical_features=[2], device="cpu").fit(X[:, :1], y)
    with pytest.raises(ValueError, match="duplicates"):
        MPSBoostRegressor(categorical_features=[0, 0], device="cpu").fit(X, y)
    with pytest.raises(TypeError, match="non-categorical"):
        MPSBoostRegressor(categorical_features=[0], device="cpu").fit(X, y)


def test_catboost_cat_features_alias_uses_shared_categorical_encoder():
    """CatBoost-style cat_features should route to the same categorical semantics."""

    X = np.asarray([["a"], ["a"], ["b"], ["b"]], dtype=object)
    y_reg = np.asarray([0.0, 0.0, 4.0, 4.0], dtype=np.float32)
    regressor = CatBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        cat_features=[0],
        device="cpu",
    ).fit(X, y_reg)
    assert regressor.predict(np.asarray([["a"], ["b"]], dtype=object))[0] < regressor.predict(
        np.asarray([["a"], ["b"]], dtype=object)
    )[1]
    assert regressor.training_summary_["cat_features"] == [0]

    y_clf = np.asarray([0, 0, 1, 1], dtype=np.int64)
    classifier = CatBoostClassifier(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        cat_features=[0],
        device="cpu",
    ).fit(X, y_clf)
    probabilities = classifier.predict_proba(np.asarray([["a"], ["b"]], dtype=object))
    assert probabilities[0, 1] < probabilities[1, 1]


def test_random_forest_encodes_categorical_features_once_before_tree_sampling():
    """Forest feature sampling should consume the shared encoded matrix without re-encoding."""

    X = np.asarray(
        [["x", 0.0], ["x", 0.1], ["y", 1.0], ["y", 1.1], ["z", 2.0], ["z", 2.1]],
        dtype=object,
    )
    y = np.asarray([0.0, 0.0, 4.0, 4.0, 8.0, 8.0], dtype=np.float32)
    model = RandomForestRegressor(
        n_estimators=2,
        max_depth=2,
        min_samples_leaf=1,
        min_child_weight=0.0,
        max_features=1.0,
        sample_fraction=1.0,
        bootstrap=False,
        categorical_features=[0],
        random_state=41,
        device="cpu",
    ).fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == (6,)
    assert model.training_summary_["categorical_features"] == [0]
    assert float(predictions[:2].mean()) < float(predictions[-2:].mean())
