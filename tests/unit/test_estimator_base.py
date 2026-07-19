"""Shared imports for estimator unit tests."""

import numpy as np
import pytest

from mpsboost import (
    CatBoostClassifier,
    CatBoostRegressor,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    MPSBoostClassifier,
    MPSBoostRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from mpsboost.estimator import NotFittedError




def test_get_and_set_params_follow_estimator_protocol():
    """全部显式构造参数必须可发现，未知参数失败且 set_params 返回自身。"""

    model = MPSBoostRegressor(n_estimators=3, device="cpu")
    assert set(model.get_params()) == set(model._PARAMETER_NAMES)
    assert model.set_params(n_estimators=5) is model
    assert model.n_estimators == 5
    with pytest.raises(ValueError, match="未知参数"):
        model.set_params(unknown=1)


def test_leaf_wise_parameters_are_sklearn_visible_and_train_native_model():
    """Leaf-wise parameters should be constructor-visible and routed to native training."""

    X = np.asarray([[float(value)] for value in range(8)], dtype=np.float32)
    y = np.asarray([0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 11.0, 12.0], dtype=np.float32)
    model = MPSBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=3,
        min_samples_leaf=1,
        min_child_weight=0.0,
        growth_strategy="leaf_wise",
        max_leaves=3,
        device="cpu",
    ).fit(X, y)

    parameters = model.get_params()
    assert parameters["growth_strategy"] == "leaf_wise"
    assert parameters["max_leaves"] == 3
    predictions = model.predict(X)
    assert predictions.shape == (8,)
    assert np.all(np.isfinite(predictions))
    assert float(predictions[:3].mean()) < float(predictions[3:].mean())


def test_unfitted_and_wrong_feature_prediction_fail_explicitly():
    """未拟合与特征数变化必须给出稳定异常。"""

    model = MPSBoostRegressor(
        n_estimators=1, max_depth=0, min_samples_leaf=1, device="cpu"
    )
    with pytest.raises(NotFittedError):
        model.predict(np.ones((1, 1), dtype=np.float32))
    model.fit(np.ones((2, 1), dtype=np.float32), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="feature count"):
        model.predict(np.ones((2, 2), dtype=np.float32))


def test_score_returns_r2_and_requires_fitted_model():
    """score must expose sklearn-style R2 without bypassing fitted-state checks."""

    model = MPSBoostRegressor(
        n_estimators=1, max_depth=0, min_samples_leaf=1, device="cpu"
    )
    with pytest.raises(NotFittedError):
        model.score(np.ones((2, 1), dtype=np.float32), np.array([1.0, 2.0]))

    fitted = model.fit(np.ones((3, 1), dtype=np.float32), np.array([2.0, 2.0, 2.0]))
    assert fitted.score(np.ones((3, 1), dtype=np.float32), np.array([2.0, 2.0, 2.0])) == 1.0
    assert fitted.score(np.ones((3, 1), dtype=np.float32), np.array([1.0, 2.0, 3.0])) == 0.0


def test_sample_weight_changes_regression_training_and_score():
    """Sample weights should flow into native gradients, Hessians, and regression scoring."""

    X = np.ones((3, 1), dtype=np.float32)
    y = np.array([0.0, 10.0, 10.0], dtype=np.float32)
    unweighted = MPSBoostRegressor(
        n_estimators=1,
        max_depth=0,
        min_samples_leaf=1,
        min_child_weight=0.0,
        device="cpu",
    ).fit(X, y)
    weighted = MPSBoostRegressor(
        n_estimators=1,
        max_depth=0,
        min_samples_leaf=1,
        min_child_weight=0.0,
        device="cpu",
    ).fit(X, y, sample_weight=np.array([10.0, 1.0, 1.0]))

    assert float(weighted.predict(X)[0]) < float(unweighted.predict(X)[0])
    assert weighted.training_summary_["weighted"] is True
    assert np.isfinite(weighted.score(X, y, sample_weight=np.array([10.0, 1.0, 1.0])))


def test_sample_weight_validates_shape_values_and_total():
    """Invalid sample weights should fail before device initialization."""

    X = np.ones((3, 1), dtype=np.float32)
    y = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    model = MPSBoostRegressor(device="cpu")
    with pytest.raises(ValueError, match="length"):
        model.fit(X, y, sample_weight=np.ones(2))
    with pytest.raises(ValueError, match="finite"):
        model.fit(X, y, sample_weight=np.array([1.0, np.nan, 1.0]))
    with pytest.raises(ValueError, match="non-negative"):
        model.fit(X, y, sample_weight=np.array([1.0, -1.0, 1.0]))
    with pytest.raises(ValueError, match="positive"):
        model.fit(X, y, sample_weight=np.zeros(3))


def test_sklearn_tags_are_available_without_sklearn_dependency():
    """The estimator should expose old-style lightweight tags without importing sklearn."""

    tags = MPSBoostRegressor(device="cpu")._more_tags()
    assert tags["requires_y"] is True
    assert tags["allow_nan"] is False
    assert tags["X_types"] == ["2darray"]


def test_failed_refit_clears_previous_model_instead_of_exposing_partial_state():
    """失败 refit 不得继续呈现与当前训练请求无关的旧模型。"""

    model = MPSBoostRegressor(
        n_estimators=1, max_depth=0, min_samples_leaf=1, device="cpu"
    ).fit(np.ones((2, 1), dtype=np.float32), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="样本数量"):
        model.fit(np.ones((3, 1), dtype=np.float32), np.array([1.0, 2.0]))
    with pytest.raises(NotFittedError):
        model.predict(np.ones((1, 1), dtype=np.float32))


def test_dense_adapter_rejects_sparse_complex_and_invalid_rank():
    """当前版本范围外输入不能被静默展开或截断。"""

    model = MPSBoostRegressor(device="cpu")
    with pytest.raises(TypeError, match="dtype"):
        model.fit(np.ones((2, 1), dtype=np.complex64), np.ones(2))
    with pytest.raises(ValueError, match="二维"):
        model.fit(np.ones(2), np.ones(2))


def test_auto_device_selects_cpu_for_small_workloads():
    """device='auto' should select CPU for small jobs and record the decision."""

    model = MPSBoostRegressor(
        n_estimators=1,
        max_depth=0,
        min_samples_leaf=1,
        max_bins=16,
        device="auto",
    ).fit(np.ones((3, 1), dtype=np.float32), np.array([1.0, 2.0, 3.0]))

    assert model.device_ == "cpu"
    assert model.device_decision_["requested"] == "auto"
    assert model.device_decision_["selected"] == "cpu"
    assert model.training_summary_["device_decision"] == model.device_decision_
