"""Estimator 参数、输入和状态原子性测试。

本文件只验证 Python 用户契约；训练正确性由 trainer 与真实 Metal 测试覆盖，不使用假
native handle 或 mock 设备让公共 fit 路径成功。
"""

import numpy as np
import pytest

from mpsboost import (
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


def test_feature_importance_reads_native_tree_splits():
    """feature_importances_ must be derived from real fitted tree nodes, not a mock summary."""

    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    model = MPSBoostRegressor(
        n_estimators=1,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        device="cpu",
    ).fit(X, y)

    gain_importance = model.feature_importances_
    split_importance = model.feature_importance(kind="split")

    assert gain_importance.shape == (2,)
    assert split_importance.shape == (2,)
    assert np.isclose(float(gain_importance.sum()), 1.0)
    assert np.isclose(float(split_importance.sum()), 1.0)
    assert gain_importance[0] == pytest.approx(1.0)
    assert split_importance[0] == pytest.approx(1.0)


def test_feature_importance_requires_fitted_model_and_valid_kind():
    """feature importance should share the estimator fitted-state contract."""

    model = MPSBoostRegressor(device="cpu")
    with pytest.raises(NotFittedError):
        _ = model.feature_importances_

    fitted = MPSBoostRegressor(
        n_estimators=1,
        max_depth=0,
        min_samples_leaf=1,
        device="cpu",
    ).fit(np.ones((2, 1), dtype=np.float32), np.array([1.0, 2.0]))
    assert fitted.feature_importances_.tolist() == [0.0]
    with pytest.raises(ValueError, match="feature importance kind"):
        fitted.feature_importance(kind="permutation")


def test_permutation_importance_uses_estimator_score_for_regression():
    """Permutation importance should use score() instead of duplicating prediction metrics."""

    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0.0, 0.0, 5.0, 5.0, 10.0, 10.0], dtype=np.float32)
    model = MPSBoostRegressor(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        device="cpu",
    ).fit(X, y)

    result = model.permutation_importance(X, y, n_repeats=4, random_state=7)

    assert result["importances"].shape == (2, 4)
    assert np.isfinite(result["baseline_score"])
    assert result["importances_mean"][0] > result["importances_mean"][1]


def test_permutation_importance_uses_classifier_accuracy():
    """Classifier permutation importance should reuse accuracy score through the shared method."""

    X = np.array([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]], dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    model = MPSBoostClassifier(
        n_estimators=3,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        device="cpu",
    ).fit(X, y)

    result = model.permutation_importance(X, y, n_repeats=3, random_state=11)

    assert result["importances"].shape == (1, 3)
    assert result["baseline_score"] == 1.0
    assert result["importances_mean"][0] >= 0.0


def test_permutation_importance_validates_state_and_arguments():
    """Permutation importance should share fitted-state and input validation contracts."""

    model = MPSBoostRegressor(device="cpu")
    with pytest.raises(NotFittedError):
        model.permutation_importance(np.ones((2, 1), dtype=np.float32), np.ones(2))

    fitted = MPSBoostRegressor(
        n_estimators=1,
        max_depth=0,
        min_samples_leaf=1,
        device="cpu",
    ).fit(np.ones((2, 1), dtype=np.float32), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="n_repeats"):
        fitted.permutation_importance(
            np.ones((2, 1), dtype=np.float32), np.array([1.0, 2.0]), n_repeats=0
        )
    with pytest.raises(ValueError, match="feature count"):
        fitted.permutation_importance(
            np.ones((2, 2), dtype=np.float32), np.array([1.0, 2.0])
        )


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


def test_binary_classifier_trains_predicts_probabilities_and_scores():
    """GradientBoostingClassifier must use the real native binary-logistic objective."""

    X = np.array(
        [[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]],
        dtype=np.float32,
    )
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    model = MPSBoostClassifier(
        n_estimators=3,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        device="cpu",
    ).fit(X, y)

    probabilities = model.predict_proba(X)
    predictions = model.predict(X)

    assert model.classes_.tolist() == [0, 1]
    assert probabilities.shape == (6, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[:3, 1].mean() < probabilities[3:, 1].mean()
    assert predictions.tolist() == y.tolist()
    assert model.score(X, y) == 1.0
    assert model.feature_importances_.shape == (1,)


def test_binary_classifier_rejects_non_binary_training_labels():
    """Classifier fit should fail before native training when labels are outside strict 0/1."""

    model = MPSBoostClassifier(device="cpu")
    X = np.ones((3, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="exactly 0 and 1"):
        model.fit(X, np.array([0, 1, 2]))
    with pytest.raises(ValueError, match="requires both class"):
        model.fit(X, np.array([1, 1, 1]))


def test_decision_tree_regressor_trains_exactly_one_native_tree():
    """DecisionTreeRegressor should expose a one-tree estimator without duplicating training."""

    X = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32)
    y = np.array([0.0, 0.0, 4.0, 4.0], dtype=np.float32)
    model = DecisionTreeRegressor(
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        device="cpu",
    ).fit(X, y)

    assert model.n_estimators_ == 1
    assert model.get_params()["max_depth"] == 1
    assert "n_estimators" not in model.get_params()
    assert model.feature_importances_.tolist() == [1.0]
    assert model.score(X, y) > 0.0


def test_decision_tree_classifier_trains_exactly_one_native_tree():
    """DecisionTreeClassifier should reuse the binary-logistic one-tree objective."""

    X = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    model = DecisionTreeClassifier(
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        device="cpu",
    ).fit(X, y)

    assert model.n_estimators_ == 1
    assert model.predict_proba(X).shape == (4, 2)
    assert model.predict(X).tolist() == y.tolist()
    assert model.score(X, y) == 1.0


def test_extra_tree_estimators_use_random_threshold_strategy():
    """Single ExtraTree estimators should be seeded native random-threshold trees."""

    X = np.array([[float(value)] for value in range(8)], dtype=np.float32)
    y_reg = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0], dtype=np.float32)
    first = ExtraTreeRegressor(
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        reg_lambda=0.0,
        random_state=101,
        device="cpu",
    ).fit(X, y_reg)
    second = ExtraTreeRegressor(
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        reg_lambda=0.0,
        random_state=101,
        device="cpu",
    ).fit(X, y_reg)
    different = ExtraTreeRegressor(
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        reg_lambda=0.0,
        random_state=202,
        device="cpu",
    ).fit(X, y_reg)

    assert first.model_.trees[0].nodes == second.model_.trees[0].nodes
    assert (
        first.model_.trees[0].nodes[0]["threshold_bin"]
        != different.model_.trees[0].nodes[0]["threshold_bin"]
    )

    y_clf = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    classifier = ExtraTreeClassifier(
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        random_state=101,
        device="cpu",
    ).fit(X, y_clf)
    assert classifier.predict_proba(X).shape == (8, 2)


def test_random_forest_regressor_trains_independent_native_trees():
    """RandomForestRegressor should aggregate real native decision trees."""

    X = np.array(
        [
            [0.0, 1.0],
            [0.1, 1.0],
            [1.0, 0.0],
            [1.1, 0.0],
            [2.0, 1.0],
            [2.1, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0.0, 0.0, 4.0, 4.0, 8.0, 8.0], dtype=np.float32)
    model = RandomForestRegressor(
        n_estimators=3,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        max_features=1.0,
        sample_fraction=1.0,
        random_state=3,
        device="cpu",
    ).fit(X, y)

    predictions = model.predict(X)

    assert model.n_estimators_ == 3
    assert len(model.estimators_) == 3
    assert predictions.shape == (6,)
    assert model.score(X, y) > 0.0
    assert model.feature_importances_.shape == (2,)


def test_random_forest_classifier_trains_independent_native_trees():
    """RandomForestClassifier should average real native tree probabilities."""

    X = np.array([[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]], dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    model = RandomForestClassifier(
        n_estimators=3,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        sample_fraction=1.0,
        random_state=5,
        device="cpu",
    ).fit(X, y)

    probabilities = model.predict_proba(X)

    assert model.n_estimators_ == 3
    assert probabilities.shape == (6, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[:3, 1].mean() < probabilities[3:, 1].mean()
    assert model.score(X, y) >= 5.0 / 6.0


def test_random_forest_random_state_is_deterministic():
    """The same random_state should reproduce row and feature sampling exactly."""

    X = np.array(
        [[0.0, 1.0], [0.1, 1.0], [1.0, 0.0], [1.1, 0.0], [2.0, 1.0], [2.1, 1.0]],
        dtype=np.float32,
    )
    y = np.array([0.0, 0.0, 4.0, 4.0, 8.0, 8.0], dtype=np.float32)
    parameters = dict(
        n_estimators=3,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        max_features=0.5,
        sample_fraction=0.8,
        random_state=23,
        device="cpu",
    )
    first = RandomForestRegressor(**parameters).fit(X, y)
    second = RandomForestRegressor(**parameters).fit(X, y)

    np.testing.assert_allclose(first.predict(X), second.predict(X))
    assert [item.tolist() for item in first.feature_subsets_] == [
        item.tolist() for item in second.feature_subsets_
    ]


def test_random_forest_n_jobs_preserves_deterministic_results():
    """Parallel tree fitting should preserve the deterministic sampling contract."""

    X = np.array(
        [[0.0, 1.0], [0.1, 1.0], [1.0, 0.0], [1.1, 0.0], [2.0, 1.0], [2.1, 1.0]],
        dtype=np.float32,
    )
    y = np.array([0.0, 0.0, 4.0, 4.0, 8.0, 8.0], dtype=np.float32)
    parameters = dict(
        n_estimators=4,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        max_features=0.5,
        sample_fraction=0.8,
        random_state=29,
        device="cpu",
    )
    serial = RandomForestRegressor(n_jobs=1, **parameters).fit(X, y)
    parallel = RandomForestRegressor(n_jobs=2, **parameters).fit(X, y)

    np.testing.assert_allclose(serial.predict(X), parallel.predict(X))
    assert parallel.training_summary_["n_jobs"] == 2
    assert [item.tolist() for item in serial.feature_subsets_] == [
        item.tolist() for item in parallel.feature_subsets_
    ]


def test_random_forest_validates_parameters():
    """Forest parameter boundaries should fail explicitly."""

    X = np.ones((4, 1), dtype=np.float32)
    y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="max_features"):
        RandomForestRegressor(max_features=0.0, device="cpu").fit(X, y)
    with pytest.raises(TypeError, match="bootstrap"):
        RandomForestRegressor(bootstrap=1, device="cpu").fit(X, y)
    with pytest.raises(ValueError, match="n_jobs"):
        RandomForestRegressor(n_jobs=0, device="cpu").fit(X, y)


def test_extra_trees_regressor_and_classifier_share_forest_contracts():
    """ExtraTrees ensembles should reuse RF aggregation with native random-threshold trees."""

    X = np.array([[float(value)] for value in range(8)], dtype=np.float32)
    y_reg = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0], dtype=np.float32)
    regressor = ExtraTreesRegressor(
        n_estimators=3,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        sample_fraction=1.0,
        random_state=303,
        device="cpu",
    ).fit(X, y_reg)
    assert regressor.predict(X).shape == (8,)
    assert regressor.feature_importances_.shape == (1,)

    y_clf = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    classifier = ExtraTreesClassifier(
        n_estimators=3,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        sample_fraction=1.0,
        random_state=307,
        device="cpu",
    ).fit(X, y_clf)
    assert classifier.predict_proba(X).shape == (8, 2)
