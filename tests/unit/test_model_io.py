"""版本化模型保存、加载与损坏拒绝测试。

测试通过正式 estimator 和 native loader 读写真实文件；不使用 pickle、mock 文件系统或
绕过校验的节点构造。失败加载必须保留调用方原有模型状态。
"""

import os

import numpy as np
import pytest

from mpsboost import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    MPSBoostClassifier,
    MPSBoostRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)


def _fitted_model():
    """构造小型真实 CPU oracle 模型，避免多个用例复制训练参数。"""

    X = np.arange(12, dtype=np.float32).reshape(6, 2)
    y = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    model = MPSBoostRegressor(
        n_estimators=3,
        learning_rate=0.5,
        max_depth=2,
        max_bins=8,
        min_samples_leaf=1,
        device="cpu",
    ).fit(X, y)
    return X, model


def test_save_load_round_trip_and_file_permissions(tmp_path):
    """加载后预测逐位一致，模型文件不得包含宽松权限或临时文件。"""

    X, original = _fitted_model()
    path = tmp_path / "model.mb"
    original.save_model(path)
    restored = MPSBoostRegressor(device="cpu").load_model(path)
    np.testing.assert_array_equal(original.predict(X), restored.predict(X))
    assert os.stat(path).st_mode & 0o077 == 0
    assert list(tmp_path.iterdir()) == [path]


def test_classifier_model_round_trip_preserves_probabilities(tmp_path):
    """Binary-logistic models must persist objective metadata for safe classifier loading."""

    X = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    model = MPSBoostClassifier(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        device="cpu",
    ).fit(X, y)
    path = tmp_path / "classifier.mb"
    model.save_model(path)

    restored = MPSBoostClassifier(device="cpu").load_model(path)

    np.testing.assert_allclose(restored.predict_proba(X), model.predict_proba(X))
    assert restored.classes_.tolist() == [0, 1]


def test_regressor_rejects_classifier_model_file(tmp_path):
    """A regressor must not load binary-logistic raw margins as regression values."""

    X = np.array([[0.0], [1.0]], dtype=np.float32)
    path = tmp_path / "classifier.mb"
    MPSBoostClassifier(
        n_estimators=1,
        max_depth=0,
        min_samples_leaf=1,
        device="cpu",
    ).fit(X, np.array([0, 1])).save_model(path)

    with pytest.raises(ValueError, match="incompatible"):
        MPSBoostRegressor(device="cpu").load_model(path)


def test_decision_tree_load_rejects_boosted_ensemble(tmp_path):
    """Decision tree estimators must not accept multi-tree boosted model files."""

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    path = tmp_path / "boosted.mb"
    MPSBoostRegressor(
        n_estimators=2,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        device="cpu",
    ).fit(X, np.array([0.0, 0.0, 2.0, 2.0])).save_model(path)

    with pytest.raises(ValueError, match="one-tree"):
        DecisionTreeRegressor(device="cpu").load_model(path)


def test_decision_tree_classifier_round_trip(tmp_path):
    """One-tree classifier model files should restore as decision tree classifiers."""

    X = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    path = tmp_path / "tree_classifier.mb"
    original = DecisionTreeClassifier(
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        device="cpu",
    ).fit(X, y)
    original.save_model(path)
    restored = DecisionTreeClassifier(device="cpu").load_model(path)

    np.testing.assert_allclose(restored.predict_proba(X), original.predict_proba(X))


def test_random_forest_regressor_round_trip_preserves_predictions(tmp_path):
    """Forest containers should preserve native per-tree model bytes and feature subsets."""

    X = np.array([[0.0, 1.0], [0.1, 1.0], [1.0, 0.0], [1.1, 0.0]], dtype=np.float32)
    y = np.array([0.0, 0.0, 4.0, 4.0], dtype=np.float32)
    original = RandomForestRegressor(
        n_estimators=2,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        sample_fraction=1.0,
        random_state=13,
        device="cpu",
    ).fit(X, y)
    path = tmp_path / "forest_regressor.mb"
    original.save_model(path)
    restored = RandomForestRegressor(device="cpu").load_model(path)

    np.testing.assert_allclose(restored.predict(X), original.predict(X))
    assert restored.feature_subsets_[0].tolist() == original.feature_subsets_[0].tolist()


def test_random_forest_classifier_round_trip_preserves_probabilities(tmp_path):
    """Classifier forest containers should restore class probabilities exactly enough."""

    X = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    original = RandomForestClassifier(
        n_estimators=2,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        sample_fraction=1.0,
        random_state=17,
        device="cpu",
    ).fit(X, y)
    path = tmp_path / "forest_classifier.mb"
    original.save_model(path)
    restored = RandomForestClassifier(device="cpu").load_model(path)

    np.testing.assert_allclose(restored.predict_proba(X), original.predict_proba(X))


def test_random_forest_rejects_incompatible_container_type(tmp_path):
    """Regressor and classifier forest containers must not be interchangeable."""

    X = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    path = tmp_path / "forest_classifier.mb"
    RandomForestClassifier(
        n_estimators=1,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        sample_fraction=1.0,
        random_state=19,
        device="cpu",
    ).fit(X, y).save_model(path)

    with pytest.raises(ValueError, match="incompatible"):
        RandomForestRegressor(device="cpu").load_model(path)


def test_extra_trees_round_trip_uses_forest_container(tmp_path):
    """ExtraTrees should reuse the forest container with random-threshold native trees."""

    X = np.array([[float(value)] for value in range(8)], dtype=np.float32)
    y_reg = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0], dtype=np.float32)
    regressor = ExtraTreesRegressor(
        n_estimators=2,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        sample_fraction=1.0,
        random_state=401,
        device="cpu",
    ).fit(X, y_reg)
    regressor_path = tmp_path / "extra_trees_regressor.mb"
    regressor.save_model(regressor_path)
    restored_regressor = ExtraTreesRegressor(device="cpu").load_model(regressor_path)
    np.testing.assert_allclose(restored_regressor.predict(X), regressor.predict(X))

    y_clf = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    classifier = ExtraTreesClassifier(
        n_estimators=2,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        sample_fraction=1.0,
        random_state=409,
        device="cpu",
    ).fit(X, y_clf)
    classifier_path = tmp_path / "extra_trees_classifier.mb"
    classifier.save_model(classifier_path)
    restored_classifier = ExtraTreesClassifier(device="cpu").load_model(classifier_path)
    np.testing.assert_allclose(
        restored_classifier.predict_proba(X),
        classifier.predict_proba(X),
    )


@pytest.mark.parametrize("mutation", ["truncate", "checksum", "major"])
def test_corrupt_model_is_rejected_without_replacing_existing_state(tmp_path, mutation):
    """截断、内容篡改和未知 major 必须拒绝，已有模型仍可预测。"""

    X, model = _fitted_model()
    expected = model.predict(X)
    path = tmp_path / "model.mb"
    model.save_model(path)
    content = bytearray(path.read_bytes())
    if mutation == "truncate":
        content = content[:-1]
    elif mutation == "checksum":
        content[-1] ^= 0xFF
    else:
        content[8:10] = (999).to_bytes(2, "little")
    path.write_bytes(content)
    with pytest.raises(ValueError, match="长度|校验|版本"):
        model.load_model(path)
    np.testing.assert_array_equal(model.predict(X), expected)
