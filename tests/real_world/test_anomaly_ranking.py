"""Real-world smoke tests for anomaly detection and ranking estimators."""

import warnings

import numpy as np
import pytest

import mpsboost as mb

sklearn_datasets = pytest.importorskip("sklearn.datasets")
sklearn_model_selection = pytest.importorskip("sklearn.model_selection")
sklearn_preprocessing = pytest.importorskip("sklearn.preprocessing")


def test_breast_cancer_isolation_forest_real_world_scores_are_finite():
    """Isolation forest should produce stable finite anomaly scores on real tabular data."""

    dataset = sklearn_datasets.load_breast_cancer()
    X = dataset.data.astype(np.float32, copy=False)
    scaler = sklearn_preprocessing.StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32, copy=False)

    model = mb.MPSIsolationForest(
        n_estimators=32,
        max_samples=0.4,
        max_features=0.75,
        contamination=0.08,
        random_state=1871,
        device="cpu",
    ).fit(X)
    scores = model.anomaly_score(X)
    predictions = model.predict(X)

    assert scores.shape == (X.shape[0],)
    assert np.all(np.isfinite(scores))
    assert set(np.unique(predictions)) <= {-1, 1}
    assert np.mean(predictions == -1) > 0.0


def test_diabetes_learning_to_rank_real_world_group_score_is_finite():
    """Pointwise ranking should validate groups and return finite NDCG on real rows."""

    dataset = sklearn_datasets.load_diabetes()
    X = dataset.data.astype(np.float32, copy=False)
    y = dataset.target.astype(np.float32, copy=False)
    X_train, X_test, y_train, y_test = sklearn_model_selection.train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=1872,
    )
    train_group = np.full(8, X_train.shape[0] // 8, dtype=np.int64)
    train_group[-1] += X_train.shape[0] - int(train_group.sum())
    test_group = np.full(4, X_test.shape[0] // 4, dtype=np.int64)
    test_group[-1] += X_test.shape[0] - int(test_group.sum())

    model = mb.LearningToRankRegressor(
        n_estimators=12,
        learning_rate=0.08,
        max_depth=2,
        max_bins=32,
        min_samples_leaf=4,
        min_child_weight=0.0,
        random_state=1872,
        device="cpu",
    ).fit(X_train, y_train, group=train_group)
    score = model.score(X_test, y_test, group=test_group)

    assert np.isfinite(score)
    assert 0.0 <= score <= 1.0
    assert model.training_summary_["objective"] == "pointwise_ranking"


def test_real_world_mps_requests_route_s17_models_to_cpu_suitable_path():
    """S17 model families should keep MPS-requested runs executable on CPU-suitable routing."""

    dataset = sklearn_datasets.load_diabetes()
    X = dataset.data[:32].astype(np.float32, copy=False)
    y = dataset.target[:32].astype(np.float32, copy=False)
    group = np.array([8, 8, 8, 8], dtype=np.int64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        anomaly = mb.MPSIsolationForest(
            n_estimators=8,
            max_samples=0.5,
            random_state=1873,
            device="mps",
        ).fit(X)
        ranking = mb.LearningToRankRegressor(
            n_estimators=6,
            max_depth=2,
            min_samples_leaf=2,
            min_child_weight=0.0,
            random_state=1873,
            device="mps",
        ).fit(X, y, group=group)

    assert anomaly.training_summary_["device"] == "cpu"
    assert ranking.training_summary_["device"] == "cpu"
    assert len(caught) == 2
    assert all("CPU backend selected" in str(item.message) for item in caught)
