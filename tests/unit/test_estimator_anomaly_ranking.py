"""Estimator tests for anomaly detection and learning-to-rank families."""

import warnings

import numpy as np

import mpsboost as mb


def test_isolation_forest_scores_injected_outliers_higher():
    """Injected distant rows should receive larger anomaly scores than the dense cluster."""

    rng = np.random.default_rng(7)
    normal = rng.normal(0.0, 0.2, size=(80, 3)).astype(np.float32)
    outliers = np.array([[4.0, 4.0, 4.0], [-4.0, -4.0, -4.0]], dtype=np.float32)
    X = np.vstack([normal, outliers])

    model = mb.MPSIsolationForest(
        n_estimators=64,
        max_samples=0.75,
        contamination=0.05,
        random_state=11,
    ).fit(X)

    scores = model.anomaly_score(X)
    assert scores[-2:].mean() > scores[:-2].mean()
    assert set(model.predict(X[-2:])) == {-1}
    assert model.training_summary_["objective"] == "path_length"
    assert model.device_ == "cpu"


def test_isolation_forest_is_deterministic_and_sklearn_style():
    """A fixed random state should produce stable scores and support get/set params."""

    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 4)).astype(np.float32)
    first = mb.IsolationForest(n_estimators=16, random_state=5).fit(X)
    second = mb.IsolationForest(n_estimators=16, random_state=5).fit(X)

    np.testing.assert_allclose(first.anomaly_score(X), second.anomaly_score(X))
    assert first.get_params()["n_estimators"] == 16
    first.set_params(n_estimators=8)
    assert not hasattr(first, "trees_")


def test_isolation_forest_mps_request_warns_and_falls_back_to_cpu():
    """MPS requests should stay runnable while disclosing the faster CPU-suitable path."""

    X = np.arange(24, dtype=np.float32).reshape(12, 2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = mb.MPSIsolationForest(device="mps", n_estimators=8, random_state=0).fit(X)

    assert model.device_ == "cpu"
    assert model.training_summary_["requested_device"] == "mps"
    assert "fallback_reason" in model.training_summary_
    assert any("CPU backend selected" in str(item.message) for item in caught)


def test_learning_to_rank_validates_groups_and_scores_ndcg():
    """Ranking fit should enforce query groups and report full-list NDCG."""

    X = np.array(
        [
            [3.0, 0.0],
            [2.0, 0.0],
            [1.0, 0.0],
            [0.0, 3.0],
            [0.0, 2.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.array([3.0, 2.0, 0.0, 3.0, 1.0, 0.0], dtype=np.float32)
    group = np.array([3, 3], dtype=np.int64)

    model = mb.LearningToRankRegressor(
        n_estimators=20,
        learning_rate=0.2,
        max_depth=2,
        min_samples_leaf=1,
        random_state=4,
    ).fit(X, y, group=group)

    assert 0.0 <= model.score(X, y, group=group) <= 1.0
    assert model.score(X, y, group=group) > 0.8
    assert model.training_summary_["objective"] == "pointwise_ranking"
    assert model.training_summary_["query_count"] == 2

    try:
        model.fit(X, y, group=np.array([2, 2], dtype=np.int64))
    except ValueError as exc:
        assert "sum" in str(exc)
    else:
        raise AssertionError("invalid ranking group did not fail")


def test_learning_to_rank_mps_request_warns_and_falls_back_to_cpu():
    """Ranking should keep MPS-requested runs executable via explicit CPU-suitable routing."""

    X = np.arange(24, dtype=np.float32).reshape(8, 3)
    y = np.array([3, 2, 1, 0, 0, 1, 2, 3], dtype=np.float32)
    group = np.array([4, 4], dtype=np.int64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = mb.LearningToRankRegressor(
            device="mps",
            n_estimators=8,
            max_depth=2,
            min_samples_leaf=1,
            random_state=2,
        ).fit(X, y, group=group)

    assert model.device == "mps"
    assert model.training_summary_["device"] == "cpu"
    assert model.training_summary_["requested_device"] == "mps"
    assert "fallback_reason" in model.training_summary_
    assert any("CPU backend selected" in str(item.message) for item in caught)
