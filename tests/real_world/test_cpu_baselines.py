"""Strong CPU baseline comparisons for bundled real-world datasets.

These tests keep sklearn baselines visible beside the MPSBoost CPU oracle. They
do not turn quality comparison into a leaderboard gate; instead they ensure each
record contains the dataset, metric, project score, baseline score, and explicit
gap that the public real-world report can surface honestly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import mpsboost as mb

sklearn_datasets = pytest.importorskip("sklearn.datasets")
sklearn_ensemble = pytest.importorskip("sklearn.ensemble")
sklearn_metrics = pytest.importorskip("sklearn.metrics")
sklearn_model_selection = pytest.importorskip("sklearn.model_selection")
sklearn_preprocessing = pytest.importorskip("sklearn.preprocessing")


@dataclass(frozen=True)
class BaselineRecord:
    """One comparable real-world score row for the S18 CPU baseline gate."""

    dataset: str
    task: str
    metric: str
    project_score: float
    sklearn_score: float

    @property
    def quality_gap(self) -> float:
        """Return positive values when MPSBoost is ahead of the sklearn baseline."""

        return self.project_score - self.sklearn_score


def test_breast_cancer_classifier_records_sklearn_cpu_baseline():
    """Binary classification should record project CPU oracle and sklearn quality."""

    dataset = sklearn_datasets.load_breast_cancer()
    X = dataset.data.astype(np.float32, copy=False)
    y = dataset.target.astype(np.int64, copy=False)
    X_train, X_test, y_train, y_test = sklearn_model_selection.train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=1861,
        stratify=y,
    )
    scaler = sklearn_preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32, copy=False)
    X_test = scaler.transform(X_test).astype(np.float32, copy=False)

    project = mb.GradientBoostingClassifier(
        n_estimators=12,
        learning_rate=0.1,
        max_depth=2,
        max_bins=64,
        min_samples_leaf=8,
        min_child_weight=0.0,
        random_state=1861,
        device="cpu",
    ).fit(X_train, y_train)
    baseline = sklearn_ensemble.GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=2,
        random_state=1861,
    ).fit(X_train, y_train)
    record = BaselineRecord(
        dataset="Breast Cancer Wisconsin",
        task="binary_classification",
        metric="roc_auc",
        project_score=float(
            sklearn_metrics.roc_auc_score(y_test, project.predict_proba(X_test)[:, 1])
        ),
        sklearn_score=float(
            sklearn_metrics.roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1])
        ),
    )

    assert record.project_score >= 0.90
    assert record.sklearn_score >= 0.90
    assert np.isfinite(record.quality_gap)


def test_diabetes_regressor_records_sklearn_cpu_baseline():
    """Regression should record project CPU oracle and sklearn quality."""

    dataset = sklearn_datasets.load_diabetes()
    X = dataset.data.astype(np.float32, copy=False)
    y = dataset.target.astype(np.float32, copy=False)
    X_train, X_test, y_train, y_test = sklearn_model_selection.train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=1862,
    )
    project = mb.GradientBoostingRegressor(
        n_estimators=32,
        learning_rate=0.08,
        max_depth=2,
        max_bins=64,
        min_samples_leaf=6,
        min_child_weight=0.0,
        random_state=1862,
        device="cpu",
    ).fit(X_train, y_train)
    baseline = sklearn_ensemble.HistGradientBoostingRegressor(
        max_iter=80,
        learning_rate=0.05,
        max_leaf_nodes=15,
        random_state=1862,
    ).fit(X_train, y_train)
    record = BaselineRecord(
        dataset="Diabetes",
        task="regression",
        metric="r2",
        project_score=float(sklearn_metrics.r2_score(y_test, project.predict(X_test))),
        sklearn_score=float(sklearn_metrics.r2_score(y_test, baseline.predict(X_test))),
    )

    assert record.project_score >= 0.15
    assert record.sklearn_score >= 0.15
    assert np.isfinite(record.quality_gap)
