"""Cached Covertype subset acceptance and CPU/MPS parity checks."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pytest

import mpsboost as mb

sklearn_datasets = pytest.importorskip("sklearn.datasets")
sklearn_metrics = pytest.importorskip("sklearn.metrics")
sklearn_model_selection = pytest.importorskip("sklearn.model_selection")
sklearn_preprocessing = pytest.importorskip("sklearn.preprocessing")

from .download_datasets import DATA_ROOT


def _load_cached_covertype():
    """Load Covertype from ignored local cache without network access."""

    data_home = DATA_ROOT / "sklearn"
    try:
        return sklearn_datasets.fetch_covtype(
            data_home=data_home,
            download_if_missing=False,
        )
    except OSError as exc:
        pytest.skip(
            "Covertype cache is missing; run "
            "`python tests/real_world/download_datasets.py covertype-subset`"
        )
        raise AssertionError("pytest.skip should stop execution") from exc


def _covertype_subset(rows: int = 30000):
    """Return a deterministic large-row subset with zero-based class labels."""

    dataset = _load_cached_covertype()
    X = dataset.data[:rows].astype(np.float32, copy=False)
    y = (dataset.target[:rows] - 1).astype(np.int64, copy=False)
    return sklearn_model_selection.train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=1821,
        stratify=y,
    )


def test_covertype_subset_multiclass_cpu_acceptance():
    """Large-row multiclass tabular data should train through native CPU softmax."""

    X_train, X_test, y_train, y_test = _covertype_subset(rows=30000)
    scaler = sklearn_preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32, copy=False)
    X_test = scaler.transform(X_test).astype(np.float32, copy=False)
    model = mb.GradientBoostingClassifier(
        n_estimators=8,
        learning_rate=0.12,
        max_depth=2,
        max_bins=64,
        min_samples_leaf=32,
        min_child_weight=0.0,
        reg_lambda=1.0,
        max_delta_step=0.5,
        random_state=1821,
        device="cpu",
    ).fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert model.training_summary_["strategy"] == "native_softmax"
    assert len(model.classes_) >= 4
    assert float(sklearn_metrics.accuracy_score(y_test, predictions)) >= 0.45


@pytest.mark.gpu
def test_covertype_subset_real_mps_parity_and_timing_smoke():
    """Large-row Covertype should run on MPS and stay numerically aligned with CPU."""

    X_train, X_test, y_train, _ = _covertype_subset(rows=30000)
    scaler = sklearn_preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32, copy=False)
    X_test = scaler.transform(X_test).astype(np.float32, copy=False)
    parameters = dict(
        n_estimators=4,
        learning_rate=0.12,
        max_depth=2,
        max_bins=64,
        min_samples_leaf=32,
        min_child_weight=0.0,
        reg_lambda=1.0,
        max_delta_step=0.5,
        random_state=1822,
    )

    started = perf_counter()
    cpu = mb.GradientBoostingClassifier(
        device="cpu", multi_strategy="ovr", **parameters
    ).fit(X_train, y_train)
    cpu_seconds = perf_counter() - started
    started = perf_counter()
    mps = mb.GradientBoostingClassifier(device="mps", **parameters).fit(X_train, y_train)
    mps_seconds = perf_counter() - started

    assert cpu.training_summary_["strategy"] == "one_vs_rest"
    assert mps.training_summary_["strategy"] == "one_vs_rest"
    np.testing.assert_allclose(
        mps.predict_proba(X_test[:512]),
        cpu.predict_proba(X_test[:512]),
        rtol=2e-4,
        atol=2e-4,
    )
    assert mps.device_ == "mps"
    assert mps_seconds > 0.0
    assert cpu_seconds > 0.0
