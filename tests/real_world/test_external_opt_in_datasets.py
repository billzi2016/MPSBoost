"""Opt-in external real-world dataset acceptance tests.

These tests never download data implicitly. They run only when the corresponding ignored local
cache already exists, and otherwise skip with copy-paste setup commands.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

import mpsboost as mb

pandas = pytest.importorskip("pandas")
sklearn_datasets = pytest.importorskip("sklearn.datasets")
sklearn_metrics = pytest.importorskip("sklearn.metrics")
sklearn_model_selection = pytest.importorskip("sklearn.model_selection")
sklearn_preprocessing = pytest.importorskip("sklearn.preprocessing")

from .download_datasets import DATA_ROOT, MANIFEST_ROOT


def _load_cached_openml(name: str, *, version: int):
    """Load one OpenML dataset from the ignored cache without network access."""

    manifest_path = MANIFEST_ROOT / f"{name.lower().replace('-', '_')}_manifest.json"
    if not manifest_path.is_file():
        pytest.skip(
            f"{name} cache manifest is missing; run "
            f"`python tests/real_world/download_datasets.py {_download_slug(name)}`"
        )
    try:
        return sklearn_datasets.fetch_openml(
            name=name,
            version=version,
            data_home=DATA_ROOT / "openml",
            as_frame=True,
            parser="auto",
        )
    except (OSError, ValueError) as exc:
        pytest.skip(
            f"{name} cache is incomplete; rerun "
            f"`python tests/real_world/download_datasets.py {_download_slug(name)}`"
        )
        raise AssertionError("pytest.skip should stop execution") from exc


def _download_slug(name: str) -> str:
    """Return the explicit download command slug for one OpenML dataset."""

    if name == "mnist_784":
        return "mnist-subset"
    if name == "adult":
        return "adult-income"
    return name


def test_mnist_subset_multiclass_acceptance():
    """Cached MNIST subset should exercise flattened-image multiclass classification."""

    dataset = _load_cached_openml("mnist_784", version=1)
    X = np.asarray(dataset.data.iloc[:2000], dtype=np.float32) / 255.0
    y = np.asarray(dataset.target.iloc[:2000], dtype=np.int64)
    X_train, X_test, y_train, y_test = sklearn_model_selection.train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=1881,
        stratify=y,
    )
    model = mb.GradientBoostingClassifier(
        n_estimators=10,
        learning_rate=0.08,
        max_depth=2,
        max_bins=32,
        min_samples_leaf=16,
        min_child_weight=0.0,
        max_delta_step=0.25,
        random_state=1881,
        device="cpu",
    ).fit(X_train, y_train)

    assert model.training_summary_["strategy"] == "native_softmax"
    assert float(sklearn_metrics.accuracy_score(y_test, model.predict(X_test))) >= 0.45


def test_titanic_missing_and_categorical_acceptance():
    """Cached Titanic should exercise categorical features and missing-value handling."""

    dataset = _load_cached_openml("titanic", version=1)
    frame = dataset.frame[["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]].copy()
    frame["age"] = frame["age"].astype(float)
    frame["fare"] = frame["fare"].astype(float)
    y = np.asarray(dataset.target.astype(int), dtype=np.int64)
    X = frame.to_numpy(dtype=object)
    categorical = [1, 6]
    X_train, X_test, y_train, y_test = sklearn_model_selection.train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=1882,
        stratify=y,
    )
    model = mb.CatBoostClassifier(
        n_estimators=12,
        learning_rate=0.12,
        max_depth=2,
        max_bins=32,
        min_samples_leaf=8,
        min_child_weight=0.0,
        cat_features=categorical,
        random_state=1882,
        device="cpu",
    ).fit(X_train, y_train)

    assert model.training_summary_["cat_features"] == categorical
    assert float(sklearn_metrics.accuracy_score(y_test, model.predict(X_test))) >= 0.65


def test_adult_income_categorical_acceptance():
    """Cached Adult Income should exercise larger mixed categorical binary classification."""

    dataset = _load_cached_openml("adult", version=2)
    frame = dataset.frame.drop(columns=["class"]).iloc[:5000].copy()
    y = (dataset.target.iloc[:5000].astype(str).str.contains(">50K")).astype(np.int64).to_numpy()
    categorical = [
        index
        for index, dtype in enumerate(frame.dtypes)
        if not pandas.api.types.is_numeric_dtype(dtype)
    ]
    X = frame.to_numpy(dtype=object)
    X_train, X_test, y_train, y_test = sklearn_model_selection.train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=1883,
        stratify=y,
    )
    model = mb.RandomForestClassifier(
        n_estimators=5,
        max_depth=4,
        max_bins=32,
        min_samples_leaf=16,
        min_child_weight=0.0,
        categorical_features=categorical,
        sample_fraction=0.8,
        bootstrap=True,
        random_state=1883,
        device="cpu",
    ).fit(X_train, y_train)

    assert model.training_summary_["categorical_features"] == categorical
    assert float(sklearn_metrics.accuracy_score(y_test, model.predict(X_test))) >= 0.70


def test_higgs_subset_large_numeric_acceptance():
    """Local HIGGS subset should exercise large numeric binary classification."""

    path = DATA_ROOT / "higgs" / "HIGGS.csv.gz"
    if not path.is_file():
        pytest.skip(
            "HIGGS cache is missing; place HIGGS.csv.gz under tests/real_world/data/higgs/"
        )
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = np.loadtxt(handle, delimiter=",", max_rows=5000, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] < 2:
        raise ValueError("HIGGS.csv.gz must contain label plus numeric features")
    y = rows[:, 0].astype(np.int64)
    X = rows[:, 1:].astype(np.float32, copy=False)
    X_train, X_test, y_train, y_test = sklearn_model_selection.train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=1884,
        stratify=y,
    )
    scaler = sklearn_preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32, copy=False)
    X_test = scaler.transform(X_test).astype(np.float32, copy=False)
    model = mb.GradientBoostingClassifier(
        n_estimators=10,
        learning_rate=0.1,
        max_depth=2,
        max_bins=64,
        min_samples_leaf=32,
        min_child_weight=0.0,
        random_state=1884,
        device="cpu",
    ).fit(X_train, y_train)

    assert float(sklearn_metrics.roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])) >= 0.55
