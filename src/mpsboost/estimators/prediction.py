"""Shared prediction helpers for fitted MPSBoost estimators.

The helpers centralize dense input adaptation, feature-count checks, native
model prediction, feature-subset slicing, and forest aggregation. They are kept
outside individual estimator classes so every delivered tree family uses the
same prediction contract.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from ..matrix import as_dense_matrix


def _checked_matrix(X: Any, n_features_in: int) -> NDArray[np.float32]:
    """Return dense float32 input after enforcing the fitted feature count."""

    matrix = as_dense_matrix(X)
    if matrix.shape[1] != n_features_in:
        raise ValueError("prediction feature count does not match training data")
    return matrix


def predict_native_model(model: Any, X: Any, n_features_in: int) -> NDArray[np.float32]:
    """Predict with one native regression model using the shared matrix contract."""

    matrix = _checked_matrix(X, n_features_in)
    return np.asarray(model.predict(matrix), dtype=np.float32)


def predict_forest_regression(
    estimators: Iterable[Any],
    feature_subsets: Iterable[NDArray[np.int64]],
    X: Any,
    n_features_in: int,
) -> NDArray[np.float32]:
    """Return mean regression predictions across fitted tree estimators."""

    matrix = _checked_matrix(X, n_features_in)
    estimator_tuple = tuple(estimators)
    feature_tuple = tuple(feature_subsets)
    if len(estimator_tuple) == 0 or len(estimator_tuple) != len(feature_tuple):
        raise ValueError("forest prediction requires matching fitted trees and features")
    predictions = np.zeros(matrix.shape[0], dtype=np.float64)
    for tree, features in zip(estimator_tuple, feature_tuple):
        predictions += tree.predict(matrix[:, features]).astype(np.float64, copy=False)
    predictions /= float(len(estimator_tuple))
    return predictions.astype(np.float32)


def predict_forest_proba(
    estimators: Iterable[Any],
    feature_subsets: Iterable[NDArray[np.int64]],
    X: Any,
    n_features_in: int,
) -> NDArray[np.float32]:
    """Return mean binary probabilities across fitted classifier trees."""

    matrix = _checked_matrix(X, n_features_in)
    estimator_tuple = tuple(estimators)
    feature_tuple = tuple(feature_subsets)
    if len(estimator_tuple) == 0 or len(estimator_tuple) != len(feature_tuple):
        raise ValueError("forest prediction requires matching fitted trees and features")
    probabilities = np.zeros((matrix.shape[0], 2), dtype=np.float64)
    for tree, features in zip(estimator_tuple, feature_tuple):
        probabilities += tree.predict_proba(matrix[:, features]).astype(
            np.float64, copy=False
        )
    probabilities /= float(len(estimator_tuple))
    return probabilities.astype(np.float32)
