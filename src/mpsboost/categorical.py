"""Categorical feature encoding for the shared native tree engine.

The native trainer only consumes dense floating-point matrices. This module is
the single adapter that turns user-marked categorical columns into deterministic
ordered category codes before native quantization. Missing and unknown category
values become NaN, so they reuse the native missing-value default-direction
semantics instead of introducing a second prediction path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable

import numpy as np
from numpy.typing import NDArray

from .matrix import as_dense_matrix


@dataclass(frozen=True)
class CategoricalFeatureMapping:
    """Frozen mapping for one training-time categorical feature."""

    feature_index: int
    categories: tuple[Hashable, ...]
    codes: dict[Hashable, float]


@dataclass(frozen=True)
class CategoricalMetadata:
    """All categorical mappings needed to encode future prediction matrices."""

    n_features: int
    features: tuple[int, ...]
    mappings: tuple[CategoricalFeatureMapping, ...]


def has_categorical_features(categorical_features: Any) -> bool:
    """Return whether the user requested categorical handling."""

    if categorical_features is None:
        return False
    if isinstance(categorical_features, (str, bytes)):
        raise TypeError("categorical_features must contain integer feature indices")
    return len(tuple(categorical_features)) > 0


def fit_transform_categorical(
    X: Any,
    categorical_features: Any,
    labels: NDArray[np.float64],
    sample_weight: NDArray[np.float64],
) -> tuple[NDArray[np.float32] | NDArray[np.float64], CategoricalMetadata | None]:
    """Encode marked categorical columns and return frozen metadata.

    Categories are ordered by weighted target mean, then by stable representation.
    A native threshold split over these ordered codes is therefore a real ordered
    categorical split, while all gain computation stays in the existing tree
    engine.
    """

    if not has_categorical_features(categorical_features):
        return as_dense_matrix(X), None
    array = _as_two_dimensional_array(X)
    if labels.shape[0] != array.shape[0] or sample_weight.shape[0] != array.shape[0]:
        raise ValueError("categorical encoding requires X, y, and sample_weight row alignment")
    features = _normalize_categorical_features(categorical_features, array.shape[1])
    encoded = _numeric_copy_with_nan(array, features)
    mappings: list[CategoricalFeatureMapping] = []
    for feature in features:
        mapping = _fit_one_feature(array[:, feature], labels, sample_weight, feature)
        _encode_one_feature(array[:, feature], encoded[:, feature], mapping, allow_unknown=False)
        mappings.append(mapping)
    return np.ascontiguousarray(encoded, dtype=np.float32), CategoricalMetadata(
        n_features=array.shape[1],
        features=features,
        mappings=tuple(mappings),
    )


def transform_categorical(
    X: Any,
    metadata: CategoricalMetadata | None,
) -> NDArray[np.float32] | NDArray[np.float64]:
    """Apply frozen categorical metadata, sending unknown categories to NaN."""

    if metadata is None:
        return as_dense_matrix(X)
    array = _as_two_dimensional_array(X)
    if array.shape[1] != metadata.n_features:
        raise ValueError("prediction feature count does not match training data")
    encoded = _numeric_copy_with_nan(array, metadata.features)
    for mapping in metadata.mappings:
        _encode_one_feature(
            array[:, mapping.feature_index],
            encoded[:, mapping.feature_index],
            mapping,
            allow_unknown=True,
        )
    return np.ascontiguousarray(encoded, dtype=np.float32)


def _as_two_dimensional_array(value: Any) -> NDArray[Any]:
    """Return a dense 2D array while preserving object category values."""

    if hasattr(value, "toarray") and not isinstance(value, np.ndarray):
        raise TypeError("sparse matrices are not supported")
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError("X must be a two-dimensional dense array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("X must contain at least one row and one feature")
    return array


def _normalize_categorical_features(features: Any, n_features: int) -> tuple[int, ...]:
    """Validate integer feature indices and return them sorted without duplicates."""

    raw = tuple(features)
    if len(raw) == n_features and all(isinstance(item, (bool, np.bool_)) for item in raw):
        indices = tuple(index for index, flag in enumerate(raw) if bool(flag))
    else:
        indices = tuple(_normalize_feature_index(item, n_features) for item in raw)
    if len(set(indices)) != len(indices):
        raise ValueError("categorical_features must not contain duplicates")
    return tuple(sorted(indices))


def _normalize_feature_index(value: Any, n_features: int) -> int:
    """Normalize one non-negative or negative Python-style feature index."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("categorical_features must contain integer feature indices")
    index = int(value)
    if index < 0:
        index += n_features
    if not 0 <= index < n_features:
        raise ValueError("categorical feature index is out of range")
    return index


def _numeric_copy_with_nan(array: NDArray[Any], categorical_features: tuple[int, ...]) -> NDArray[np.float32]:
    """Convert non-categorical columns to floats and leave categorical columns for encoding."""

    encoded = np.empty(array.shape, dtype=np.float32)
    categorical = set(categorical_features)
    for feature in range(array.shape[1]):
        if feature in categorical:
            encoded[:, feature] = np.nan
            continue
        try:
            encoded[:, feature] = np.asarray(array[:, feature], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise TypeError("non-categorical features must be numeric") from exc
    return encoded


def _fit_one_feature(
    values: NDArray[Any],
    labels: NDArray[np.float64],
    sample_weight: NDArray[np.float64],
    feature_index: int,
) -> CategoricalFeatureMapping:
    """Build a target-ordered code mapping for one categorical column."""

    totals: dict[Hashable, float] = {}
    weights: dict[Hashable, float] = {}
    for value, label, weight in zip(values, labels, sample_weight, strict=True):
        if _is_missing(value):
            continue
        key = _category_key(value)
        totals[key] = totals.get(key, 0.0) + float(label) * float(weight)
        weights[key] = weights.get(key, 0.0) + float(weight)
    if not totals:
        raise ValueError("categorical feature must contain at least one non-missing category")
    ordered = tuple(
        key
        for key, _ in sorted(
            ((key, totals[key] / weights[key]) for key in totals),
            key=lambda item: (item[1], repr(item[0])),
        )
    )
    return CategoricalFeatureMapping(
        feature_index=feature_index,
        categories=ordered,
        codes={key: float(index) for index, key in enumerate(ordered)},
    )


def _encode_one_feature(
    values: NDArray[Any],
    output: NDArray[np.float32],
    mapping: CategoricalFeatureMapping,
    *,
    allow_unknown: bool,
) -> None:
    """Encode known categories and route missing or allowed unknown values to NaN."""

    for row, value in enumerate(values):
        if _is_missing(value):
            output[row] = np.nan
            continue
        key = _category_key(value)
        if key in mapping.codes:
            output[row] = mapping.codes[key]
        elif allow_unknown:
            output[row] = np.nan
        else:
            raise ValueError("training categories changed while fitting")


def _is_missing(value: Any) -> bool:
    """Return whether a categorical scalar should use native missing semantics."""

    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def _category_key(value: Any) -> Hashable:
    """Convert NumPy scalar categories into stable hashable Python keys."""

    if isinstance(value, np.generic):
        value = value.item()
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError("categorical values must be hashable scalars") from exc
    return value
