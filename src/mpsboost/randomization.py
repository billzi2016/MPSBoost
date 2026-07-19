"""Deterministic randomization contracts for v2 tree families.

Random forests, extra trees, and CatBoost-like ordered boosting all need reproducible sampling
semantics before they can share a real MPS training backend. This module owns those public
contracts: row sampling, feature sampling, random threshold candidates, and ordered permutations.
It does not train models and does not allocate device resources.
"""

from __future__ import annotations

from math import ceil
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


def _rng(random_state: int | None) -> np.random.Generator:
    """Create the only RNG type used by public randomization helpers."""

    if random_state is not None and (
        isinstance(random_state, bool) or not isinstance(random_state, int)
    ):
        raise TypeError("random_state must be an integer or None")
    return np.random.default_rng(random_state)


def _positive_count(total: int, fraction: float, *, name: str) -> int:
    """Convert a positive fraction to a deterministic non-zero sample count."""

    if isinstance(total, bool) or not isinstance(total, int):
        raise TypeError(f"{name} total must be an integer")
    if total <= 0:
        raise ValueError(f"{name} total must be positive")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise TypeError(f"{name} fraction must be numeric")
    numeric_fraction = float(fraction)
    if not np.isfinite(numeric_fraction) or numeric_fraction <= 0.0:
        raise ValueError(f"{name} fraction must be finite and positive")
    return max(1, int(ceil(total * numeric_fraction)))


def bootstrap_sample_indices(
    n_samples: int,
    *,
    sample_fraction: float = 1.0,
    random_state: int | None = None,
) -> NDArray[np.int64]:
    """Return row indices sampled with replacement for bagging-style tree training."""

    count = _positive_count(n_samples, sample_fraction, name="sample")
    return _rng(random_state).integers(0, n_samples, size=count, dtype=np.int64)


def sample_without_replacement_indices(
    n_samples: int,
    *,
    sample_fraction: float = 1.0,
    random_state: int | None = None,
) -> NDArray[np.int64]:
    """Return unique row indices sampled without replacement in deterministic RNG order."""

    count = _positive_count(n_samples, sample_fraction, name="sample")
    if count > n_samples:
        raise ValueError("sample_fraction selects more rows than available without replacement")
    return _rng(random_state).choice(n_samples, size=count, replace=False).astype(np.int64)


def subsample_feature_indices(
    n_features: int,
    *,
    feature_fraction: float = 1.0,
    random_state: int | None = None,
) -> NDArray[np.int64]:
    """Return unique feature indices for column sampling."""

    count = _positive_count(n_features, feature_fraction, name="feature")
    if count > n_features:
        raise ValueError("feature_fraction selects more features than available")
    return _rng(random_state).choice(n_features, size=count, replace=False).astype(np.int64)


def random_threshold_candidates(
    lower_bound: float,
    upper_bound: float,
    *,
    n_candidates: int,
    random_state: int | None = None,
) -> NDArray[np.float64]:
    """Return sorted random split thresholds strictly inside a finite open interval."""

    if isinstance(n_candidates, bool) or not isinstance(n_candidates, int):
        raise TypeError("n_candidates must be an integer")
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")
    lower = float(lower_bound)
    upper = float(upper_bound)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("threshold bounds must be finite")
    if not lower < upper:
        raise ValueError("lower_bound must be smaller than upper_bound")
    values = _rng(random_state).uniform(lower, upper, size=n_candidates)
    return np.sort(values.astype(np.float64))


def ordered_boosting_permutations(
    n_samples: int,
    *,
    n_permutations: int,
    random_state: int | None = None,
) -> tuple[NDArray[np.int64], ...]:
    """Return reproducible permutations for CatBoost-like ordered boosting semantics."""

    if isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise TypeError("n_samples must be an integer")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if isinstance(n_permutations, bool) or not isinstance(n_permutations, int):
        raise TypeError("n_permutations must be an integer")
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")
    generator = _rng(random_state)
    return tuple(
        generator.permutation(n_samples).astype(np.int64)
        for _ in range(n_permutations)
    )


def validate_indices_cover_range(indices: Iterable[int], upper_bound: int) -> None:
    """Validate public index arrays before they are passed into future native backends."""

    if isinstance(upper_bound, bool) or not isinstance(upper_bound, int):
        raise TypeError("upper_bound must be an integer")
    if upper_bound <= 0:
        raise ValueError("upper_bound must be positive")
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("indices must be integers")
        if not 0 <= int(index) < upper_bound:
            raise ValueError("index out of bounds")
