"""Deterministic randomization contract tests for v2 tree families."""

import numpy as np
import pytest

import mpsboost as mb


def test_bootstrap_sampling_is_reproducible_and_allows_duplicates():
    """Bootstrap sampling must be deterministic for a seed and sample with replacement."""

    first = mb.bootstrap_sample_indices(4, sample_fraction=3.0, random_state=7)
    second = mb.bootstrap_sample_indices(4, sample_fraction=3.0, random_state=7)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (12,)
    assert len(set(first.tolist())) < len(first)
    mb.validate_indices_cover_range(first, 4)


def test_without_replacement_sampling_and_feature_subsampling_are_unique():
    """Row and feature subsampling without replacement must never emit duplicates."""

    rows = mb.sample_without_replacement_indices(10, sample_fraction=0.4, random_state=11)
    features = mb.subsample_feature_indices(8, feature_fraction=0.5, random_state=11)

    assert rows.shape == (4,)
    assert features.shape == (4,)
    assert len(set(rows.tolist())) == 4
    assert len(set(features.tolist())) == 4
    mb.validate_indices_cover_range(rows, 10)
    mb.validate_indices_cover_range(features, 8)


def test_random_threshold_candidates_are_sorted_inside_bounds():
    """ExtraTrees-style random thresholds must be finite, sorted, and inside the interval."""

    thresholds = mb.random_threshold_candidates(
        1.5,
        4.5,
        n_candidates=16,
        random_state=19,
    )

    assert thresholds.dtype == np.float64
    assert np.all(thresholds > 1.5)
    assert np.all(thresholds < 4.5)
    assert np.all(thresholds[:-1] <= thresholds[1:])
    np.testing.assert_array_equal(
        thresholds,
        mb.random_threshold_candidates(1.5, 4.5, n_candidates=16, random_state=19),
    )


def test_ordered_boosting_permutations_are_reproducible_full_permutations():
    """CatBoost-like ordered boosting must get full permutations, not sampled subsets."""

    permutations = mb.ordered_boosting_permutations(
        6,
        n_permutations=3,
        random_state=23,
    )
    repeated = mb.ordered_boosting_permutations(6, n_permutations=3, random_state=23)

    assert len(permutations) == 3
    for left, right in zip(permutations, repeated, strict=True):
        np.testing.assert_array_equal(left, right)
        assert sorted(left.tolist()) == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: mb.bootstrap_sample_indices(0), "positive"),
        (lambda: mb.bootstrap_sample_indices(4, sample_fraction=0.0), "positive"),
        (
            lambda: mb.sample_without_replacement_indices(4, sample_fraction=2.0),
            "more rows",
        ),
        (
            lambda: mb.subsample_feature_indices(4, feature_fraction=2.0),
            "more features",
        ),
        (
            lambda: mb.random_threshold_candidates(1.0, 1.0, n_candidates=1),
            "smaller",
        ),
        (
            lambda: mb.random_threshold_candidates(0.0, 1.0, n_candidates=0),
            "positive",
        ),
        (
            lambda: mb.ordered_boosting_permutations(3, n_permutations=0),
            "positive",
        ),
        (lambda: mb.validate_indices_cover_range([4], 4), "out of bounds"),
    ],
)
def test_invalid_randomization_inputs_fail_early(call, message):
    """Invalid randomization inputs must fail before reaching a native backend."""

    with pytest.raises((TypeError, ValueError), match=message):
        call()
