"""Hand-calculated domain tests for squared error and training mathematics.

Tests call the sole C++ implementation directly and validate gradient/Hessian,
node scores, leaf values, and split gain with independently calculated constants.
Do not rewrite general formulas in Python as expected values.
"""

import math

import pytest

from mpsboost import _native


def test_squared_error_gradients_match_hand_computation():
    """Squared error must produce prediction-label and Hessian fixed at 1."""

    actual = _native._squared_error_gradients([1.0, -2.0, 4.0], [1.5, -3.0, 2.0])
    assert actual == [(0.5, 1.0), (-1.0, 1.0), (-2.0, 1.0)]


def test_score_weight_and_gain_match_hand_computation():
    """Freeze where lambda and gamma enter formulas to prevent backend semantic drift."""

    assert _native._node_score(-4.0, 2.0, 1.0) == pytest.approx(16.0 / 3.0)
    assert _native._leaf_weight(-4.0, 2.0, 1.0) == pytest.approx(4.0 / 3.0)
    # With left=(0,2), right=(-4,2), and lambda=0, parent score is 4 and right
    # score is 8, yielding gain 2 before gamma; explicitly subtract 0.25 here.
    assert _native._split_gain(0.0, 2.0, -4.0, 2.0, 0.0, 0.25) == 1.75


def test_binary_logistic_gradients_match_hand_computation():
    """Binary logistic uses raw margins, p-label gradients, and p*(1-p) Hessians."""

    actual = _native._binary_logistic_gradients([0.0, 1.0, 1.0], [0.0, 2.0, -2.0])
    assert actual[0] == pytest.approx((0.5, 0.25))
    assert actual[1] == pytest.approx((-0.11920292202211755, 0.10499358540350662))
    assert actual[2] == pytest.approx((-0.8807970779778824, 0.1049935854035065))


def test_advanced_regression_gradients_match_hand_computation():
    """Advanced objectives should expose one native gradient/Hessian contract."""

    assert _native._quantile_gradients([0.0, 2.0], [1.0, 1.0], 0.25) == [
        (0.75, 1.0),
        (-0.25, 1.0),
    ]
    poisson = _native._poisson_gradients([0.0, 2.0], [0.0, math.log(2.0)])
    assert poisson[0] == pytest.approx((1.0, 1.0))
    assert poisson[1] == pytest.approx((0.0, 2.0))
    tweedie = _native._tweedie_gradients([0.0, 4.0], [0.0, math.log(4.0)], 1.5)
    assert tweedie[0] == pytest.approx((1.0, 0.5))
    assert tweedie[1] == pytest.approx((0.0, 2.0))


def test_softmax_probabilities_and_gradients_match_native_contract():
    """Native softmax should normalize rows and expose diagonal class statistics."""

    probabilities = _native._softmax_probabilities([1.0, 2.0, 3.0])
    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[2] > probabilities[1] > probabilities[0]

    class_two = _native._multiclass_softmax_gradients(
        [0.0, 2.0],
        [2.0, 1.0, 0.0, 0.0, 1.0, 2.0],
        3,
        2,
    )

    assert class_two[0][0] > 0.0
    assert class_two[1][0] < 0.0
    assert class_two[0][1] > 0.0
    assert class_two[1][1] > 0.0


def test_binary_logistic_probability_is_stable_for_extreme_logits():
    """Extreme margins must not overflow or produce probabilities outside [0, 1]."""

    assert _native._logistic_probability(-1000.0) == pytest.approx(0.0)
    assert _native._logistic_probability(1000.0) == pytest.approx(1.0)
    assert _native._binary_logistic_gradients([0.0, 1.0], [-1000.0, 1000.0]) == [
        (0.0, 0.0),
        (0.0, 0.0),
    ]


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: _native._squared_error_gradients([], []), "Labels must not be empty"),
        (lambda: _native._squared_error_gradients([1.0], []), "lengths do not match"),
        (lambda: _native._squared_error_gradients([math.nan], [0.0]), "must be finite"),
        (lambda: _native._node_score(1.0, -1.0, 1.0), "Hessian"),
        (lambda: _native._leaf_weight(1.0, 0.0, 0.0), "finite and positive"),
        (lambda: _native._split_gain(0.0, 0.0, 1.0, 1.0, 1.0, 0.0), "strictly positive"),
        (lambda: _native._split_gain(1.0, 1.0, 1.0, 1.0, 0.0, -1.0), "gamma"),
        (lambda: _native._binary_logistic_gradients([], []), "Labels must not be empty"),
        (lambda: _native._binary_logistic_gradients([1.0], []), "lengths do not match"),
        (lambda: _native._binary_logistic_gradients([0.5], [0.0]), "0 or 1"),
        (lambda: _native._binary_logistic_gradients([0.0], [math.inf]), "must be finite"),
        (lambda: _native._quantile_gradients([1.0], [1.0], 0.0), "quantile"),
        (lambda: _native._poisson_gradients([-1.0], [0.0]), "non-negative"),
        (lambda: _native._tweedie_gradients([1.0], [0.0], 2.0), "tweedie"),
        (lambda: _native._logistic_probability(math.nan), "must be finite"),
    ],
)
def test_invalid_objective_inputs_fail_before_model_construction(call, message):
    """Non-finite values, invalid regularization, and Hessians must not yield trainable statistics."""

    with pytest.raises(_native.TrainingError, match=message):
        call()
