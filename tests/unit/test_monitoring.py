"""Training monitoring and early-stopping contract tests."""

import pytest

import mpsboost as mb


def test_metric_history_tracks_minimum_with_earliest_tie():
    """Minimization metrics must choose the earliest best value deterministically."""

    history = mb.MetricHistory("rmse", "minimize")
    history.append(0, 1.0)
    history.append(1, 0.8)
    history.append(2, 0.8)

    assert history.best() == mb.MetricObservation(1, "rmse", 0.8)
    assert [item.iteration for item in history.observations()] == [0, 1, 2]


def test_metric_history_tracks_maximum_with_earliest_tie():
    """Maximization metrics must choose the earliest best value deterministically."""

    history = mb.MetricHistory("auc", "maximize")
    history.append(0, 0.5)
    history.append(1, 0.7)
    history.append(2, 0.7)

    assert history.best() == mb.MetricObservation(1, "auc", 0.7)


def test_early_stopping_minimize_uses_patience_and_min_delta():
    """A minimization monitor stops only after patience is exceeded."""

    monitor = mb.EarlyStoppingMonitor(
        metric_name="logloss",
        direction="minimize",
        patience=1,
        min_delta=0.05,
    )

    assert monitor.update(0, 1.0).improved is True
    assert monitor.update(1, 0.97).improved is False
    decision = monitor.update(2, 0.96)
    assert decision.should_stop is True
    assert decision.best_iteration == 0
    assert decision.rounds_since_improvement == 2


def test_early_stopping_maximize_resets_after_improvement():
    """A maximization monitor resets patience after a real improvement."""

    monitor = mb.EarlyStoppingMonitor(
        metric_name="auc",
        direction="maximize",
        patience=1,
        min_delta=0.01,
    )

    assert monitor.update(0, 0.70).improved is True
    assert monitor.update(1, 0.705).improved is False
    decision = monitor.update(2, 0.72)
    assert decision.improved is True
    assert decision.should_stop is False
    assert decision.best_iteration == 2
    assert decision.rounds_since_improvement == 0


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: mb.MetricHistory("", "minimize"), "non-empty"),
        (lambda: mb.MetricHistory("rmse", "lower"), "direction"),
        (lambda: mb.MetricHistory("rmse", "minimize").best(), "empty"),
        (lambda: mb.MetricHistory("rmse", "minimize").append(-1, 1.0), "non-negative"),
        (lambda: mb.MetricHistory("rmse", "minimize").append(0, float("nan")), "finite"),
        (
            lambda: mb.EarlyStoppingMonitor(
                metric_name="rmse",
                direction="minimize",
                patience=-1,
            ),
            "non-negative",
        ),
        (
            lambda: mb.EarlyStoppingMonitor(
                metric_name="rmse",
                direction="minimize",
                patience=1,
                min_delta=float("inf"),
            ),
            "finite",
        ),
    ],
)
def test_invalid_monitoring_inputs_fail_early(call, message):
    """Invalid monitoring inputs must fail before estimator training starts."""

    with pytest.raises((TypeError, ValueError), match=message):
        call()


def test_metric_iterations_must_be_strictly_increasing():
    """Validation history must not accept repeated or out-of-order iteration numbers."""

    history = mb.MetricHistory("rmse", "minimize")
    history.append(1, 0.9)
    with pytest.raises(ValueError, match="strictly increasing"):
        history.append(1, 0.8)
