"""Training monitoring and early-stopping contracts for tree estimators.

This module owns model-family-independent monitoring semantics: metric direction, validation
history, best-iteration tracking, and patience-based early stopping. Estimators can reuse these
objects without duplicating callback logic or embedding monitoring policy inside device backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

MetricDirection = Literal["minimize", "maximize"]


@dataclass(frozen=True)
class MetricObservation:
    """One immutable validation metric observation."""

    iteration: int
    name: str
    value: float


@dataclass(frozen=True)
class EarlyStoppingDecision:
    """Result of processing one metric observation."""

    should_stop: bool
    improved: bool
    best_iteration: int
    best_score: float
    rounds_since_improvement: int


def _validate_iteration(iteration: int) -> None:
    """Validate non-negative boosting iteration indexes."""

    if isinstance(iteration, bool) or not isinstance(iteration, int):
        raise TypeError("iteration must be an integer")
    if iteration < 0:
        raise ValueError("iteration must be non-negative")


def _validate_metric_value(value: float) -> float:
    """Validate and normalize one finite metric value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("metric value must be numeric")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError("metric value must be finite")
    return numeric


class MetricHistory:
    """Append-only metric history with deterministic best-observation lookup."""

    def __init__(self, name: str, direction: MetricDirection) -> None:
        """Create an empty metric history for one named metric."""

        if not name:
            raise ValueError("metric name must be non-empty")
        if direction not in {"minimize", "maximize"}:
            raise ValueError("metric direction must be 'minimize' or 'maximize'")
        self.name = name
        self.direction = direction
        self._observations: list[MetricObservation] = []

    def append(self, iteration: int, value: float) -> MetricObservation:
        """Append one observation and return its immutable record."""

        _validate_iteration(iteration)
        numeric = _validate_metric_value(value)
        if self._observations and iteration <= self._observations[-1].iteration:
            raise ValueError("metric iterations must be strictly increasing")
        observation = MetricObservation(iteration, self.name, numeric)
        self._observations.append(observation)
        return observation

    def observations(self) -> tuple[MetricObservation, ...]:
        """Return all observations in insertion order."""

        return tuple(self._observations)

    def best(self) -> MetricObservation:
        """Return the best observation, preferring the earliest iteration on ties."""

        if not self._observations:
            raise ValueError("metric history is empty")
        if self.direction == "minimize":
            return min(self._observations, key=lambda item: (item.value, item.iteration))
        return max(self._observations, key=lambda item: (item.value, -item.iteration))


class EarlyStoppingMonitor:
    """Patience-based early stopping state machine for one validation metric."""

    def __init__(
        self,
        *,
        metric_name: str,
        direction: MetricDirection,
        patience: int,
        min_delta: float = 0.0,
    ) -> None:
        """Create a monitor without observing data or touching estimator state."""

        if isinstance(patience, bool) or not isinstance(patience, int):
            raise TypeError("patience must be an integer")
        if patience < 0:
            raise ValueError("patience must be non-negative")
        if isinstance(min_delta, bool) or not isinstance(min_delta, (int, float)):
            raise TypeError("min_delta must be numeric")
        numeric_delta = float(min_delta)
        if not np.isfinite(numeric_delta) or numeric_delta < 0.0:
            raise ValueError("min_delta must be finite and non-negative")
        self.history = MetricHistory(metric_name, direction)
        self.patience = patience
        self.min_delta = numeric_delta
        self._best: MetricObservation | None = None
        self._rounds_since_improvement = 0

    def update(self, iteration: int, value: float) -> EarlyStoppingDecision:
        """Process one metric value and return a deterministic stop decision."""

        observation = self.history.append(iteration, value)
        improved = self._is_improvement(observation)
        if improved:
            self._best = observation
            self._rounds_since_improvement = 0
        else:
            self._rounds_since_improvement += 1
        best = self._best
        if best is None:
            raise RuntimeError("early stopping monitor failed to initialize best observation")
        should_stop = self._rounds_since_improvement > self.patience
        return EarlyStoppingDecision(
            should_stop=should_stop,
            improved=improved,
            best_iteration=best.iteration,
            best_score=best.value,
            rounds_since_improvement=self._rounds_since_improvement,
        )

    def _is_improvement(self, observation: MetricObservation) -> bool:
        """Return whether an observation improves over the current best by min_delta."""

        if self._best is None:
            return True
        if self.history.direction == "minimize":
            return observation.value < self._best.value - self.min_delta
        return observation.value > self._best.value + self.min_delta
