"""Basic learning-to-rank estimator for MPSBoost.

This module validates query/group contracts and trains a pointwise tree model whose scoring and
validation are ranking-aware. MPS requests warn and run the in-project CPU path because this
latency-sensitive workflow is expected to be faster on CPU.
"""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
from numpy.typing import NDArray

from .base import MPSBoostRegressor


def _groups_to_offsets(group: Any, n_samples: int) -> NDArray[np.int64]:
    """Normalize group sizes to cumulative offsets after contract validation."""

    values = np.asarray(group)
    if values.ndim != 1:
        raise ValueError("group must be a one-dimensional array of query sizes")
    if values.dtype.kind not in "iu":
        raise TypeError("group values must be integers")
    if np.any(values <= 0):
        raise ValueError("group values must be positive")
    offsets = np.concatenate(([0], np.cumsum(values, dtype=np.int64)))
    if int(offsets[-1]) != n_samples:
        raise ValueError("group sizes must sum to the number of training rows")
    return offsets


def _mean_ndcg(labels: NDArray[np.float64], scores: NDArray[np.float64], offsets: NDArray[np.int64]) -> float:
    """Compute mean full-list NDCG across query groups."""

    values: list[float] = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        group_labels = labels[start:end]
        group_scores = scores[start:end]
        order = np.argsort(-group_scores, kind="mergesort")
        ideal = np.argsort(-group_labels, kind="mergesort")
        discounts = 1.0 / np.log2(np.arange(2, group_labels.size + 2, dtype=np.float64))
        gains = np.power(2.0, group_labels) - 1.0
        dcg = float(np.sum(gains[order] * discounts))
        ideal_dcg = float(np.sum(gains[ideal] * discounts))
        values.append(0.0 if ideal_dcg <= 0.0 else dcg / ideal_dcg)
    return float(np.mean(values))


class LearningToRankRegressor(MPSBoostRegressor):
    """Train a pointwise ranking scorer with explicit query-group validation."""

    _PARAMETER_NAMES = MPSBoostRegressor._PARAMETER_NAMES + ("ranking_metric",)
    _fitted_error_message = "LearningToRankRegressor is not fitted or loaded"

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        max_bins: int = 256,
        growth_strategy: str = "level_wise",
        max_leaves: int | None = None,
        max_active_leaves: int | None = None,
        min_gain_to_split: float = 0.0,
        loss: str = "squared_error",
        quantile_alpha: float = 0.5,
        tweedie_variance_power: float = 1.5,
        min_child_weight: float = 1.0,
        min_samples_leaf: int = 20,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        max_delta_step: float = 0.0,
        monotonic_constraints: Any = None,
        interaction_constraints: Any = None,
        categorical_features: Any = None,
        random_state: int | None = None,
        n_jobs: int | None = None,
        device: str = "cpu",
        verbosity: int = 1,
        ranking_metric: str = "ndcg",
    ) -> None:
        """Store ranking parameters while preserving the base estimator protocol."""

        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            max_bins=max_bins,
            growth_strategy=growth_strategy,
            max_leaves=max_leaves,
            max_active_leaves=max_active_leaves,
            min_gain_to_split=min_gain_to_split,
            loss=loss,
            quantile_alpha=quantile_alpha,
            tweedie_variance_power=tweedie_variance_power,
            min_child_weight=min_child_weight,
            min_samples_leaf=min_samples_leaf,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            max_delta_step=max_delta_step,
            monotonic_constraints=monotonic_constraints,
            interaction_constraints=interaction_constraints,
            categorical_features=categorical_features,
            random_state=random_state,
            n_jobs=n_jobs,
            device=device,
            verbosity=verbosity,
        )
        self.ranking_metric = ranking_metric

    def fit(self, X: Any, y: Any, *, group: Any, sample_weight: Any = None) -> "LearningToRankRegressor":
        """Fit a pointwise rank scorer after validating query-group boundaries."""

        requested_device = self.device
        if self.device not in {"auto", "cpu", "mps"}:
            raise ValueError("device must be one of 'auto', 'cpu', or 'mps'")
        if self.device == "mps":
            warnings.warn(
                "CPU backend selected: LearningToRankRegressor is latency-sensitive and "
                "expected to be faster on CPU than Apple GPU for this workload.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.device = "cpu"
        labels = np.asarray(y)
        offsets = _groups_to_offsets(group, labels.shape[0])
        self.group_offsets_ = offsets
        try:
            fitted = super().fit(X, y, sample_weight=sample_weight)
        finally:
            self.device = requested_device
        fitted.training_summary_.update(
            {
                "objective": "pointwise_ranking",
                "ranking_metric": self.ranking_metric,
                "query_count": int(offsets.size - 1),
                "requested_device": requested_device,
                "device": "cpu",
            }
        )
        if requested_device == "mps":
            fitted.training_summary_["fallback_reason"] = (
                "CPU backend selected: this latency-sensitive workload is expected to be faster "
                "on CPU than Apple GPU for this estimator."
            )
        return fitted

    def score(self, X: Any, y: Any, *, group: Any, sample_weight: Any = None) -> float:
        """Return mean NDCG for the supplied query groups."""

        del sample_weight
        if self.ranking_metric != "ndcg":
            raise ValueError("ranking_metric must be 'ndcg'")
        predictions = self.predict(X).astype(np.float64, copy=False)
        labels = np.asarray(y, dtype=np.float64)
        offsets = _groups_to_offsets(group, labels.shape[0])
        return _mean_ndcg(labels, predictions, offsets)
