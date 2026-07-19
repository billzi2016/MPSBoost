"""Device-selection policy for CPU, MPS, and automatic backend choice.

The policy keeps backend choice outside estimator training logic. It is deliberately simple and
deterministic for now: explicit devices are honored, unavailable MPS falls back to CPU under
``auto``, and small or synchronization-heavy workloads stay on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

RequestedDevice = Literal["cpu", "mps", "auto"]
ResolvedDevice = Literal["cpu", "mps"]


@dataclass(frozen=True)
class DeviceDecision:
    """Immutable record explaining one backend-selection decision."""

    requested: RequestedDevice
    selected: ResolvedDevice
    reason: str
    estimated_work: int
    mps_available: bool


def _estimated_histogram_work(
    *,
    n_samples: int,
    n_features: int,
    n_estimators: int,
    max_bins: int,
) -> int:
    """Estimate tree histogram work with overflow-safe Python integers."""

    for name, value in (
        ("n_samples", n_samples),
        ("n_features", n_features),
        ("n_estimators", n_estimators),
        ("max_bins", max_bins),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    return n_samples * n_features * n_estimators * max_bins


def choose_device(
    *,
    requested: RequestedDevice,
    n_samples: int,
    n_features: int,
    n_estimators: int,
    max_bins: int,
    mps_available: bool,
    mps_work_threshold: int = 134_217_728,
) -> DeviceDecision:
    """Choose the backend for one fit call and explain the decision."""

    if requested not in {"cpu", "mps", "auto"}:
        raise ValueError("requested device must be 'cpu', 'mps', or 'auto'")
    if not isinstance(mps_available, bool):
        raise TypeError("mps_available must be a bool")
    if isinstance(mps_work_threshold, bool) or not isinstance(mps_work_threshold, int):
        raise TypeError("mps_work_threshold must be an integer")
    if mps_work_threshold <= 0:
        raise ValueError("mps_work_threshold must be positive")

    estimated_work = _estimated_histogram_work(
        n_samples=n_samples,
        n_features=n_features,
        n_estimators=n_estimators,
        max_bins=max_bins,
    )
    if requested == "cpu":
        return DeviceDecision("cpu", "cpu", "explicit cpu request", estimated_work, mps_available)
    if requested == "mps":
        return DeviceDecision("mps", "mps", "explicit mps request", estimated_work, mps_available)
    if not mps_available:
        return DeviceDecision("auto", "cpu", "mps unavailable", estimated_work, mps_available)
    if estimated_work < mps_work_threshold:
        return DeviceDecision(
            "auto",
            "cpu",
            "estimated work below mps threshold",
            estimated_work,
            mps_available,
        )
    return DeviceDecision(
        "auto",
        "mps",
        "estimated work meets mps threshold",
        estimated_work,
        mps_available,
    )


def decision_to_dict(decision: DeviceDecision) -> dict[str, object]:
    """Convert a decision to a stable serializable diagnostic mapping."""

    return {
        "requested": decision.requested,
        "selected": decision.selected,
        "reason": decision.reason,
        "estimated_work": int(decision.estimated_work),
        "mps_available": bool(decision.mps_available),
    }
