"""Device-selection policy tests."""

import pytest

import mpsboost as mb


def test_explicit_devices_are_honored():
    """Explicit CPU and MPS requests must not be rewritten by the auto policy."""

    cpu = mb.choose_device(
        requested="cpu",
        n_samples=10,
        n_features=2,
        n_estimators=1,
        max_bins=16,
        mps_available=True,
    )
    mps = mb.choose_device(
        requested="mps",
        n_samples=10,
        n_features=2,
        n_estimators=1,
        max_bins=16,
        mps_available=False,
    )

    assert cpu.selected == "cpu"
    assert cpu.reason == "explicit cpu request"
    assert mps.selected == "mps"
    assert mps.reason == "explicit mps request"


def test_auto_uses_cpu_when_mps_is_unavailable_or_work_is_small():
    """Auto should choose CPU for unavailable MPS or small estimated work."""

    unavailable = mb.choose_device(
        requested="auto",
        n_samples=10_000,
        n_features=100,
        n_estimators=100,
        max_bins=256,
        mps_available=False,
    )
    small = mb.choose_device(
        requested="auto",
        n_samples=100,
        n_features=8,
        n_estimators=20,
        max_bins=64,
        mps_available=True,
    )

    assert unavailable.selected == "cpu"
    assert unavailable.reason == "mps unavailable"
    assert small.selected == "cpu"
    assert small.reason == "estimated work below mps threshold"


def test_auto_uses_mps_for_large_available_workloads():
    """Auto should choose MPS only when availability and estimated work both justify it."""

    decision = mb.choose_device(
        requested="auto",
        n_samples=50_000,
        n_features=100,
        n_estimators=200,
        max_bins=256,
        mps_available=True,
    )

    assert decision.selected == "mps"
    assert decision.reason == "estimated work meets mps threshold"
    assert decision.estimated_work == 256_000_000_000


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"requested": "gpu"}, "requested device"),
        ({"n_samples": 0}, "positive"),
        ({"n_features": 0}, "positive"),
        ({"n_estimators": 0}, "positive"),
        ({"max_bins": 0}, "positive"),
        ({"mps_available": 1}, "bool"),
        ({"mps_work_threshold": 0}, "positive"),
    ],
)
def test_invalid_device_policy_inputs_fail_early(kwargs, message):
    """Invalid policy inputs must fail before estimator training starts."""

    base = {
        "requested": "auto",
        "n_samples": 10,
        "n_features": 2,
        "n_estimators": 1,
        "max_bins": 16,
        "mps_available": True,
    }
    base.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        mb.choose_device(**base)
