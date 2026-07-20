"""Diagnostics and user-facing setup guidance tests."""

import warnings

import mpsboost.diagnostics as diagnostics


def test_mps_setup_instructions_include_fix_and_skip_commands():
    """Environment guidance should be copy-pasteable and non-interactive."""

    text = diagnostics.mps_setup_instructions()

    assert "xcode-select --install" in text
    assert "xcodebuild -downloadComponent MetalToolchain" in text
    assert "MPSBOOST_SKIP_ENV_CHECK=1 python your_script.py" in text
    assert "input(" not in text


def test_import_time_environment_check_can_be_skipped(monkeypatch):
    """CPU-only workers should be able to suppress import-time GPU diagnostics."""

    monkeypatch.setenv("MPSBOOST_SKIP_ENV_CHECK", "1")
    monkeypatch.setattr(diagnostics, "is_available", lambda: False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        diagnostics.warn_if_mps_unavailable()

    assert caught == []


def test_import_time_environment_check_warns_without_blocking(monkeypatch):
    """Unavailable MPS should produce setup guidance without prompting or raising."""

    monkeypatch.delenv("MPSBOOST_SKIP_ENV_CHECK", raising=False)
    monkeypatch.setattr(diagnostics, "is_available", lambda: False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        diagnostics.warn_if_mps_unavailable()

    assert len(caught) == 1
    assert "MPSBOOST_SKIP_ENV_CHECK=1" in str(caught[0].message)
    assert "xcodebuild -downloadComponent MetalToolchain" in str(caught[0].message)
