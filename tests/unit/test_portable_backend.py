"""Portable backend policy and optional dependency diagnostics tests."""

import mpsboost as mb
import numpy as np
import pytest


def test_optional_dependency_status_reports_copy_paste_commands():
    """Optional extras should be discoverable without importing heavy dependencies."""

    status = mb.optional_dependency_status()

    assert {"shap", "sklearn", "xgboost", "cuda"} <= set(status)
    assert status["cuda"]["install_command"] == "python -m pip install 'mpsboost[cuda]'"
    assert isinstance(status["xgboost"]["available"], bool)


def test_portable_setup_instructions_are_non_interactive():
    """Portable guidance should be copy-pasteable and must not prompt workers."""

    text = mb.portable_setup_instructions(dependency="xgboost")

    assert "python -m pip install 'mpsboost[xgboost]'" in text
    assert "MPSBOOST_SKIP_ENV_CHECK=1" in text
    assert "input(" not in text


def test_native_policy_remains_default():
    """S22 must not replace the native CPU/MPS backend by default."""

    decision = mb.choose_portable_backend(
        mode="native",
        requested_device="auto",
        platform_system="Linux",
        cuda_available=True,
        xgboost_available=True,
        sklearn_available=True,
    )

    assert decision.backend == "native_cpu"
    assert decision.to_dict()["backend"] == "native_cpu"
    assert "native CPU/MPS" in decision.reason


def test_cuda_portable_policy_reports_missing_extra():
    """Explicit CUDA requests should stay runnable with native CPU guidance when XGBoost is absent."""

    decision = mb.choose_portable_backend(
        mode="portable",
        requested_device="cuda",
        platform_system="Linux",
        cuda_available=True,
        xgboost_available=False,
        sklearn_available=False,
    )

    assert decision.backend == "native_cpu"
    assert decision.requires_extra == "mpsboost[cuda]"
    assert decision.install_command == "python -m pip install 'mpsboost[cuda]'"


def test_portable_policy_can_select_observable_external_cpu_adapter():
    """External CPU adapters must be explicit and visible in summaries."""

    decision = mb.choose_portable_backend(
        mode="portable",
        requested_device="cpu",
        platform_system="Linux",
        cuda_available=False,
        xgboost_available=False,
        sklearn_available=True,
    )

    assert decision.backend == "sklearn_cpu"
    assert decision.requires_extra == "mpsboost[sklearn]"
    assert decision.to_dict()["mode"] == "portable"


def test_portable_estimator_adapter_preserves_sklearn_protocol_on_native_cpu():
    """The unified adapter should delegate fit/predict/score and expose backend summary."""

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    adapter = mb.PortableEstimatorAdapter(
        mb.GradientBoostingRegressor(
            n_estimators=1,
            max_depth=1,
            min_samples_leaf=1,
            min_child_weight=0.0,
        ),
        mode="native",
        requested_device="auto",
    )

    adapter.set_params(estimator__learning_rate=0.2)
    adapter.fit(X, y)

    assert adapter.get_params()["estimator__learning_rate"] == 0.2
    assert adapter.predict(X).shape == y.shape
    assert isinstance(adapter.score(X, y), float)
    assert adapter.training_summary_["portable_backend"]["backend"] == "native_cpu"


def test_portable_estimator_adapter_keeps_external_policy_executable():
    """External paths should warn, record requested backend, and keep native CPU runnable."""

    X = np.array([[0.0], [1.0]], dtype=np.float32)
    y = np.array([0.0, 1.0], dtype=np.float32)
    adapter = mb.PortableEstimatorAdapter(
        mb.GradientBoostingRegressor(),
        mode="portable",
        requested_device="cpu",
        platform_system="Linux",
    )

    with pytest.warns(RuntimeWarning, match="native CPU compatibility path"):
        adapter.fit(X, y)

    assert adapter.predict(X).shape == y.shape
    assert adapter.training_summary_["portable_backend"]["backend"] == "native_cpu"
    assert adapter.training_summary_["portable_backend_requested"]["backend"] in {
        "sklearn_cpu",
        "xgboost_cpu",
    }
