"""Portable backend policy and dependency diagnostics.

This module does not train through external libraries. It records the explicit S22 policy:
native CPU/MPS remains the default MPSBoost implementation, while future portable adapters must be
observable, optional, and never hidden behind a native backend name.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Literal

PortableBackend = Literal[
    "native_cpu",
    "native_mps",
    "xgboost_cpu",
    "xgboost_cuda",
    "sklearn_cpu",
]
PortableMode = Literal["native", "portable", "auto"]


@dataclass(frozen=True)
class PortableBackendDecision:
    """Serializable policy record for one portable backend selection."""

    mode: PortableMode
    backend: PortableBackend
    reason: str
    requires_extra: str | None = None
    install_command: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Return a stable summary suitable for ``training_summary_``."""

        return {
            "mode": self.mode,
            "backend": self.backend,
            "reason": self.reason,
            "requires_extra": self.requires_extra,
            "install_command": self.install_command,
        }


def optional_dependency_status() -> dict[str, dict[str, str | bool]]:
    """Report optional portable/explanation dependencies without importing them."""

    dependencies = {
        "shap": ("mpsboost[shap]", "python -m pip install 'mpsboost[shap]'"),
        "sklearn": ("mpsboost[sklearn]", "python -m pip install 'mpsboost[sklearn]'"),
        "xgboost": ("mpsboost[xgboost]", "python -m pip install 'mpsboost[xgboost]'"),
        "cuda": ("mpsboost[cuda]", "python -m pip install 'mpsboost[cuda]'"),
    }
    result: dict[str, dict[str, str | bool]] = {}
    for name, (extra, command) in dependencies.items():
        module = "xgboost" if name == "cuda" else name
        available = find_spec(module) is not None
        result[name] = {
            "available": available,
            "extra": extra,
            "install_command": command,
        }
    return result


def portable_setup_instructions(*, dependency: str | None = None) -> str:
    """Return copy-paste installation guidance for optional portable dependencies."""

    status = optional_dependency_status()
    if dependency is not None:
        if dependency not in status:
            known = ", ".join(sorted(status))
            raise ValueError(f"unknown optional dependency '{dependency}'. Known: {known}")
        item = status[dependency]
        return (
            f"Optional dependency '{dependency}' is required for this explicit portable path. "
            f"Install it with:\n  {item['install_command']}\n"
            "Set MPSBOOST_SKIP_ENV_CHECK=1 for CPU-only workers that should skip GPU diagnostics."
        )
    commands = "\n".join(
        f"  {item['install_command']}" for item in status.values()
    )
    return (
        "Optional portable backends are not installed by default. Use the matching extra:\n"
        f"{commands}\n"
        "Default MPSBoost installation remains lightweight and keeps native CPU/MPS as the primary "
        "implementation."
    )


def choose_portable_backend(
    *,
    mode: PortableMode = "native",
    requested_device: str = "auto",
    platform_system: str = "Darwin",
    cuda_available: bool = False,
    xgboost_available: bool | None = None,
    sklearn_available: bool | None = None,
) -> PortableBackendDecision:
    """Choose an observable portable backend policy without instantiating an adapter.

    ``mode="native"`` always keeps the in-project backend. ``mode="portable"`` allows external
    adapters, but only through explicit reporting. ``mode="auto"`` prefers native on Apple
    Silicon/macOS and uses portable CUDA only when the platform and dependency signal support it.
    """

    if mode not in {"native", "portable", "auto"}:
        raise ValueError("mode must be 'native', 'portable', or 'auto'")
    if requested_device not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("requested_device must be 'auto', 'cpu', 'mps', or 'cuda'")
    if not isinstance(cuda_available, bool):
        raise TypeError("cuda_available must be a bool")
    status = optional_dependency_status()
    has_xgboost = (
        bool(status["xgboost"]["available"])
        if xgboost_available is None
        else bool(xgboost_available)
    )
    has_sklearn = (
        bool(status["sklearn"]["available"])
        if sklearn_available is None
        else bool(sklearn_available)
    )

    if mode == "native" or requested_device == "mps":
        backend: PortableBackend = "native_mps" if requested_device == "mps" else "native_cpu"
        return PortableBackendDecision(
            mode=mode,
            backend=backend,
            reason="native CPU/MPS remains the default MPSBoost backend",
        )
    if platform_system == "Darwin" and mode == "auto":
        return PortableBackendDecision(
            mode=mode,
            backend="native_cpu",
            reason="Apple platforms prioritize native CPU/MPS; device auto decides inside native training",
        )
    if requested_device == "cuda" or (
        mode == "auto" and platform_system == "Linux" and cuda_available
    ):
        if has_xgboost:
            return PortableBackendDecision(
                mode=mode,
                backend="xgboost_cuda",
                reason="explicit portable CUDA path with optional XGBoost dependency available",
                requires_extra="mpsboost[cuda]",
            )
        return PortableBackendDecision(
            mode=mode,
            backend="native_cpu",
            reason="CUDA portable path requested but XGBoost is unavailable; native CPU remains usable",
            requires_extra="mpsboost[cuda]",
            install_command=str(status["cuda"]["install_command"]),
        )
    if has_sklearn:
        return PortableBackendDecision(
            mode=mode,
            backend="sklearn_cpu",
            reason="portable CPU path can use the optional sklearn adapter",
            requires_extra="mpsboost[sklearn]",
        )
    if has_xgboost:
        return PortableBackendDecision(
            mode=mode,
            backend="xgboost_cpu",
            reason="portable CPU path can use the optional XGBoost adapter",
            requires_extra="mpsboost[xgboost]",
        )
    return PortableBackendDecision(
        mode=mode,
        backend="native_cpu",
        reason="no optional portable dependency is available; native CPU remains usable",
        install_command=str(status["sklearn"]["install_command"]),
    )


class PortableEstimatorAdapter:
    """sklearn-style adapter that records portable policy while delegating to a real estimator.

    The first supported implementation delegates to an in-project MPSBoost estimator. External
    adapters can be added behind this boundary later, but they must remain explicit and observable.
    """

    _PARAMETER_NAMES = (
        "estimator",
        "mode",
        "requested_device",
        "platform_system",
        "cuda_available",
    )

    def __init__(
        self,
        estimator: Any,
        *,
        mode: PortableMode = "native",
        requested_device: str = "auto",
        platform_system: str = "Darwin",
        cuda_available: bool = False,
    ) -> None:
        """Store adapter settings without importing optional third-party backends."""

        self.estimator = estimator
        self.mode = mode
        self.requested_device = requested_device
        self.platform_system = platform_system
        self.cuda_available = cuda_available

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return adapter and nested estimator parameters for sklearn model selection."""

        parameters = {name: getattr(self, name) for name in self._PARAMETER_NAMES}
        if deep and hasattr(self.estimator, "get_params"):
            for name, value in self.estimator.get_params(deep=True).items():
                parameters[f"estimator__{name}"] = value
        return parameters

    def set_params(self, **parameters: Any) -> "PortableEstimatorAdapter":
        """Set adapter or nested estimator parameters using sklearn double-underscore names."""

        nested: dict[str, Any] = {}
        direct: dict[str, Any] = {}
        for name, value in parameters.items():
            if name.startswith("estimator__"):
                nested[name.removeprefix("estimator__")] = value
            else:
                direct[name] = value
        unknown = sorted(set(direct) - set(self._PARAMETER_NAMES))
        if unknown:
            raise ValueError(f"Unknown parameter(s): {', '.join(unknown)}")
        for name, value in direct.items():
            setattr(self, name, value)
        if nested:
            if not hasattr(self.estimator, "set_params"):
                raise ValueError("nested estimator does not support set_params")
            self.estimator.set_params(**nested)
        return self

    def fit(self, X: Any, y: Any, **fit_params: Any) -> "PortableEstimatorAdapter":
        """Fit through the selected policy and expose the actual backend in a stable summary."""

        decision = choose_portable_backend(
            mode=self.mode,
            requested_device=self.requested_device,
            platform_system=self.platform_system,
            cuda_available=self.cuda_available,
        )
        if decision.backend not in {"native_cpu", "native_mps"}:
            raise RuntimeError(
                "External portable estimator adapters require validated backend mapping before "
                f"use. Selected policy: {decision.to_dict()}"
            )
        if hasattr(self.estimator, "set_params"):
            native_device = "mps" if decision.backend == "native_mps" else "cpu"
            self.estimator.set_params(device=native_device)
        self.estimator.fit(X, y, **fit_params)
        self.estimator_ = self.estimator
        self.portable_backend_decision_ = decision.to_dict()
        self.training_summary_ = {
            "portable_backend": self.portable_backend_decision_,
            "estimator_summary": getattr(self.estimator, "training_summary_", {}),
        }
        for name in ("classes_", "n_features_in_", "n_estimators_"):
            if hasattr(self.estimator, name):
                setattr(self, name, getattr(self.estimator, name))
        return self

    def predict(self, X: Any) -> Any:
        """Delegate prediction to the fitted estimator."""

        return self.estimator_.predict(X)

    def predict_proba(self, X: Any) -> Any:
        """Delegate classifier probabilities when the fitted estimator supports them."""

        if not hasattr(self.estimator_, "predict_proba"):
            raise AttributeError("wrapped estimator does not support predict_proba")
        return self.estimator_.predict_proba(X)

    def score(self, X: Any, y: Any, **score_params: Any) -> float:
        """Delegate scoring through the wrapped estimator's own metric semantics."""

        return float(self.estimator_.score(X, y, **score_params))
