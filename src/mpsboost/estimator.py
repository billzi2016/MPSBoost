"""sklearn-style regression estimator for MPSBoost.

This module owns parameter storage, input adaptation, concurrent-fit protection, fitted-state
replacement, and sklearn protocol methods. Quantization, boosting, Metal scheduling, and model
format logic stay in their single native implementations instead of being duplicated in Python.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import _native
from .device_policy import choose_device, decision_to_dict
from .diagnostics import _metallib_path, is_available
from .matrix import as_dense_matrix, as_labels


class NotFittedError(RuntimeError):
    """Raised when fitted-only functionality is called before a complete model exists."""


class MPSBoostRegressor:
    """Train a squared-error histogram GBDT regressor on the CPU oracle or MPS backend.

    The constructor only stores parameters. It does not initialize devices, create caches, or
    allocate training memory. ``device="cpu"`` selects the explicit oracle/diagnostic backend;
    ``device="mps"`` fails clearly when MPS is unavailable and never silently falls back.
    """

    _estimator_type = "regressor"
    _PARAMETER_NAMES = (
        "n_estimators",
        "learning_rate",
        "max_depth",
        "max_bins",
        "min_child_weight",
        "min_samples_leaf",
        "reg_lambda",
        "random_state",
        "device",
        "verbosity",
    )

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        max_bins: int = 256,
        min_child_weight: float = 1.0,
        min_samples_leaf: int = 20,
        reg_lambda: float = 1.0,
        random_state: int | None = None,
        device: str = "mps",
        verbosity: int = 1,
    ) -> None:
        """Store estimator parameters without expensive side effects."""

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_bins = max_bins
        self.min_child_weight = min_child_weight
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.device = device
        self.verbosity = verbosity
        self._fit_lock = Lock()

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return all constructor parameters for sklearn-style model selection.

        ``deep`` is accepted for protocol compatibility. This estimator currently has no nested
        estimators, so the returned mapping is identical for both values.
        """

        del deep
        return {name: getattr(self, name) for name in self._PARAMETER_NAMES}

    def set_params(self, **parameters: Any) -> "MPSBoostRegressor":
        """Set known constructor parameters and return ``self``.

        Unknown parameters fail early. Any real parameter change clears fitted state so
        ``get_params`` cannot describe one model while prediction uses another.
        """

        unknown = sorted(set(parameters) - set(self._PARAMETER_NAMES))
        if unknown:
            raise ValueError(f"未知参数：{', '.join(unknown)}")
        for name, value in parameters.items():
            setattr(self, name, value)
        if parameters:
            self._clear_fitted_state()
        return self

    def fit(self, X: Any, y: Any) -> "MPSBoostRegressor":
        """Train a complete model and replace fitted state only after success.

        The same estimator instance does not support concurrent ``fit`` calls. Failures clear any
        partial state so callers never observe a partially trained ensemble.
        """

        if not self._fit_lock.acquire(blocking=False):
            raise RuntimeError("同一 estimator 不支持并发 fit")
        try:
            self._validate_parameters()
            matrix = as_dense_matrix(X)
            labels = as_labels(y, matrix.shape[0])
            parameters = _native._TrainingParameters(
                self.n_estimators,
                self.learning_rate,
                self.max_bins,
                self.max_depth,
                self.min_samples_leaf,
                self.min_child_weight,
                self.reg_lambda,
            )
            mps_available = is_available()
            device_decision = choose_device(
                requested=self.device,
                n_samples=matrix.shape[0],
                n_features=matrix.shape[1],
                n_estimators=self.n_estimators,
                max_bins=self.max_bins,
                mps_available=mps_available,
            )
            started = perf_counter()
            if device_decision.selected == "mps":
                if not mps_available:
                    raise _native.BackendError(
                        "MPS 后端不可用；请在受支持的 Apple Silicon Mac 上运行"
                    )
                with _metallib_path() as metallib_path:
                    candidate = _native._train_regressor_mps(
                        matrix, labels, parameters, metallib_path
                    )
            else:
                candidate = _native._train_regressor_cpu(matrix, labels, parameters)
            elapsed = perf_counter() - started

            # Commit fitted fields only at the end of the success path. Device errors, OOM, or
            # input errors must not leave a half-trained model behind.
            self.model_ = candidate
            self.n_features_in_ = matrix.shape[1]
            self.device_ = device_decision.selected
            self.device_decision_ = decision_to_dict(device_decision)
            self.n_estimators_ = candidate.tree_count
            self.training_summary_ = {
                "fit_seconds": elapsed,
                "input_contiguous": bool(matrix.flags.c_contiguous),
                "device": device_decision.selected,
                "device_decision": self.device_decision_,
                "n_estimators": candidate.tree_count,
            }
            return self
        except Exception:
            self._clear_fitted_state()
            raise
        finally:
            self._fit_lock.release()

    def predict(self, X: Any) -> NDArray[np.float32]:
        """Return one-dimensional float32 predictions using the frozen training schema."""

        model = self._require_model()
        matrix = as_dense_matrix(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("预测特征数量与训练数据不一致")
        return np.asarray(model.predict(matrix), dtype=np.float32)

    def score(self, X: Any, y: Any) -> float:
        """Return the default regression R² score for sklearn model-selection tools."""

        predictions = self.predict(X).astype(np.float64, copy=False)
        labels = as_labels(y, predictions.shape[0]).astype(np.float64, copy=False)
        residual_sum = float(np.sum((labels - predictions) ** 2))
        centered = labels - float(np.mean(labels))
        total_sum = float(np.sum(centered**2))
        if total_sum == 0.0:
            return 1.0 if residual_sum == 0.0 else 0.0
        score = 1.0 - residual_sum / total_sum
        if not np.isfinite(score):
            raise ValueError("R2 score is not finite")
        return float(score)

    @property
    def feature_importances_(self) -> NDArray[np.float32]:
        """Return normalized gain-based feature importance for sklearn-style tooling."""

        return self.feature_importance(kind="gain")

    def feature_importance(self, kind: str = "gain") -> NDArray[np.float32]:
        """Return normalized split statistics from the real trained native trees.

        ``kind="gain"`` accumulates native split gains. ``kind="split"`` counts split usage.
        Both variants read the single C++ tree representation exposed by the binding instead of
        re-implementing split logic in Python. A model with no internal split returns all zeros.
        """

        model = self._require_model()
        if kind not in {"gain", "split"}:
            raise ValueError("feature importance kind must be 'gain' or 'split'")
        values = np.zeros(self.n_features_in_, dtype=np.float64)
        for tree in model.trees:
            for node in tree.nodes:
                if node["is_leaf"]:
                    continue
                feature_index = int(node["feature_index"])
                if kind == "gain":
                    values[feature_index] += max(float(node["gain"]), 0.0)
                else:
                    values[feature_index] += 1.0
        total = float(values.sum())
        if total > 0.0:
            values /= total
        return values.astype(np.float32, copy=False)

    def _more_tags(self) -> dict[str, Any]:
        """Return sklearn compatibility tags without importing sklearn at runtime."""

        return {
            "X_types": ["2darray"],
            "allow_nan": False,
            "requires_y": True,
            "poor_score": False,
        }

    def __sklearn_tags__(self) -> Any:
        """Return sklearn 1.6+ structured tags while keeping sklearn optional.

        sklearn 1.6 changed several meta-estimator paths from the historical ``_more_tags``
        dictionary to a structured ``Tags`` object. Importing those classes at module import time
        would make sklearn a hard runtime dependency for normal MPSBoost users, so the import stays
        inside this compatibility hook and only runs when sklearn itself asks for tags.
        """

        try:
            from sklearn.utils import InputTags, RegressorTags, Tags, TargetTags
        except ImportError as exc:  # pragma: no cover - only reachable when sklearn is absent.
            raise AttributeError("sklearn tag classes are unavailable") from exc
        return Tags(
            estimator_type="regressor",
            target_tags=TargetTags(required=True, one_d_labels=True, single_output=True),
            regressor_tags=RegressorTags(poor_score=False),
            input_tags=InputTags(two_d_array=True, allow_nan=False, sparse=False),
            requires_fit=True,
        )

    def save_model(self, path: str | Path) -> None:
        """Save the model in a versioned format without training data or device identifiers."""

        self._require_model().save(str(path))

    def load_model(self, path: str | Path) -> "MPSBoostRegressor":
        """Load and validate a model, replacing fitted state only after complete success."""

        if not self._fit_lock.acquire(blocking=False):
            raise RuntimeError("模型训练或加载正在进行")
        try:
            candidate = _native._load_regression_model(str(path))
            self.model_ = candidate
            self.n_features_in_ = candidate.feature_count
            self.device_ = self.device if self.device != "auto" else "cpu"
            self.n_estimators_ = candidate.tree_count
            self.training_summary_ = {"loaded": True, "device": self.device_}
            return self
        finally:
            self._fit_lock.release()

    def _require_model(self) -> Any:
        """Return the complete native model or raise a stable unfitted exception."""

        if not hasattr(self, "model_"):
            raise NotFittedError("MPSBoostRegressor 尚未拟合或加载模型")
        return self.model_

    def _clear_fitted_state(self) -> None:
        """Delete every fitted field through one path to avoid stale partial state."""

        for name in (
            "model_",
            "n_features_in_",
            "device_",
            "device_decision_",
            "n_estimators_",
            "training_summary_",
        ):
            self.__dict__.pop(name, None)

    def _validate_parameters(self) -> None:
        """Validate all public Python parameters before device initialization."""

        integer_ranges = {
            "n_estimators": (self.n_estimators, 1, 2**32 - 1),
            "max_depth": (self.max_depth, 0, 31),
            "max_bins": (self.max_bins, 2, 65536),
            "min_samples_leaf": (self.min_samples_leaf, 1, 2**63 - 1),
            "verbosity": (self.verbosity, 0, 3),
        }
        for name, (value, minimum, maximum) in integer_ranges.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} 必须是整数")
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} 必须位于 [{minimum}, {maximum}]")
        for name, numeric_value, lower, upper, lower_inclusive in (
            ("learning_rate", self.learning_rate, 0.0, 1.0, False),
            ("min_child_weight", self.min_child_weight, 0.0, np.inf, True),
            ("reg_lambda", self.reg_lambda, 0.0, np.inf, True),
        ):
            if isinstance(numeric_value, bool) or not isinstance(
                numeric_value, (int, float)
            ):
                raise TypeError(f"{name} 必须是实数")
            numeric = float(numeric_value)
            valid_lower = numeric >= lower if lower_inclusive else numeric > lower
            if not np.isfinite(numeric) or not valid_lower or numeric > upper:
                bracket = "[" if lower_inclusive else "("
                raise ValueError(f"{name} 必须位于 {bracket}{lower}, {upper}]")
        if self.device not in {"mps", "cpu", "auto"}:
            raise ValueError("device 只能是 'mps'、'cpu' 或 'auto'")
        if self.random_state is not None and (
            isinstance(self.random_state, bool) or not isinstance(self.random_state, int)
        ):
            raise TypeError("random_state 必须是整数或 None")
