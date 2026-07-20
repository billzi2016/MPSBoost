"""CPU isolation-forest estimator for MPSBoost.

The implementation owns anomaly-specific random isolation trees and path-length scoring. It does
not reuse supervised split gain or allocate MPS resources. MPS requests warn and run the
in-project CPU path because this branch-heavy estimator is expected to be faster on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Any
import warnings

import numpy as np
from numpy.typing import NDArray

from ..matrix import as_dense_matrix
from .errors import NotFittedError


@dataclass(frozen=True)
class _IsolationNode:
    """Flat isolation-tree node stored in Python-owned immutable form."""

    feature: int
    threshold: float
    left: int
    right: int
    size: int
    is_leaf: bool


def _average_path_length(sample_count: int) -> float:
    """Return the standard unsuccessful-search path-length correction."""

    if sample_count <= 1:
        return 0.0
    if sample_count == 2:
        return 1.0
    return 2.0 * (log(sample_count - 1.0) + 0.5772156649015329) - (
        2.0 * (sample_count - 1.0) / sample_count
    )


class MPSIsolationForest:
    """Detect anomalies with independent random isolation trees on CPU."""

    _estimator_type = "outlier_detector"
    _PARAMETER_NAMES = (
        "n_estimators",
        "max_samples",
        "max_features",
        "contamination",
        "random_state",
        "device",
        "verbosity",
    )

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int | float = 256,
        max_features: float = 1.0,
        contamination: float = 0.1,
        random_state: int | None = None,
        device: str = "cpu",
        verbosity: int = 1,
    ) -> None:
        """Store parameters without sampling rows, allocating trees, or touching devices."""

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.contamination = contamination
        self.random_state = random_state
        self.device = device
        self.verbosity = verbosity

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters for sklearn-style model selection."""

        del deep
        return {name: getattr(self, name) for name in self._PARAMETER_NAMES}

    def set_params(self, **parameters: Any) -> "MPSIsolationForest":
        """Set known parameters and clear fitted state after any change."""

        unknown = sorted(set(parameters) - set(self._PARAMETER_NAMES))
        if unknown:
            raise ValueError(f"Unknown parameters: {', '.join(unknown)}")
        for name, value in parameters.items():
            setattr(self, name, value)
        if parameters:
            self._clear_fitted_state()
        return self

    def fit(self, X: Any, y: Any = None) -> "MPSIsolationForest":
        """Fit random isolation trees and choose the contamination threshold."""

        del y
        self._validate_parameters()
        resolved_device = self._resolved_device()
        matrix = as_dense_matrix(X).astype(np.float32, copy=False)
        if matrix.shape[0] < 2:
            raise ValueError("MPSIsolationForest requires at least two training rows")
        generator = np.random.default_rng(self.random_state)
        sample_count = self._resolved_max_samples(matrix.shape[0])
        feature_count = self._resolved_max_features(matrix.shape[1])
        max_depth = int(np.ceil(np.log2(max(sample_count, 2))))
        trees: list[tuple[tuple[_IsolationNode, ...], NDArray[np.int64]]] = []
        for _ in range(self.n_estimators):
            rows = generator.choice(matrix.shape[0], size=sample_count, replace=False)
            features = np.sort(
                generator.choice(matrix.shape[1], size=feature_count, replace=False)
            ).astype(np.int64)
            nodes: list[_IsolationNode] = []
            self._build_tree(matrix[rows][:, features], np.arange(sample_count), 0, max_depth, nodes, generator)
            trees.append((tuple(nodes), features))
        self.trees_ = tuple(trees)
        self.n_features_in_ = matrix.shape[1]
        self.max_samples_ = sample_count
        self.max_features_ = feature_count
        self.device_ = resolved_device
        scores = self.anomaly_score(matrix)
        self.offset_ = float(np.quantile(scores, 1.0 - float(self.contamination)))
        self.training_summary_ = {
            "n_estimators": self.n_estimators,
            "max_samples": sample_count,
            "max_features": feature_count,
            "requested_device": self.device,
            "device": resolved_device,
            "objective": "path_length",
        }
        if self.device != resolved_device:
            self.training_summary_["fallback_reason"] = (
                "CPU backend selected: this branch-heavy workload is expected to be faster on "
                "CPU than Apple GPU for this estimator."
            )
        return self

    def anomaly_score(self, X: Any) -> NDArray[np.float32]:
        """Return anomaly scores where larger values mean more anomalous rows."""

        matrix = self._checked_matrix(X)
        normalizer = _average_path_length(self.max_samples_)
        if normalizer <= 0.0:
            raise ValueError("max_samples_ is too small for anomaly scoring")
        path_lengths = np.zeros(matrix.shape[0], dtype=np.float64)
        for nodes, features in self.trees_:
            path_lengths += self._tree_path_lengths(nodes, matrix[:, features])
        path_lengths /= float(len(self.trees_))
        return np.power(2.0, -path_lengths / normalizer).astype(np.float32)

    def score_samples(self, X: Any) -> NDArray[np.float32]:
        """Return sklearn-style normality scores where larger values are less anomalous."""

        return -self.anomaly_score(X)

    def decision_function(self, X: Any) -> NDArray[np.float32]:
        """Return positive values for inliers and negative values for outliers."""

        return (self.offset_ - self.anomaly_score(X)).astype(np.float32)

    def predict(self, X: Any) -> NDArray[np.int64]:
        """Return 1 for inliers and -1 for outliers."""

        return np.where(self.decision_function(X) >= 0.0, 1, -1).astype(np.int64)

    def _build_tree(
        self,
        matrix: NDArray[np.float32],
        rows: NDArray[np.int64],
        depth: int,
        max_depth: int,
        nodes: list[_IsolationNode],
        generator: np.random.Generator,
    ) -> int:
        """Append one random isolation subtree and return its node index."""

        node_index = len(nodes)
        nodes.append(_IsolationNode(0, 0.0, -1, -1, int(rows.size), True))
        if depth >= max_depth or rows.size <= 1:
            return node_index
        ranges = np.ptp(matrix[rows], axis=0)
        candidates = np.flatnonzero(ranges > 0.0)
        if candidates.size == 0:
            return node_index
        feature = int(generator.choice(candidates))
        values = matrix[rows, feature]
        minimum = float(values.min())
        maximum = float(values.max())
        threshold = float(generator.uniform(minimum, maximum))
        left_rows = rows[values <= threshold]
        right_rows = rows[values > threshold]
        if left_rows.size == 0 or right_rows.size == 0:
            return node_index
        left = self._build_tree(matrix, left_rows, depth + 1, max_depth, nodes, generator)
        right = self._build_tree(matrix, right_rows, depth + 1, max_depth, nodes, generator)
        nodes[node_index] = _IsolationNode(feature, threshold, left, right, int(rows.size), False)
        return node_index

    def _tree_path_lengths(
        self, nodes: tuple[_IsolationNode, ...], matrix: NDArray[np.float32]
    ) -> NDArray[np.float64]:
        """Return path lengths for every row through one fitted tree."""

        result = np.zeros(matrix.shape[0], dtype=np.float64)
        for row_index, row in enumerate(matrix):
            node_index = 0
            depth = 0
            while True:
                node = nodes[node_index]
                if node.is_leaf:
                    result[row_index] = depth + _average_path_length(node.size)
                    break
                node_index = node.left if row[node.feature] <= node.threshold else node.right
                depth += 1
        return result

    def _checked_matrix(self, X: Any) -> NDArray[np.float32]:
        """Return dense input after fitted-state and feature-count checks."""

        if not hasattr(self, "trees_"):
            raise NotFittedError("MPSIsolationForest is not fitted")
        matrix = as_dense_matrix(X).astype(np.float32, copy=False)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("prediction feature count does not match training data")
        return matrix

    def _validate_parameters(self) -> None:
        """Validate isolation-forest parameters before fitting."""

        if self.device not in {"auto", "cpu", "mps"}:
            raise ValueError("device must be one of 'auto', 'cpu', or 'mps'")
        if isinstance(self.n_estimators, bool) or not isinstance(self.n_estimators, int):
            raise TypeError("n_estimators must be an integer")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if isinstance(self.contamination, bool) or not isinstance(self.contamination, (int, float)):
            raise TypeError("contamination must be numeric")
        if not 0.0 < float(self.contamination) < 0.5:
            raise ValueError("contamination must be in (0, 0.5)")
        if isinstance(self.max_features, bool) or not isinstance(self.max_features, (int, float)):
            raise TypeError("max_features must be numeric")
        if not 0.0 < float(self.max_features) <= 1.0:
            raise ValueError("max_features must be in (0, 1]")
        if self.random_state is not None and (
            isinstance(self.random_state, bool) or not isinstance(self.random_state, int)
        ):
            raise TypeError("random_state must be an integer or None")

    def _resolved_max_samples(self, row_count: int) -> int:
        if isinstance(self.max_samples, bool) or not isinstance(self.max_samples, (int, float)):
            raise TypeError("max_samples must be an integer or fraction")
        if isinstance(self.max_samples, float):
            if not 0.0 < self.max_samples <= 1.0:
                raise ValueError("fractional max_samples must be in (0, 1]")
            return min(row_count, max(2, int(np.ceil(row_count * self.max_samples))))
        if self.max_samples <= 1:
            raise ValueError("integer max_samples must be greater than 1")
        return min(int(self.max_samples), row_count)

    def _resolved_max_features(self, feature_count: int) -> int:
        return max(1, int(np.ceil(feature_count * float(self.max_features))))

    def _resolved_device(self) -> str:
        if self.device == "mps":
            warnings.warn(
                "CPU backend selected: MPSIsolationForest is branch-heavy and expected to be "
                "faster on CPU than Apple GPU for this workload.",
                RuntimeWarning,
                stacklevel=2,
            )
        return "cpu"

    def _clear_fitted_state(self) -> None:
        for name in (
            "trees_",
            "n_features_in_",
            "max_samples_",
            "max_features_",
            "device_",
            "offset_",
            "training_summary_",
        ):
            self.__dict__.pop(name, None)
