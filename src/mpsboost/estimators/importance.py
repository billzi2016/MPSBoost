"""Feature-importance and permutation-importance mixin for MPSBoost estimators.

This module keeps explanation helpers out of core training classes while reusing each estimator's
own prediction and scoring semantics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..matrix import as_dense_matrix


class FeatureImportanceMixin:
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

    def permutation_importance(
        self,
        X: Any,
        y: Any,
        *,
        n_repeats: int = 5,
        random_state: int | None = None,
    ) -> dict[str, NDArray[np.float32] | float]:
        """Estimate score drop after permuting each feature with the estimator's own score.

        This method intentionally delegates all prediction and scoring semantics to ``score``.
        Regression therefore uses R², classification uses accuracy, and future estimators can reuse
        the same implementation without copying metric logic into the explanation layer.
        """

        self._require_model()
        if isinstance(n_repeats, bool) or not isinstance(n_repeats, int):
            raise TypeError("n_repeats must be an integer")
        if n_repeats <= 0:
            raise ValueError("n_repeats must be positive")
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, int)
        ):
            raise TypeError("random_state must be an integer or None")
        matrix = as_dense_matrix(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("permutation feature count does not match training data")
        baseline_score = float(self.score(matrix, y))
        generator = np.random.default_rng(random_state)
        importances = np.zeros((self.n_features_in_, n_repeats), dtype=np.float32)
        for feature_index in range(self.n_features_in_):
            for repeat_index in range(n_repeats):
                permuted = np.array(matrix, copy=True)
                order = generator.permutation(matrix.shape[0])
                permuted[:, feature_index] = permuted[order, feature_index]
                importances[feature_index, repeat_index] = baseline_score - float(
                    self.score(permuted, y)
                )
        return {
            "baseline_score": baseline_score,
            "importances": importances,
            "importances_mean": importances.mean(axis=1),
            "importances_std": importances.std(axis=1),
        }

    def approximate_shap_values(
        self,
        X: Any,
        *,
        background: Any = None,
    ) -> dict[str, NDArray[np.float32] | float]:
        """Return controlled SHAP-like feature attributions from mean ablation.

        This is not exact TreeSHAP. It estimates per-feature effects by replacing one feature at a
        time with the background mean, then rescales each row so contributions sum to the model
        prediction minus the background baseline. Regression explains ``predict`` output; binary
        classification explains the positive-class probability.
        """

        self._require_model()
        matrix = as_dense_matrix(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("explanation feature count does not match training data")
        if background is None:
            background_matrix = matrix
        else:
            background_matrix = as_dense_matrix(background)
            if background_matrix.shape[1] != self.n_features_in_:
                raise ValueError("background feature count does not match training data")
        if background_matrix.shape[0] == 0:
            raise ValueError("background must contain at least one row")

        baseline = float(np.mean(self._explanation_output(background_matrix)))
        predictions = self._explanation_output(matrix).astype(np.float64, copy=False)
        background_mean = background_matrix.mean(axis=0)
        effects = np.zeros((matrix.shape[0], self.n_features_in_), dtype=np.float64)
        for feature_index in range(self.n_features_in_):
            masked = np.array(matrix, copy=True)
            masked[:, feature_index] = background_mean[feature_index]
            effects[:, feature_index] = predictions - self._explanation_output(masked)

        target_delta = predictions - baseline
        effect_sums = effects.sum(axis=1)
        contributions = np.zeros_like(effects)
        nonzero = np.abs(effect_sums) > 1e-12
        contributions[nonzero] = (
            effects[nonzero] * (target_delta[nonzero] / effect_sums[nonzero])[:, None]
        )
        return {
            "baseline": baseline,
            "contributions": contributions.astype(np.float32),
            "prediction_approximation": (
                baseline + contributions.sum(axis=1)
            ).astype(np.float32),
        }

    def _explanation_output(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        """Return the one-dimensional model output explained by attribution helpers."""

        if hasattr(self, "predict_proba"):
            probabilities = self.predict_proba(X)
            if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
                return probabilities[:, 1].astype(np.float64, copy=False)
        return self.predict(X).astype(np.float64, copy=False)


