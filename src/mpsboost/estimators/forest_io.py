"""Forest model persistence helpers for MPSBoost estimators.

Forest containers store a small manifest plus one native model file per tree. This module keeps zip
I/O out of the forest training and prediction implementation.
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from .single_tree import DecisionTreeClassifier, DecisionTreeRegressor


class ForestPersistenceMixin:
    def save_model(self, path: str | Path) -> None:
        """Save a forest container while reusing the native format for every tree."""

        self._require_model()
        target = Path(path)
        directory = target.parent if target.parent != Path("") else Path(".")
        if not directory.is_dir():
            raise ValueError("model save directory does not exist")
        manifest = {
            "format": "mpsboost-forest",
            "version": 1,
            "estimator": type(self).__name__,
            "n_features_in": int(self.n_features_in_),
            "feature_subsets": [features.tolist() for features in self.feature_subsets_],
            "parameters": self.get_params(),
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.mpsboost-",
            suffix=".tmp",
            dir=directory,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with zipfile.ZipFile(temporary, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
                with tempfile.TemporaryDirectory(dir=directory) as tree_directory:
                    tree_root = Path(tree_directory)
                    for index, tree in enumerate(self.estimators_):
                        tree_path = tree_root / f"tree_{index}.mb"
                        tree.save_model(tree_path)
                        archive.write(tree_path, f"trees/tree_{index}.mb")
            os.replace(temporary, target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def load_model(self, path: str | Path) -> "_ForestMixin":
        """Load a forest container and validate every embedded native tree."""

        if not self._fit_lock.acquire(blocking=False):
            raise RuntimeError("model training or loading is already in progress")
        try:
            with zipfile.ZipFile(Path(path), mode="r") as archive:
                manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
                if manifest.get("format") != "mpsboost-forest" or manifest.get("version") != 1:
                    raise ValueError("unsupported random forest model format")
                if manifest.get("estimator") != type(self).__name__:
                    raise ValueError("random forest estimator type is incompatible")
                feature_subsets = tuple(
                    np.asarray(features, dtype=np.int64)
                    for features in manifest["feature_subsets"]
                )
                estimators: list[DecisionTreeRegressor | DecisionTreeClassifier] = []
                with tempfile.TemporaryDirectory() as tree_directory:
                    tree_root = Path(tree_directory)
                    for index, features in enumerate(feature_subsets):
                        tree_path = tree_root / f"tree_{index}.mb"
                        tree_path.write_bytes(archive.read(f"trees/tree_{index}.mb"))
                        tree = self._tree_type(device=self.device).load_model(tree_path)
                        if tree.n_features_in_ != len(features):
                            raise ValueError("forest tree feature subset does not match model")
                        estimators.append(tree)
            self.estimators_ = tuple(estimators)
            self.feature_subsets_ = feature_subsets
            self.n_features_in_ = int(manifest["n_features_in"])
            self.n_estimators_ = len(estimators)
            self.device_ = self.device if self.device != "auto" else "cpu"
            self.training_summary_ = {"loaded": True, "n_estimators": len(estimators)}
            self._finalize_fitted_metadata()
            return self
        except Exception:
            self._clear_fitted_state()
            raise
        finally:
            self._fit_lock.release()

