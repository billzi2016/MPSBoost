"""Download real-world datasets into ignored local cache directories.

This script is intentionally separate from pytest. Tests must not silently
download data during ordinary CI runs. A developer or release runner should run
this script explicitly, inspect the generated manifest, and then run the
corresponding cached-data tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "tests" / "real_world" / "data"
MANIFEST_ROOT = PROJECT_ROOT / "tests" / "real_world" / "cache"


def _sha256(path: Path) -> str:
    """Return a streaming SHA256 digest for one cached dataset file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_files(root: Path) -> list[dict[str, Any]]:
    """Describe cached files without embedding machine-specific absolute paths."""

    files: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return files


def download_california_housing() -> Path:
    """Download California Housing through scikit-learn's official fetcher."""

    from sklearn.datasets import fetch_california_housing

    data_home = DATA_ROOT / "sklearn"
    data_home.mkdir(parents=True, exist_ok=True)
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    dataset = fetch_california_housing(data_home=data_home, download_if_missing=True)
    manifest = {
        "dataset": "California Housing",
        "source": "sklearn.datasets.fetch_california_housing",
        "data_home": str(data_home.relative_to(PROJECT_ROOT)),
        "cache_policy": "project-local ignored cache; deleting tests/real_world/data removes artifacts",
        "rows": int(dataset.data.shape[0]),
        "features": int(dataset.data.shape[1]),
        "target_rows": int(dataset.target.shape[0]),
        "files": _relative_files(data_home),
    }
    manifest_path = MANIFEST_ROOT / "california_housing_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def download_covertype_subset() -> Path:
    """Download Covertype through scikit-learn's official fetcher."""

    from sklearn.datasets import fetch_covtype

    data_home = DATA_ROOT / "sklearn"
    data_home.mkdir(parents=True, exist_ok=True)
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    dataset = fetch_covtype(data_home=data_home, download_if_missing=True)
    manifest = {
        "dataset": "Covertype",
        "source": "sklearn.datasets.fetch_covtype",
        "data_home": str(data_home.relative_to(PROJECT_ROOT)),
        "cache_policy": "project-local ignored cache; deleting tests/real_world/data removes artifacts",
        "rows": int(dataset.data.shape[0]),
        "features": int(dataset.data.shape[1]),
        "target_rows": int(dataset.target.shape[0]),
        "default_test_subset_rows": 30000,
        "files": _relative_files(data_home),
    }
    manifest_path = MANIFEST_ROOT / "covertype_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def _download_openml_dataset(name: str, *, version: int | str = "active") -> Path:
    """Download one OpenML dataset through scikit-learn's explicit fetcher."""

    from sklearn.datasets import fetch_openml

    data_home = DATA_ROOT / "openml"
    data_home.mkdir(parents=True, exist_ok=True)
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    dataset = fetch_openml(
        name=name,
        version=version,
        data_home=data_home,
        as_frame=True,
        parser="auto",
    )
    manifest = {
        "dataset": name,
        "source": "sklearn.datasets.fetch_openml",
        "version": version,
        "data_home": str(data_home.relative_to(PROJECT_ROOT)),
        "cache_policy": "project-local ignored cache; deleting tests/real_world/data removes artifacts",
        "rows": int(dataset.data.shape[0]),
        "features": int(dataset.data.shape[1]),
        "target_rows": int(dataset.target.shape[0]),
        "files": _relative_files(data_home),
    }
    manifest_path = MANIFEST_ROOT / f"{name.lower().replace('-', '_')}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def download_mnist_subset() -> Path:
    """Download MNIST through OpenML for the opt-in flattened-image subset test."""

    return _download_openml_dataset("mnist_784", version=1)


def download_titanic() -> Path:
    """Download Titanic through OpenML for missing-value categorical workflow tests."""

    return _download_openml_dataset("titanic", version=1)


def download_adult_income() -> Path:
    """Download Adult Income through OpenML for high-cardinality categorical tests."""

    return _download_openml_dataset("adult", version=2)


def main() -> int:
    """Parse the dataset selection and run the explicit download."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        choices=(
            "adult-income",
            "california-housing",
            "covertype-subset",
            "mnist-subset",
            "titanic",
        ),
        help="Dataset to download into ignored tests/real_world cache directories.",
    )
    arguments = parser.parse_args()
    if arguments.dataset == "california-housing":
        manifest_path = download_california_housing()
        print(f"Wrote manifest: {manifest_path}")
        return 0
    if arguments.dataset == "covertype-subset":
        manifest_path = download_covertype_subset()
        print(f"Wrote manifest: {manifest_path}")
        return 0
    if arguments.dataset == "mnist-subset":
        manifest_path = download_mnist_subset()
        print(f"Wrote manifest: {manifest_path}")
        return 0
    if arguments.dataset == "titanic":
        manifest_path = download_titanic()
        print(f"Wrote manifest: {manifest_path}")
        return 0
    if arguments.dataset == "adult-income":
        manifest_path = download_adult_income()
        print(f"Wrote manifest: {manifest_path}")
        return 0
    raise AssertionError("unreachable dataset selection")


if __name__ == "__main__":
    raise SystemExit(main())
