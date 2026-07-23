"""Dataset matrix for real-world acceptance tests.

The matrix is executable documentation for S18. It records which datasets are
default CI checks, which ones require external downloads, and which project
capability must exist before a dataset can become a passing acceptance gate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    """One real-world dataset acceptance target."""

    name: str
    task: str
    source: str
    default_ci: bool
    status: str
    reason: str


DATASET_MATRIX: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="Breast Cancer Wisconsin",
        task="binary_classification",
        source="sklearn.datasets.load_breast_cancer",
        default_ci=True,
        status="active",
        reason="Binary classification is implemented and the dataset is bundled with sklearn.",
    ),
    DatasetSpec(
        name="Diabetes",
        task="regression",
        source="sklearn.datasets.load_diabetes",
        default_ci=True,
        status="active",
        reason="Regression is implemented and the dataset is bundled with sklearn.",
    ),
    DatasetSpec(
        name="Iris",
        task="multiclass_classification",
        source="sklearn.datasets.load_iris",
        default_ci=True,
        status="active",
        reason="The public classifier defaults to native CPU softmax for multiclass labels.",
    ),
    DatasetSpec(
        name="Digits",
        task="multiclass_classification",
        source="sklearn.datasets.load_digits",
        default_ci=True,
        status="active",
        reason="The public classifier defaults to native CPU softmax for multiclass labels.",
    ),
    DatasetSpec(
        name="California Housing",
        task="regression",
        source="sklearn.datasets.fetch_california_housing",
        default_ci=False,
        status="active_cached",
        reason="Uses explicit download script and ignored local cache; tests never download implicitly.",
    ),
    DatasetSpec(
        name="MNIST subset",
        task="multiclass_classification",
        source="external opt-in cache",
        default_ci=False,
        status="active_cached",
        reason="Uses explicit OpenML cache download and a fixed flattened-image subset policy.",
    ),
    DatasetSpec(
        name="Titanic",
        task="binary_classification_categorical_missing",
        source="external opt-in cache",
        default_ci=False,
        status="active_cached",
        reason="Uses explicit OpenML cache download and categorical/missing-value preprocessing.",
    ),
    DatasetSpec(
        name="Adult Income",
        task="binary_classification_high_cardinality_categorical",
        source="external opt-in cache",
        default_ci=False,
        status="active_cached",
        reason="Uses explicit OpenML cache download, categorical stress coverage, and visible fairness notes.",
    ),
    DatasetSpec(
        name="Covertype subset",
        task="multiclass_classification_large_rows",
        source="external opt-in cache",
        default_ci=False,
        status="active_cached",
        reason="Uses explicit sklearn fetcher download, ignored local cache, and fixed large-row subset.",
    ),
    DatasetSpec(
        name="Higgs subset",
        task="binary_classification_large_numeric",
        source="external opt-in cache",
        default_ci=False,
        status="active_local_file",
        reason="Uses an explicit local HIGGS CSV file under the ignored data directory.",
    ),
)


def active_default_datasets() -> tuple[DatasetSpec, ...]:
    """Return datasets that should run in ordinary no-network CI."""

    return tuple(item for item in DATASET_MATRIX if item.default_ci and item.status == "active")
