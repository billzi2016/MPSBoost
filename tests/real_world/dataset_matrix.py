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
        reason="The public classifier supports multiclass through real one-vs-rest native models.",
    ),
    DatasetSpec(
        name="Digits",
        task="multiclass_classification",
        source="sklearn.datasets.load_digits",
        default_ci=True,
        status="active",
        reason="The public classifier supports multiclass through real one-vs-rest native models.",
    ),
    DatasetSpec(
        name="California Housing",
        task="regression",
        source="sklearn.datasets.fetch_california_housing",
        default_ci=False,
        status="planned",
        reason="Requires cached download handling before it can run in default acceptance.",
    ),
    DatasetSpec(
        name="MNIST subset",
        task="multiclass_classification",
        source="external opt-in cache",
        default_ci=False,
        status="planned",
        reason="Requires external download, hash checks, fixed subset policy, and multiclass support.",
    ),
    DatasetSpec(
        name="Titanic",
        task="binary_classification_categorical_missing",
        source="external opt-in cache",
        default_ci=False,
        status="planned",
        reason="Requires licensed cached download and reproducible preprocessing.",
    ),
    DatasetSpec(
        name="Adult Income",
        task="binary_classification_high_cardinality_categorical",
        source="external opt-in cache",
        default_ci=False,
        status="planned",
        reason="Requires licensed cached download, categorical stress coverage, and fairness notes.",
    ),
    DatasetSpec(
        name="Covertype subset",
        task="multiclass_classification_large_rows",
        source="external opt-in cache",
        default_ci=False,
        status="planned",
        reason="Requires external cache policy and multiclass support.",
    ),
    DatasetSpec(
        name="Higgs subset",
        task="binary_classification_large_numeric",
        source="external opt-in cache",
        default_ci=False,
        status="planned",
        reason="Requires opt-in long-test policy, hash checks, and performance reporting.",
    ),
)


def active_default_datasets() -> tuple[DatasetSpec, ...]:
    """Return datasets that should run in ordinary no-network CI."""

    return tuple(item for item in DATASET_MATRIX if item.default_ci and item.status == "active")
