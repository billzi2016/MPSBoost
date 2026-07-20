# V3 Real World Tests Specification

## 1. Goal

V3 Real World Tests are the real-world acceptance gate before a stable `1.x`
release. After `0.x` prereleases, capability expansion, and engineering
optimization, MPSBoost must demonstrate correctness, stability, installation
experience, and end-to-end performance on public, reproducible datasets with real
tabular-learning difficulty before releasing any `1.x` version.

Until this stage is complete, public PyPI versions may use only `0.x`. `1.0.0`
means a stable commitment real users may depend upon; it must not be reached merely
because many features exist, CI passes, or synthetic benchmarks pass.

## 2. Version Discipline

- `0.x`: permits rapid iteration, prereleases, and capability expansion, but
  limitations must be stated honestly.
- `0.x alpha/beta/rc`: used for module milestones, performance gates, and
  stability candidates.
- `1.x`: released only after every V3 real-world dataset acceptance passes.
- Published versions cannot be overwritten; every fix increments a new version.
- Synthetic benchmarks prove only local capability and cannot alone justify a `1.x` release.

## 3. Dataset Requirements

Real-world test datasets must:

- have legal sources, clear licenses, and user-reproducible acquisition;
- cover delivered model families for regression, binary classification, multiclass
  classification, anomaly detection, and ranking;
- include small, medium, many-row, wide-table, high-cardinality categorical,
  missing-value, and skewed-label data;
- define train/validation/test splits;
- record downloading, validation, preprocessing, and caching strategies;
- not package raw, private, or unclearly licensed data in wheels.

### 3.1 Dataset Acceptance Matrix

V3 must establish the `tests/real_world/` suite. It answers whether the library is
reliable on data shapes real users encounter, rather than replacing unit,
integration, or synthetic-benchmark tests.

The first dataset set prioritizes stable sources, clear licenses, caching, and
offline reruns:

| Dataset | Task type | Main purpose | Acquisition strategy |
| --- | --- | --- | --- |
| Iris | Multiclass | Minimal multiclass sanity test; validates sklearn workflows, class probabilities, and save/load. | Prefer built-in `sklearn.datasets.load_iris` data. |
| Breast Cancer Wisconsin | Binary classification | Validates binary probabilities, thresholds, AUC/accuracy, and small/medium numerical tabular behavior. | Prefer built-in `sklearn.datasets.load_breast_cancer` data. |
| Diabetes | Regression | Small regression sanity test for metrics, model I/O, and reproducibility. | Prefer built-in `sklearn.datasets.load_diabetes` data. |
| California Housing | Regression | Medium regression baseline covering real numerical tables and performance boundaries. | Use `sklearn.datasets.fetch_california_housing`; cache it and permit offline reruns. |
| Digits | Multiclass | Lightweight multiclass tree-model stress test over flattened images. | Prefer built-in `sklearn.datasets.load_digits` data. |
| MNIST subset | Multiclass | Flattened-feature stress test nearer real image-classification scale; limit samples to keep CI bounded. | External download locks version, caches, and validates hash; it does not block ordinary tests by default. |
| Titanic | Binary classification + missing values + categorical features | Validates real cleaning, missing-value strategy, categorical encoding, and pipeline compatibility. | Use a stable mirror or project download script; do not package raw data in wheels. |
| Adult Income | Binary classification + high-cardinality features | Validates categorical features, larger row counts, category cardinality, and fairness-record metrics. | External download caches, validates hash, and supports a manual predownload path. |
| Covertype subset | Multiclass + many rows | Validates larger row counts, multiclass, and training-throughput boundaries. | External/OpenML acquisition is opt-in and uses a fixed subset by default. |
| Higgs subset | Binary classification + performance boundary | Validates large numerical-table throughput, peak memory, and CPU/MPS degradation boundaries. | Opt-in long test only; limit subset size and record machine information. |

### 3.2 Data and Cache Discipline

- `tests/real_world/` permits neither mock backends nor synthetic data presented as
  real-data acceptance.
- Ordinary CI runs only built-in dataset tests requiring no network or external
  downloads and with controlled runtime.
- External datasets download through project scripts into layered-cache directories
  and do not download again on cache hits.
- Download scripts record source, license, version, file size, hash, and local cache path.
- Preprocessing is reproducible; random splits use fixed seeds and record
  train/validation/test ratios.
- When downloading fails, tests explicitly skip and state the missing data file;
  they must not silently degrade to mocks.
- Real-world performance tests record device, OS version, Python version, package
  version, data scale, and parameter configuration.

## 4. Reference Baselines

Every delivered model family needs at least one strong CPU baseline and one project
CPU-oracle comparison:

- CPU oracle validates numerical semantics and boundaries;
- strong CPU baselines reference performance and model quality;
- GPU results record training time, prediction time, metrics, model size, and peak memory;
- publish small-data degradation rather than showing only winning cases.

## 5. Quality Gates

Before entering `1.x`, all of the following must hold:

- Clean-environment installation succeeds without local compilation;
- real-dataset training, prediction, saving, loading, and reproduction all pass;
- end-to-end reports cover preprocessing, binning, training, synchronization,
  prediction, and model I/O;
- model quality is not materially worse than CPU baselines; differences require explanation;
- deleting, corrupting, or disabling caches does not affect results;
- long repeated training has no memory growth, resource leak, or GPU command failure;
- documentation clearly states support, unsupported scope, and degradation ranges.

## 6. 1.x Release Gate

`tasks.md` may allow `1.0.0` only after all of the following complete:

1. The V1 MPS histogram engine is stable;
2. Every V2 arboretum model family planned for 1.0 is complete;
3. The entire V3 real-world test matrix passes;
4. CI, PyPI, wheel, license, size, and permission audits all pass;
5. The user gives final confirmation of public-commitment scope, version, and artifact hash.
