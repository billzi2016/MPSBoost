# All Trees v2 Specification

## 1. Goal

All Trees v2 progressively brings every practically valuable tree-model capability
suited to local tabular learning onto the Apple Silicon MPS/Metal backend. MPSBoost
must not stop at one regression GBDT. It must make common tree-training hot paths a
reusable foundation so boosting, bagging, random splitting, and inference share data,
memory, scheduling, and model semantics.

This document is the v2 product-direction specification after 0.2.0. Implementations
must continue to follow the constitution, PRD, SOLID/DRY, real testing, and no-mock rules.

## 1.1 Language Rules for v2 and Later

Starting with v2, persisted project materials must use English, while Agent/user
conversation remains Chinese. This includes code-file intent comments, function
comments, key implementation comments, commit messages, README files, release notes,
CI wording, error documentation, and public examples. This keeps the project
consistent for international users, external contributors, the PyPI/GitHub ecosystem,
and academic/engineering reproduction.

Existing v0/v1 Chinese materials may remain as historical specifications and
development records. New or substantially modified v2-and-later code, public
documentation, and release materials must switch to English. Specifications do not
require in-place translation; MkDocs maintains or generates separate language
versions. Agents still use Chinese to explain design tradeoffs in conversation.

## 2. Scope

### 2.1 Tree Model Families That Must Be Covered

- Histogram GBDT: the main line of current MPSBoost regression GBDT, continuing with
  classification, early stopping, regularization, sampling, and more complete
  training hot paths.
- LightGBM-like leaf-wise / level-wise boosting: supports histogram reuse,
  histogram subtraction, level- or leaf-wise scheduling, and small-leaf degradation
  strategies, but must not copy any third-party implementation.
- XGBoost-like estimators: provide familiar parameter names and fit/predict/save/load
  experience with clear errors; compatible experience does not mean copying source
  or promising complete API equivalence.
- Random Forest: independent-tree parallel training, row sampling, column sampling,
  regression averaging, and classification voting.
- ExtraTrees: random threshold candidates, random feature subsets, and highly
  parallel independent-tree training.
- CatBoost-like ordered boosting: ordered-boosting semantics, category-friendly
  paths, reproducible permutation strategies, and classification/regression entries,
  without copying third-party implementations or promising binary compatibility.
- CART single trees: retained as debugging, teaching, oracle, and lightweight-model
  paths, but must not become a second training semantic.
- Inference optimization: every trained tree model enters one unified flat model
  format and supports batch prediction hot paths.
- Isolation Forest: anomaly detection using independent-tree parallel training and
  batched path-length prediction.
- Quantile / Poisson / Tweedie boosting: common non-squared-error regression objectives.
- Ranking trees: ranking learning only after basic classification/regression is stable.

### 2.2 Explicitly Deferred Capabilities

- Distributed training, multi-GPU, and non-Apple-Silicon platforms;
- multi-output, multi-label, and complex categorical-feature combinations;
- complete parameter, model-binary, or source compatibility with third-party libraries;
- exposing unfinished model entries early for marketing.

## 3. Unified Foundation

Every model family shares the following foundation; do not duplicate low-quality
implementations for each model:

- input validation, dtype, finite values, contiguity, and lifetime checks;
- deterministic binning, schema, compact `uint8/uint16` representation, and stored model boundaries;
- domain functions for split gain, leaf weight, minimum samples, minimum Hessian, and stable tie-breaks;
- MPS gradients, histograms, split scan, partition/compaction, and prediction update;
- buffer pools, pipeline caches, dispatch plans, and synchronization error handling;
- model I/O, version validation, least privilege, and no telemetry;
- benchmarks, CPU oracle, real Metal tests, and clean-wheel installation validation.

## 3.1 Essential Tabular-Learning Semantics

The following capabilities are key to elevating tree models from merely runnable to
industrially usable. Design them centrally rather than assembling them temporarily in
each estimator:

- Missing values: NaN detection, default direction, optimal training-time default
  direction selection, and model persistence.
- Categorical features: explicit category metadata, low-cardinality category splits,
  high-cardinality fallback strategy, and unknown-category handling.
- Sample weights: all objectives, histograms, split gain, leaf values, and metrics
  handle weights consistently.
- Monotonic constraints: feature-level monotonic constraints enforced in split
  selection and leaf-value updates.
- Interaction constraints: limit feature combinations entering the same path to avoid
  violation of business constraints.
- Regularization: L1, L2, minimum gain, minimum child weight, maximum leaves, and
  leaf-value clipping.
- Explainability: gain, split count, permutation importance, and later SHAP-like approximations.
- Training observability: per-round metrics, tree count, effective splits,
  degradation reasons, and device-phase timings.

These semantics enter the CPU oracle before the MPS backend. Do not disguise
Python-layer preprocessing as backend support.

## 4. Design Principles

### 4.1 User Entry Points

Public entry points remain estimator-style. v2 may add specialized estimators:

- `GradientBoostingRegressor`
- `GradientBoostingClassifier`
- `RandomForestRegressor`
- `RandomForestClassifier`
- `ExtraTreesRegressor`
- `ExtraTreesClassifier`
- `CatBoostRegressor`
- `CatBoostClassifier`
- `MPSBoostRegressor`
- `MPSBoostClassifier`
- `MPSRandomForestRegressor`
- `MPSRandomForestClassifier`
- `MPSExtraTreesRegressor`
- `MPSExtraTreesClassifier`

Concise names are the primary entries; `MPS*` names remain project-brand and
backward-compatible aliases. Parameter names should resemble familiar tree-model
ecosystems, but unknown parameters fail early. Compatibility layers map user
experience only and must not make third-party internals, names, or source dependencies.

Model selection defaults to standard sklearn protocols and tools such as
`GridSearchCV`, `RandomizedSearchCV`, and `cross_val_score`. Do not lightly add
specialized search APIs such as `MPSGridSearchCV`; permit one only after real
benchmarks show GPU batch hyperparameter scheduling needs a separate API and it
substantially reduces user complexity or improves end-to-end performance.

### 4.2 Backend Capabilities

The MPS backend exposes computation only and does not own product-parameter
semantics. The training core chooses model families, sampling, objectives, tree-growth
strategies, and final split validation; the backend executes batchable real hot paths.

GPU hot-path priority:
1. resident or low-copy upload of binned data;
2. batched histograms across nodes, trees, and features;
3. split scan and histogram subtraction;
4. stable partition/compaction;
5. independent-tree parallel scheduling;
6. batched prediction traversal;
7. buffer pooling and synchronization compression.

### 4.3 Correctness

Every model family first has a CPU oracle and hand-calculated/property tests on small
data, then connects MPS hot paths. MPS results may have justified floating tolerance,
but tree structure, node constraints, and error semantics must be explainable and
reproducible.

### 4.4 Performance

Record performance conclusions separately for each model family. Small data, shallow
trees, narrow features, and extremely sparse effective splits may not suit GPUs; the
degraded range must be quantified and published. Every faster claim includes
end-to-end time, not one kernel alone.

### 4.5 Installation and Size

v2 retains light dependencies and prebuilt-wheel principles. Do not introduce heavy
runtime dependencies merely to cover more model families. New models must not package
test data, benchmarks, specifications, caches, or development toolchains in wheels.

## 5. Roadmap

### V2-A: Unified Tree Backend Abstraction
- Abstract objectives, sampling strategies, split strategies, tree-growth strategies,
  and prediction aggregation;
- retain current regression GBDT behavior;
- add a model-family capability matrix and explicit errors;
- test that abstractions do not duplicate mathematical formulas.

### V2-B: Classification and More Complete GBDT
- Implement binary logistic objectives;
- add `predict_proba`;
- implement early stopping, validation monitoring, and training diagnostics;
- keep MPS hot paths real.

### V2-C: Random Forest
- Implement bootstrap/without-replacement sampling;
- implement feature subsampling;
- support regression averaging and classification voting;
- schedule multi-tree training to MPS in batches, avoiding one-command-per-tree paths.

### V2-D: ExtraTrees
- Implement random threshold candidate generation;
- support reproducible random seeds;
- batch-evaluate random splits on GPUs;
- share sampling, aggregation, and prediction formats with Random Forest.

### V2-E: LightGBM-like Training Strategy
- Support a controlled leaf-wise growth version;
- support histogram subtraction, active-leaf queues, and small-leaf scheduling;
- state memory limits and overfitting controls;
- benchmark level-wise and leaf-wise end to end.

### V2-E2: CatBoost-like Ordered Boosting
- Support a controlled ordered-boosting semantic version;
- support category-friendly training paths and reproducible permutation strategies;
- share the unified foundation for binning, missing values, categorical features, and model I/O;
- clearly state compatibility boundaries differing from every third-party implementation.

### V2-F: Inference Hot Paths
- Implement MPS batch tree traversal or equivalent optimization;
- support shared prediction kernels across model families;
- validate that loaded-model predictions match post-training predictions;
- record CPU/MPS inference applicability boundaries.

### V2-G: Complete Industrial Semantics
- Implement missing-value default directions;
- implement sample weights;
- implement categorical feature splits;
- implement monotonic and interaction constraints;
- implement non-squared-error regression objectives;
- implement basic explainability.

### V2-H: Anomaly Detection and Ranking
- Implement Isolation Forest;
- implement ranking objectives and group/query input contracts;
- implement ranking metrics and validation monitoring;
- state MPS applicability boundaries for these models.

## 6. Acceptance Gates

- Every model family has CPU oracle, real MPS, integration, and model-I/O tests;
- every model family has at least two preregistered benchmark scenarios and degraded ranges;
- public README files describe only completed model families and make no early promises;
- wheel installation remains simple and size/dependency audits pass;
- deleting caches affects no model result;
- check all tasks only after every item in `tasks.md` is complete.
