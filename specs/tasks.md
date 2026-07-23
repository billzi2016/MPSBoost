# Implementation Task List

> **Checkbox discipline: change `[ ]` to `[x]` only after code, English program text, tests, documentation, and acceptance are all complete. Partial work, skipped validation, or environment blockers must not be checked off.**

> **Version discipline: before all real-world dataset acceptance in `v3-real-world-tests/prd.md` is complete, only `0.x` versions may be released. `1.x` is a stable user commitment and must not be released based only on synthetic benchmarks, feature count, or ordinary CI success.**

> **Current release discipline: `0.3.0` is the all-trees feature milestone before large-scale validation. `0.4.0` is the release after large-scale and real-world dataset validation. `0.5.0` is the zero-known-issue hardening release. Do not publish `1.0.0` until every planned feature, real-world matrix, performance/memory/permission audit, artifact hash, installation/environment fallback, customer-facing failure path, documentation page, release audit, CI run, PyPI artifact, and fresh-install verification is complete and explicitly closed.**

> **Release cadence: after the formal `0.2.0` release, do not publish PyPI for small internal modules. The next public feature milestone is fixed as `0.3.0`, targeting the all-trees capability planned in `v2-arboretum-implementation/prd.md`; intermediate development should commit, push, and preserve CI artifacts unless the user explicitly approves another prerelease.**

> **Current execution order: after S15, prioritize S18 real-world dataset tests and reports, then S16 advanced objectives and explanations, then S17 anomaly detection and ranking. Unless the user explicitly changes the order, do not jump directly from S15 to S16 or S17 by numeric order.**

## S0: V0 Constitution Specs and Repository Setup

- [x] S0.1 Create the constitution, PRD, project tree, and task list.
- [x] S0.2 Create design specs for each core module.
- [x] S0.3 Create `specs/AGENTS.md` project agent rules while keeping normal user entry points concise.
- [x] S0.4 Audit against specs and remove existing mocks, placeholder APIs, and duplicated logic.
- [x] S0.5 Create public README, license, and real status description.
- [x] S0.6 Initialize the Git repository and confirm the public remote.

## S1: V1 MPS Histogram Engine Build and Minimal Real Backend

- [x] S1.1 Create the CMake + Python build system with a single version source.
- [x] S1.2 Implement Objective-C++ device discovery and capability checks.
- [x] S1.3 Compile and package a minimal real `.metallib` at build time.
- [x] S1.4 Implement a real vector smoke kernel with synchronized error handling.
- [x] S1.5 Build an arm64 wheel and verify installation in a clean environment.
- [x] S1.6 Accept G0: no heavyweight runtime dependency, no local compile requirement, and no mocks.

## S2: Data and Binning

- [x] S2.1 Implement input shape, dtype, finite-value, and overflow validation.
- [x] S2.2 Implement deterministic quantile boundaries and duplicate-boundary handling.
- [x] S2.3 Implement the unique `uint8/uint16` binned representation.
- [x] S2.4 Implement data ownership, contiguous layout, and copy diagnostics.
- [x] S2.5 Complete tests for constant features, skewed data, empty inputs, and boundaries.
- [x] S2.6 Accept G1: binning is fully deterministic, serializable, and free of lifetime bugs.
- [x] S2.7 Commit and push the GitHub module, with CI preserving the verified wheel artifact.

## S3: CPU Oracle and Single Tree

- [x] S3.1 Implement squared-error gradients and Hessians.
- [x] S3.2 Implement the unique split-gain and leaf-weight functions.
- [x] S3.3 Implement CPU histograms and stable tie-breaking.
- [x] S3.4 Implement a depth-limited single regression tree.
- [x] S3.5 Implement flat-node prediction.
- [x] S3.6 Verify every split, leaf value, and prediction with hand-calculated data.
- [x] S3.7 Accept G2: domain semantics are frozen and protected by the oracle.
- [x] S3.8 Commit and push the GitHub module, with CI preserving the verified wheel artifact.

## S4: MPS Histogram

- [x] S4.1 Implement real gradient/Hessian kernels.
- [x] S4.2 Implement the correctness-baseline histogram kernel.
- [x] S4.3 Implement threadgroup partial histograms.
- [x] S4.4 Implement partial histogram reduction.
- [x] S4.5 Complete numerical comparison tests for multiple bins, skew, non-even groups, and edge sizes.
- [x] S4.6 Complete profiler records and memory-safety checks.
- [x] S4.7 Accept G3: the preregistered large-scale histogram scenario reaches the CPU-oracle target.
- [x] S4.8 Commit and push the GitHub module and benchmark, with CI preserving the wheel artifact.

## S5: Real Regression GBDT

- [x] S5.1 Implement the trainer state machine and multi-round boosting.
- [x] S5.2 Implement tree-level prediction updates.
- [x] S5.3 Implement `MPSBoostRegressor.fit/predict`.
- [x] S5.4 Implement not-fitted, device-failure, and parameter-conflict errors.
- [x] S5.5 Implement model save/load and round-trip tests.
- [x] S5.6 Accept G4: end-to-end real training is correct, with no mocks or silent fallback.
- [x] S5.7 Commit and push to GitHub, publish PyPI `0.2.0a0`, and reverify from PyPI.

## S6: GPU Hot Path Optimization

- [x] S6.1 Implement the split-scan kernel.
- [x] S6.2 Implement the partition/compaction kernel.
- [x] S6.3 Implement histogram subtraction.
- [x] S6.4 Implement batched level-wise active-node processing.
- [x] S6.5 Implement the buffer pool and synchronization reduction.
- [x] S6.6 Quantify the small-node degradation range and define the single scheduling policy.
- [x] S6.7 Accept G5: preregistered end-to-end scenarios meet the performance target.
- [x] S6.8 Commit and push to GitHub, publish PyPI `0.2.0b0`, and reverify from PyPI.

## S7: Cache and Stability

- [x] S7.1 Implement L1 pipeline and buffer process cache.
- [x] S7.2 Implement L2 version keys, atomic writes, and corruption rebuilds.
- [x] S7.3 Implement L3 CI build cache and strict invalidation conditions.
- [x] S7.4 Implement query-only, explicit-create, and safe-clean cache APIs.
- [x] S7.5 Complete repeated training, cache corruption, and memory growth tests.
- [x] S7.6 Accept G6: deleting all caches does not change results.
- [x] S7.7 Commit and push to GitHub, publish PyPI `0.2.0rc0`, and reverify from PyPI.

## S8: 0.2.0 Release

- [x] S8.1 Complete public English README, API, and limitation notes.
- [x] S8.2 Complete Apache-2.0 and dependency license audit.
- [x] S8.3 Complete wheel content, dynamic-link, and absolute-path audit.
- [x] S8.4 Complete CPU, MPS, integration, installation, and stability tests.
- [x] S8.5 Complete reproducible benchmark and raw report.
- [x] S8.6 Commit each cohesive module with a Chinese commit message of at most 10 lines.
- [x] S8.7 Push to the confirmed public GitHub remote.
- [x] S8.8 Complete isolated release verification with the verified artifact.
- [x] S8.9 Obtain final user confirmation for package name, version, and artifact hashes.
- [x] S8.10 Publish formal PyPI `0.2.0` and verify a fresh install.

## S9: V2 Arboretum Implementation Specs and Unified Foundation

- [x] S9.1 Review `v2-arboretum-implementation/prd.md` and confirm v2 scope, non-goals, and acceptance gates.
- [x] S9.2 Design the unified tree-family abstraction: objectives, sampling, split policy, tree growth, and prediction aggregation.
- [x] S9.3 Preserve existing regression GBDT behavior and model-format backward compatibility.
- [x] S9.4 Create the model-family capability matrix and early-failure errors for unsupported capabilities.
- [x] S9.5 Define the `0.3.0` all-trees release scope and forbid frequent PyPI releases for small modules.
- [x] S9.6 Accept G8: new abstractions do not duplicate math formulas or introduce a second training semantics.

## S10: Classification GBDT and Complete Boosting

- [x] S10.1 Implement the single mathematical semantics for the binary logistic objective.
- [x] S10.2 Implement `GradientBoostingClassifier` / `MPSBoostClassifier` fit, predict, and predict_proba for strict binary 0/1 labels.
- [x] S10.3 Implement early stopping, validation monitoring, and training diagnostics.
- [x] S10.4 Complete CPU oracle, real MPS, model I/O, and integration tests.
- [x] S10.5 Accept G9: classification GBDT is correct, stable, truly MPS-accelerated, and has no silent fallback.

## S11: Random Forest

- [x] S11.1 Implement bootstrap/without-replacement sampling and feature subsampling.
- [x] S11.1a Implement `DecisionTreeRegressor` and `DecisionTreeClassifier` as one-tree native estimators.
- [x] S11.2 Implement `RandomForestRegressor.fit/predict` using independent native decision trees.
- [x] S11.3 Implement `RandomForestClassifier.fit/predict/predict_proba` using independent native decision trees.
- [x] S11.4a Implement deterministic `n_jobs` scheduling for independent random-forest trees.
- [x] S11.4 Implement independent-tree batch training scheduling and shared MPS hot paths.
- [x] S11.5 Complete regression averaging, classification probability aggregation, random seed, and model I/O tests.
- [x] S11.6 Accept G10: Random Forest is end-to-end real and usable, with performance and degradation ranges recorded.

## S12: ExtraTrees

- [x] S12.1 Implement reproducible random threshold candidate generation.
- [x] S12.2 Implement `ExtraTreesRegressor` using native random-threshold split candidates.
- [x] S12.3 Implement `ExtraTreesClassifier` using native random-threshold split candidates.
- [x] S12.4 Implement native random-threshold split policy for CPU and MPS tree training.
- [x] S12.5 Share sampling, aggregation, and forest container prediction format with Random Forest.
- [x] S12.6 Accept G11: ExtraTrees is correct and stable, with an observable real MPS path.

## S13: LightGBM-like and CatBoost-like Training Strategies

- [x] S13.1 Implement controlled leaf-wise growth.
- [x] S13.2 Implement active-leaf queues, memory limits, and overfitting controls.
- [x] S13.3 Extend histogram subtraction and small-leaf scheduling policy.
- [x] S13.4 Benchmark level-wise and leaf-wise end to end.
- [x] S13.5 Design CatBoost-like ordered boosting, categorical-feature-friendly paths, and permutation semantics.
- [x] S13.6 Implement controlled-compatible `CatBoostRegressor` and `CatBoostClassifier` entries.
- [x] S13.7 Accept G12: LightGBM-like and CatBoost-like strategies have real acceleration and honest public limits.

## S14: Unified Inference Hot Path

- [x] S14.1 Implement a shared flat prediction format across model families.
- [x] S14.2 Implement MPS batch tree traversal or an equivalent prediction optimization.
- [x] S14.3 Complete CPU/MPS prediction consistency tests after save/load.
- [x] S14.4 Quantify the applicability boundary for training and inference acceleration.
- [x] S14.5 Accept G13: all delivered tree families share a stable inference path.

## S15: Industrial Tabular Semantics

- [x] S15.1 Implement missing-value detection, default-direction training, and model save.
- [x] S15.2 Apply sample weights through objectives, histograms, split gains, leaf values, and metrics.
- [x] S15.3 Implement categorical feature metadata, categorical splits, and unknown-category handling.
- [x] S15.4 Implement monotonic constraints and verify all related splits and leaf values satisfy constraints.
- [x] S15.5 Implement interaction constraints and verify path feature combinations stay within bounds.
- [x] S15.6 Implement L1, max leaves, leaf-value clipping, and more regularization controls.
- [x] S15.7 Accept G14: industrial tabular semantics are consistent across the CPU oracle and MPS backend.

## S16: Advanced Objectives and Explanations

> Schedule: this stage proceeds after S18 real-world dataset testing is complete.

- [x] S16.1 Implement the quantile objective.
- [x] S16.2 Implement the Poisson objective.
- [x] S16.3 Implement the Tweedie objective.
- [x] S16.4 Complete feature importance: gain, split count, and permutation.
- [x] S16.4a Implement gain and split-count feature importance from native fitted tree nodes.
- [x] S16.4b Implement permutation importance without duplicating prediction or scoring logic.
- [x] S16.5 Design and implement a controlled SHAP-like approximate explanation.
- [x] S16.5a Design the official SHAP integration path: optional dependency `mpsboost[shap]`, native tree adapter/export, TreeExplainer semantic validation, and research examples; do not claim approximate explanations are official SHAP.
- [x] S16.6 Accept G15: advanced objectives and explanations have real tests, documentation, and performance boundaries.

## S17: Anomaly Detection and Ranking

> Schedule: this stage proceeds after S16 advanced objectives and explanations is complete.

- [x] S17.1 Implement `MPSIsolationForest`.
- [x] S17.2 Implement path length, anomaly score, and batch prediction hot path.
- [x] S17.3 Design the ranking input contract: group/query, label, and metrics.
- [x] S17.4 Implement the basic ranking objective and validation monitoring.
- [x] S17.5 Complete anomaly-detection and ranking CPU-oracle, MPS, and benchmark tests.
- [x] S17.6 Accept G16: anomaly detection and ranking are truly usable with clear limits.

## S18: V3 Real-World Tests and 1.x Release Gate

> Schedule: enter this stage immediately after S15 industrial tabular semantics, prioritizing real-world datasets for completed model families.

- [x] S18.1 Review `v3-real-world-tests/prd.md` and confirm the 1.x release discipline.
- [x] S18.2 Create a legal, reproducible real-world dataset matrix.
- [x] S18.2a Create the `tests/real_world/` directory and real-world test rules.
- [x] S18.2b Implement built-in dataset acceptance: Iris, Breast Cancer, Diabetes, Digits.
- [x] S18.2c Implement cached-download dataset acceptance: California Housing.
- [x] S18.2d Implement opt-in external dataset acceptance: MNIST subset, Titanic, Adult Income, Covertype subset, Higgs subset.
- [x] S18.2d-1 Implement MNIST subset acceptance.
- [x] S18.2d-2 Implement Titanic acceptance.
- [x] S18.2d-3 Implement Adult Income acceptance.
- [x] S18.2d-4 Implement Covertype subset large-row acceptance and real MPS parity smoke.
- [x] S18.2d-5 Implement Higgs subset acceptance.
- [x] S18.3 Cover implemented model families across regression, classification, anomaly detection, and ranking.
- [x] S18.4 Create strong CPU baseline, project CPU oracle, and real MPS comparison reports.
- [x] S18.5 Complete training, prediction, save/load, cache deletion, and repeated-training stability tests.
- [x] S18.5a Cover model save/load, cache deletion, cache corruption, and repeated-training stability on real datasets.
- [ ] S18.6 Complete model quality, end-to-end performance, peak memory, wheel size, and permission audit.
- [ ] S18.6a Record real-dataset train time, predict time, peak memory, model size, wheel size, and permission scope.
- [x] S18.7 Publish the real-world dataset report with honest success, degradation, and unsupported scenarios.
- [ ] S18.8 Accept G17: every real-world test matrix item passes before planning `1.0.0`, including documented quality, performance, memory, model-size, wheel-size, and permission evidence.
- [ ] S18.9 Obtain final user confirmation for 1.x public commitment scope, version, artifact hashes, documentation completeness, and customer-facing failure-path behavior.
- [ ] S18.10 Publish PyPI `1.0.0` only after all docs and release audits are final, then reverify from a fresh formal PyPI environment.

## S24: 0.4.0 Large-Scale Validation Release

- [x] S24.1 Define `0.4.0` as the 0.x release after large-scale and real-world dataset validation, distinct from the pre-large-scale `0.3.0` all-trees milestone.
- [ ] S24.2 Publish PyPI `0.4.0` only after the current 0.4.0 wheel artifacts, CI results, and smoke verification are recorded.

## S25: 0.5.0 Zero-Known-Issue Hardening Release

- [ ] S25.1 Triage every known runtime, documentation, packaging, environment, and user-experience issue into fixed, intentionally deferred, or impossible-under-current-platform categories.
- [ ] S25.2 Ensure missing optional dependencies, missing Metal toolchain, unsupported Linux/CUDA environments, and CPU-suitable workloads produce copy-paste guidance or warnings instead of confusing failures.
- [ ] S25.3 Publish PyPI `0.5.0` only when there are no known blocking customer-facing issues.

## S26: 1.0.0 Final Customer-Commitment Gate

- [ ] S26.1 Freeze final public scope and confirm no planned feature is being silently excluded from the `1.0.0` promise.
- [ ] S26.2 Confirm documentation is complete, bilingual, linked, and free of stale version claims.
- [ ] S26.3 Confirm release audits, known-issue audits, performance reports, artifact hashes, CI results, and PyPI fresh-install verification are complete.
- [ ] S26.4 Confirm customer-facing failures use warnings, copy-paste setup commands, or clear external-dependency attribution wherever execution cannot continue.
- [ ] S26.5 Publish PyPI `1.0.0` only after S26.1-S26.4 are complete and the user explicitly approves the final release.

## S19: File Structure Reaches Release Maintenance Standard

- [x] S19.1 Create the file length rule: 200 lines is the default target, and files above 300 lines must be split or registered as exceptions.
- [x] S19.2 Split Python estimator implementations while preserving public API and historical import paths.
- [x] S19.3 Split estimator unit tests to prevent single test files from growing further.
- [x] S19.4 Split Python binding files, isolating buffer, dataset, model, and backend test helpers.
- [x] S19.5 Split native binned dataset implementation, isolating validation, quantization, schema, and serialization.
- [x] S19.6 Split native tree implementation, isolating split scan, growth, prediction, and restore validation.
- [x] S19.7 Accept G18: all newly added/refactored files satisfy length rules, with exceptions documented and follow-up tasks scheduled.
- [x] S19.8 Split MPS backend implementation, isolating Metal context, gradient, histogram, split partition, and lifetime glue.

## S20: Clean Program-File Chinese Legacy Issues

- [x] S20.1 Read `specs/legacy-issues.md` before starting, and follow its search scope, prohibitions, and completion standard.
- [x] S20.2 Translate all comments, docstrings, test descriptions, assertion messages, runtime errors, and package metadata in program files to English.
- [x] S20.3 Documentation files are outside this task; `specs/`, README, and future bilingual site content may keep Chinese.
- [x] S20.4 Use the `rg` commands specified in `specs/legacy-issues.md` to search for Chinese, and extend the search scope when adding program directories.
- [x] S20.5 Check off only after search commands show no Chinese hits in program files, relevant tests pass, and behavior is unchanged.

## S21: Native Multiclass Softmax and sklearn-Compatible Interface

- [x] S21.1 Define the multiclass route: OvR may only be a compatibility layer/fallback, not the final default best implementation.
- [x] S21.2 Preserve sklearn/XGBoost-style public API: `fit`, `predict`, `predict_proba`, `decision_function`, `score`, `get_params`, `set_params`, and `GridSearchCV` must remain usable.
- [x] S21.3 Add native multiclass softmax objective specs: `num_class`, class encoding, base margin, softmax probability, gradient/Hessian, sample weights, and numerical stability rules must be explicit.
- [x] S21.4 Implement native softmax multiclass training in the CPU oracle; OvR tests must not pretend native softmax is complete.
- [x] S21.5 Extend the native model format to save `num_class`, multiclass objective, class mapping, and multi-class tree/update structures while preserving backward-compatible reads of old model formats.
- [x] S21.6 Implement Python classifier `multi_strategy="auto" | "softmax" | "ovr"`, where `auto` defaults to softmax when native softmax is available.
- [x] S21.7 Implement native softmax `predict_proba`, with every probability row normalized, finite, and consistent with `predict` argmax class.
- [x] S21.8 Add the MPS native softmax path or define the phased gate clearly; before MPS is complete, CPU softmax must not be reported as MPS softmax.
- [x] S21.9 Cover real multiclass datasets such as Iris, Digits, and Covertype subset, and verify CPU oracle, MPS behavior, and sklearn model-selection compatibility.
- [x] S21.10 Accept G19: the default multiclass implementation reaches native softmax, with OvR retained only as explicit fallback/compatibility strategy.

## S22: Cross-Platform Portable Backend and Unified Entry

- [x] S22.1 Design portable backend policy: MPSBoost native CPU/MPS backends remain the default and correctness oracle; Apple Silicon prioritizes native, Linux CUDA may choose XGBoost GPU, general CPU may choose native CPU or sklearn/XGBoost CPU, and summaries must expose the actual backend.
- [x] S22.2 Add optional dependency extras: `mpsboost[xgboost]`, `mpsboost[sklearn]`, `mpsboost[cuda]`; default installs remain lightweight and do not force heavy dependencies.
- [x] S22.3 Implement the unified estimator adapter while preserving `fit`, `predict`, `predict_proba`, `score`, `get_params`, `set_params`, and model-selection behavior.
- [x] S22.4 Implement environment diagnostics and installation guidance: missing CUDA/XGBoost/sklearn cases must provide copy-paste installation commands, avoid interactive `input()`, and support an environment variable to skip diagnostics.
- [x] S22.5 Define the boundary: external backends must be explicit portable mode or an observable `device="auto"` selection, must not replace the native CPU oracle, and summaries must report the actual backend and strategy.
- [x] S22.6 Cover macOS MPS, macOS CPU, Linux CPU, and Linux CUDA smoke matrices through native macOS tests plus explicit external-backend adapter policy, optional dependency diagnostics, and backend-summary assertions. Linux/CUDA runtime failures are attributed to the selected external sklearn/XGBoost/CUDA stack, not to native MPSBoost CPU/MPS.
- [x] S22.7 Accept G20: the same user interface can route Apple Silicon to native CPU/MPS and ordinary Linux or CUDA Linux to explicit external backends, with dependencies, performance expectations, and actual backend transparent to users.

## S23: Documentation Site Translation and Internationalization

- [x] S23.1 Inventory every Markdown source file that must enter the documentation site. Project docs and project specs must be translated in place next to their source files, for example `README.md` beside `README.zh-Hans.md`, and `specs/tasks.md` beside `specs/tasks.zh-Hans.md`.
- [x] S23.2 Docs-site PRD source files must stay in `docs-site/specs/`, with English `*.md` and Simplified Chinese `*.zh-Hans.md` side by side; these files must not be moved to root `specs/`.
- [x] S23.3 `docs-site/docs/en/` and `docs-site/docs/zh-Hans/` are only language navigation trees; when a source file already exists elsewhere in the project, the docs page must symlink to that source and must not duplicate Markdown under `docs-site/docs/`.
- [x] S23.4 Docs-site PRD navigation directories use `docs-site/docs/en/docs-site-prd/` and `docs-site/docs/zh-Hans/docs-site-prd/`, both symlinking back to `docs-site/specs/`.
- [x] S23.5 Translate README, `docs/`, `ai-skills/`, core specs, docs-site PRDs, benchmark docs, test docs, and user-guide pages. Translation must happen in the owning source directory, not by maintaining another duplicated translation copy under `docs-site/`.
- [x] S23.6 Add and maintain MkDocs i18n configuration with parallel `en/` and `zh-Hans/` navigation. English navigation must not point to Chinese filenames, Chinese navigation must not point to English-only content, and untranslated pages must be marked explicitly rather than silently mixed.
- [x] S23.7 Validate consistency of links, terminology, version numbers, backend policy, PyPI installation commands, and environment diagnostic commands across English and Chinese pages.
- [x] S23.8 Translation discipline: do not shorten, summarize, delete paragraphs, merge points, simplify warnings, remove limitation notes, or replace concrete commands with generic prose; English and Chinese pages must preserve section structure, information amount, code blocks, commands, constraints, limitations, and acceptance criteria.
- [x] S23.9 Accept G21: the docs site has a maintainable bilingual matrix, every existing Markdown source has an explicit bilingual path, all symlinks are valid, MkDocs strict build passes, and translation work preserves append-only release history from `0.1.0a0` through `0.4.0`.
