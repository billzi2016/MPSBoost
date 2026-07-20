# MPSBoost SDD Specification Index

This directory is the sole source of project specifications. `constitution.md` is
the shared highest engineering constitution, `project-tree.md` is the shared
directory specification, and `tasks.md` is the shared master task list.
`v1-mps-histogram-engine/` contains the foundational specifications for the 0.2.0
MPS histogram engine, `v2-arboretum-implementation/` contains the subsequent
all-tree-model roadmap, and `v3-real-world-tests/` contains the real-world dataset
acceptance gate before the 1.x line. Before implementation, read the constitution,
the applicable version specification, and the task list. When specifications
conflict, the current user instruction and constitution are the highest constraints;
correct the specifications before coding.

## Shared Specifications

| File | Purpose |
| --- | --- |
| [constitution.md](constitution.md) | Non-bypassable shared engineering constitution and quality gates |
| [project-tree.md](project-tree.md) | Shared directory structure and file responsibilities |
| [tasks.md](tasks.md) | Implementation checklist with dependencies and acceptance criteria |
| [v2-arboretum-implementation/prd.md](v2-arboretum-implementation/prd.md) | v2 all-tree model-family and industrial tabular-capability roadmap |
| [v3-real-world-tests/prd.md](v3-real-world-tests/prd.md) | v3 real-world dataset tests and the 1.x release gate |

## V1 MPS Histogram Engine Core Specification

| File | Purpose |
| --- | --- |
| [v1-mps-histogram-engine/prd.md](v1-mps-histogram-engine/prd.md) | User problem, product scope, and functional/non-functional requirements |

## V1 MPS Histogram Engine Module Design

| File | Module |
| --- | --- |
| [v1-mps-histogram-engine/modules/01-python-api.md](v1-mps-histogram-engine/modules/01-python-api.md) | Estimator-style Python API |
| [v1-mps-histogram-engine/modules/02-data-quantization.md](v1-mps-histogram-engine/modules/02-data-quantization.md) | Input, binning, and data ownership |
| [v1-mps-histogram-engine/modules/03-training-core.md](v1-mps-histogram-engine/modules/03-training-core.md) | Objective functions, tree growth, and training state machine |
| [v1-mps-histogram-engine/modules/04-mps-backend.md](v1-mps-histogram-engine/modules/04-mps-backend.md) | MPS/Metal runtime and kernels |
| [v1-mps-histogram-engine/modules/05-cache.md](v1-mps-histogram-engine/modules/05-cache.md) | Three-level cache and invalidation strategy |
| [v1-mps-histogram-engine/modules/06-model-io.md](v1-mps-histogram-engine/modules/06-model-io.md) | Model format, loading, and compatibility |
| [v1-mps-histogram-engine/modules/07-quality.md](v1-mps-histogram-engine/modules/07-quality.md) | Tests, correctness, benchmarks, and safety |
| [v1-mps-histogram-engine/modules/08-packaging-release.md](v1-mps-histogram-engine/modules/08-packaging-release.md) | Wheel, CI, Git, and PyPI release |

## Execution Rules

1. Do not implement without first reading the specification.
2. Do not maintain two implementations of the same logic or temporary mocks.
3. Check a task only after code, comments, tests, documentation, and acceptance are all complete.
4. When module design must change, update the related specification first and explain the impact scope.
