# MPSBoost SDD Specification Index

This directory is the project's sole source of specifications. `constitution.md` is
the shared highest engineering constitution, `project-tree.md` is the shared
directory specification, and `tasks.md` is the shared master task list.
`v1-mps-histogram-engine/` contains the foundational specifications for the 0.2.0
MPS histogram engine, `v2-arboretum-implementation/` contains the subsequent full
tree-model roadmap, and `v3-real-world-tests/` contains the real-world dataset
acceptance gates before 1.x. Before implementation, read the constitution, the
relevant version specifications, and the task list. If specifications conflict, the
current user instruction and the constitution are the highest constraints; correct
the specifications before coding.

## Shared Specifications

| File | Purpose |
| --- | --- |
| [constitution.md](constitution.md) | Shared engineering constitution and quality gates that cannot be bypassed |
| [project-tree.md](project-tree.md) | Shared directory structure and file responsibilities |
| [tasks.md](tasks.md) | Implementation checklist with dependencies and acceptance criteria |
| [v2-arboretum-implementation/prd.md](v2-arboretum-implementation/prd.md) | v2 roadmap for full tree-model families and industrial tabular capabilities |
| [v3-real-world-tests/prd.md](v3-real-world-tests/prd.md) | v3 real-world dataset tests and 1.x release gates |

## V1 MPS Histogram Engine Core Specification

| File | Purpose |
| --- | --- |
| [v1-mps-histogram-engine/prd.md](v1-mps-histogram-engine/prd.md) | User problems, product scope, functional, and non-functional requirements |

## V1 MPS Histogram Engine Module Designs

| File | Module |
| --- | --- |
| [v1-mps-histogram-engine/modules/01-python-api.md](v1-mps-histogram-engine/modules/01-python-api.md) | Estimator-style Python API |
| [v1-mps-histogram-engine/modules/02-data-quantization.md](v1-mps-histogram-engine/modules/02-data-quantization.md) | Inputs, binning, and data ownership |
| [v1-mps-histogram-engine/modules/03-training-core.md](v1-mps-histogram-engine/modules/03-training-core.md) | Objectives, tree growth, and the training state machine |
| [v1-mps-histogram-engine/modules/04-mps-backend.md](v1-mps-histogram-engine/modules/04-mps-backend.md) | MPS/Metal runtime and kernels |
| [v1-mps-histogram-engine/modules/05-cache.md](v1-mps-histogram-engine/modules/05-cache.md) | Three-level caching and invalidation strategy |
| [v1-mps-histogram-engine/modules/06-model-io.md](v1-mps-histogram-engine/modules/06-model-io.md) | Model format, loading, and compatibility |
| [v1-mps-histogram-engine/modules/07-quality.md](v1-mps-histogram-engine/modules/07-quality.md) | Tests, correctness, benchmarks, and security |
| [v1-mps-histogram-engine/modules/08-packaging-release.md](v1-mps-histogram-engine/modules/08-packaging-release.md) | Wheels, CI, Git, and PyPI releases |

## Execution Rules

1. Do not implement without first following the specifications.
2. Do not maintain two equivalent logic paths or temporary mocks.
3. Check a task only after its code, comments, tests, and acceptance are all complete.
4. When a module design must change, update related specifications first and state the impact scope.
