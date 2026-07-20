# Project Directory Specification

The following is the target canonical directory structure. Do not add parallel
implementation directories without a specification change.

```text
MPSBoost/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── release.yml
├── mps_boost_skill.md           # Complete AI/Agent skill entry
├── specs/                       # SDD specifications and project development rules
│   ├── AGENTS.md                # Project Agent development rules; not an ordinary user entry
│   ├── README.md                # Shared specification index
│   ├── constitution.md          # Shared engineering constitution
│   ├── project-tree.md          # Shared directory structure
│   ├── tasks.md                 # Shared master task list
│   ├── v2-arboretum-implementation/ # v2 all-tree model roadmap
│   │   └── prd.md
│   ├── v3-real-world-tests/     # v3 real-world dataset tests and 1.x release gate
│   │   └── prd.md
│   └── v1-mps-histogram-engine/
│       ├── prd.md
│       └── modules/
│           ├── 01-python-api.md
│           ├── 02-data-quantization.md
│           ├── 03-training-core.md
│           ├── 04-mps-backend.md
│           ├── 05-cache.md
│           ├── 06-model-io.md
│           ├── 07-quality.md
│           └── 08-packaging-release.md
├── include/mpsboost/
│   ├── backend.hpp              # Compute backend abstraction
│   ├── binned_dataset.hpp       # Quantized-data read-only view
│   ├── objective.hpp            # Objective function interface
│   ├── tree.hpp                 # Stable domain model
│   ├── trainer.hpp              # Training state machine
│   └── version.hpp
├── src/
│   ├── core/
│   │   ├── binned_dataset.cpp
│   │   ├── objective.cpp
│   │   ├── tree.cpp
│   │   └── trainer.cpp
│   ├── cpu/
│   │   └── reference_backend.cpp
│   ├── mps/
│   │   ├── mps_backend.mm
│   │   ├── metal_context.mm
│   │   ├── buffer_pool.mm
│   │   └── kernels/
│   │       ├── gradients.metal
│   │       ├── histogram.metal
│   │       ├── split_scan.metal
│   │       ├── partition.metal
│   │       └── prediction.metal
│   ├── io/
│   │   └── model_format.cpp
│   ├── python/
│   │   └── bindings.cpp
│   └── mpsboost/
│       ├── __init__.py
│       ├── estimator.py
│       ├── matrix.py
│       ├── booster.py
│       ├── cache.py
│       └── diagnostics.py
├── tests/
│   ├── unit/
│   ├── metal/
│   ├── integration/
│   ├── packaging/
│   └── benchmarks/
├── benchmarks/
│   ├── datasets.py
│   ├── runner.py
│   ├── report.py
│   └── results/                 # Verified raw benchmarks and readable summaries; excluded from wheels
├── CMakeLists.txt
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── LICENSE
└── .gitignore
```

## Directory Responsibility Constraints

- `include/mpsboost`: stable C++ contracts, with no Objective-C types.
- `src/core`: device-independent domain logic, with no Python, Metal, or file-system access.
- `src/cpu`: the sole CPU oracle, serving only correctness and explicit CPU mode.
- `src/mps`: device resources and kernels, and must not redefine training mathematical semantics.
- `src/io`: model format, independent of the training session.
- `src/python`: thin bindings, without duplicating Python-layer parameter logic.
- `src/mpsboost`: user experience, parameter validation, and exception translation, without hot-path implementation.
- `tests`: layered by real boundaries; shared test-data generators and no duplicate expected algorithms.
- `benchmarks`: separate from tests and must not participate in package runtime.
