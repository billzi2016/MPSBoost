# Project Directory Specification

The following is the sole target directory structure. Do not add parallel
implementation directories without a specification change.

```text
MPSBoost/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── release.yml
├── ai-skills/
│   └── mps_boost_skill.md       # Complete skill entry point for AI/Agents
├── specs/                       # SDD specifications and project development rules
│   ├── AGENTS.md                # Project Agent development rules; not a general user entry point
│   ├── README.md                # Shared specification index
│   ├── constitution.md          # Shared engineering constitution
│   ├── project-tree.md          # Shared directory structure
│   ├── tasks.md                 # Shared master task list
│   ├── v2-arboretum-implementation/ # v2 full tree-model roadmap
│   │   └── prd.md
│   ├── v3-real-world-tests/     # v3 real-world dataset tests and 1.x release gates
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
│   ├── backend.hpp              # Compute-backend abstraction
│   ├── binned_dataset.hpp       # Read-only view of quantized data
│   ├── objective.hpp            # Objective-function interface
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
├── docs/
│   ├── CHANGELOG.md
│   └── RELEASE_AUDIT_*.md
├── LICENSE
└── .gitignore
```

## Directory Responsibility Constraints

- `include/mpsboost`: stable C++ contracts; contains no Objective-C types.
- `src/core`: device-independent domain logic; does not access Python, Metal, or the file system.
- `src/cpu`: the sole CPU oracle; serves only correctness and explicit CPU mode.
- `src/mps`: device resources and kernels; must not redefine training mathematical semantics.
- `src/io`: model format; does not depend on training sessions.
- `src/python`: thin bindings; do not duplicate Python-layer parameter logic.
- `src/mpsboost`: user experience, parameter validation, and exception conversion; does not implement hot paths.
- `tests`: layered by real boundaries; share test-data generators and do not duplicate expected algorithms.
- `benchmarks`: separate from tests and must not participate in package runtime.
