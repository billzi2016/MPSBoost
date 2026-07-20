# Project Constitution

## Article I: Specifications Above All

This project uses SDD. Specifications define product facts, architectural boundaries,
data semantics, and acceptance gates. Implementation must comply with specifications,
tests must validate specifications, and releases must satisfy specifications. Existing
code, time pressure, or implementation convenience must not be used to justify
violating specifications.

## Article II: Real Implementation

The project does not accept mock training, placeholder success, fabricated device
availability, fabricated performance data, or silent CPU fallback. Incomplete
capabilities must be unavailable and remain unchecked in the task list, but the
official feature list must not present them as available.

## Article III: Single Source of Logic

Each business semantic may have only one authoritative implementation. Public
parameter validation, split gain, leaf weights, model formats, and cache keys must
not be copied across modules. CPU references and MPS backends may have different
computational implementations, but they must share the same specification, input
structure, and result contract.

## Article IV: SOLID and DRY

- The Python API is responsible only for the user contract and input adaptation.
- The data module is responsible only for validation, quantization, and data ownership.
- The training core is responsible only for the algorithm state machine and backend orchestration.
- Backends are responsible only for computation and device resources and do not decide product parameter semantics.
- Caching does not participate in correctness; results must remain equivalent after cache deletion.
- Model I/O does not depend on the lifetime of training-time objects.

The dependency direction is fixed as:
`Python API → application services → domain core ← backend interface ← MPS backend`.
The domain core must not depend in reverse on Python, Objective-C objects, or the
file system.

## Article V: Correctness First

Every optimization requires a CPU oracle, boundary tests, and reproducible experiments
beforehand. Floating-point nondeterminism must be constrained by tolerances and
stable tie-breaking. When an error, overflow, device failure, or corrupted cache is
detected, fail explicitly rather than return a partial model.

## Article VI: Efficiency Is an End-to-End Requirement

Performance measurement must include input conversion, binning, device
initialization, kernels, synchronization, and model assembly. Optimization goals,
in order, are to avoid copying, improve effective parallelism, reduce atomic
contention, reduce synchronization, and reuse memory. A single fast kernel with slow
end-to-end performance is not a success.

## Article VII: Stable and Easy to Install

Formal accelerated releases must provide prebuilt wheels for supported platforms.
Ordinary users must not need heavy frameworks, package managers, build systems, or
local shader compilers to install. Unavailable devices, missing resources, and ABI
mismatches must report clear errors before training.

## Article VIII: Chinese Maintainability

Code files must include Chinese intent descriptions; public and complex functions must
include Chinese contract comments; parallelism, memory, caching, and numerical key
points must explain their design rationale. A mismatch between comments and code is
a defect.

## Article IX: Tests Must Not Be Manipulated

It is forbidden to delete failing tests, weaken assertions, widen unjustified
tolerances, conditionally skip real errors, or test only mocks. Fixes must address
root causes. Environment blockers must be recorded as blockers and tasks must not be
marked complete.

## Article X: Truthful Task Completion

`[x]` in `tasks.md` is a declaration of completion. It may be checked only when
implementation, Chinese comments, tests, documentation, and acceptance are all
satisfied. Partial completion, interface-only work, passing only unit tests, or
verification not yet run must remain `[ ]`.

## Article XI: Minimum Public Commitment

Public materials describe only delivered capabilities. Prereleases may reserve names,
but must not include fake training interfaces or lead users to believe functionality
is available. Public versions cannot be overwritten, and releases must be traceable
to verified commit and artifact hashes.

## Article XII: Constitutional Amendments

Changing the constitution requires explicit user approval and records the reason for
the change, affected modules, and migration plan. No ordinary implementation task may
implicitly amend the constitution.
