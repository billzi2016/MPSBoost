# Project Constitution

## Article 1: Specifications First

This project uses SDD. Specifications define product facts, architecture
boundaries, data semantics, and acceptance gates. Implementations must conform to
specifications, tests must verify specifications, and releases must satisfy
specifications. Existing code, schedule pressure, or implementation convenience
must never justify violating a specification.

## Article 2: Real Implementations

The project does not accept mock training, placeholder success, fabricated device
availability, fabricated performance data, or silent CPU fallback. Incomplete
capability must be unavailable and remain unchecked in the task list, while formal
release feature lists must not present it as available.

## Article 3: Single Logic

One business semantic may have only one authoritative implementation. Public
parameter validation, split gain, leaf weight, model format, and cache keys must
not be copied across modules. CPU reference and MPS backends may differ in their
computational implementation, but must share the same specification, input
structures, and result contract.

## Article 4: SOLID and DRY

- Python API owns only user contracts and input adaptation.
- The data module owns only validation, quantization, and data ownership.
- The training core owns only the algorithm state machine and backend orchestration.
- Backends own only computation and device resources, and do not decide product parameter semantics.
- Cache does not participate in correctness; results must remain equivalent after cache deletion.
- Model I/O does not depend on training-time object lifetime.

The dependency direction is fixed as: `Python API -> application service -> domain core <- backend interface <- MPS backend`. Domain core must not depend in reverse on Python, Objective-C objects, or the file system.

## Article 5: Correctness First

Every optimization requires a CPU oracle, boundary tests, and reproducible
experiments first. Floating-point nondeterminism must be constrained by tolerance
and stable tie-breaks. Detected errors, overflow, device failures, and corrupt
caches must fail explicitly and must not return partial models.

## Article 6: Efficiency Is an End-to-End Requirement

Performance measurement must include input conversion, binning, device
initialization, kernels, synchronization, and model assembly. Optimization targets,
in order, are avoiding copies, improving effective parallelism, reducing atomic
contention, reducing synchronization, and reusing memory. A single fast kernel with
slow end-to-end execution is not success.

## Article 7: Stability and Easy Installation

Formal accelerated releases must provide prebuilt wheels for supported platforms.
Ordinary users must not need heavyweight frameworks, package managers, build
systems, or local shader compilers to install. Unavailable devices, missing
resources, and ABI mismatch must report explicit errors before training.

## Article 8: Maintainability of Documentation Language

Code files must contain intent documentation; public and complex functions must
have contract comments; parallelism, memory, cache, and numerical critical points
must explain design reasons. Inconsistency between comments and code is a defect.

## Article 9: Tests Must Not Be Manipulated

Deleting failing tests, lowering assertions, widening unjustified tolerance,
conditionally skipping real errors, or testing only mocks is prohibited. Fixes must
address root causes. Environment blocks must be recorded as blocked and must not
mark tasks complete.

## Article 10: Truthful Task Completion

`[x]` in `tasks.md` is a completion declaration. It may be checked only after
implementation, comments, tests, documentation, and acceptance are all satisfied.
Partial completion, interface-only work, unit-tests-only success, or unrun
verification must remain `[ ]`.

## Article 11: Minimum Public Commitment

External material describes only delivered capability. Pre-releases may reserve a
name, but must not contain fake training interfaces or lead users to believe a
feature is usable. Public versions cannot be overwritten; release artifacts must be
traceable to validated commits and artifact hashes.

## Article 12: Amendments

Changing this constitution requires explicit user approval and records the reason,
affected modules, and migration plan. Ordinary implementation tasks have no right to
amend the constitution implicitly.
