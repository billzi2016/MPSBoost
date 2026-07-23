# MPSBoost Agent Working Rules

This file is located in `specs/` and is a project development-rule asset, not a
general user entry point. For user and AI entry points, see the root `README.md`
and `ai-skills/mps_boost_skill.md`.

## 1. Highest Priority

This project uses specification-driven development (SDD). `specs/constitution.md`
is the project constitution, `specs/prd.md` is the source of product facts, module
designs define implementation boundaries, and `specs/tasks.zh-Hans.md` defines
execution order. Code, tests, builds, and releases must not violate specifications.

`specs/tasks.zh-Hans.md` is the authoritative source for completion status. The
English `specs/tasks.md` must remain a faithful in-place translation for the
documentation site; when task text or checkbox status changes, update both files
in the same patch so the bilingual matrix stays consistent.

Priority from highest to lowest is: the user's current explicit instruction, the
project constitution, approved PRDs, module designs, the task list, and code
implementation. When a conflict is found, implementation must stop; explain the
conflict and correct the specification first. Do not choose a more convenient
interpretation independently.

## 2. SDD Workflow

Every module must proceed in this order:

1. Read the constitution, PRD, project tree, and relevant module designs.
2. Confirm task dependencies, inputs and outputs, error semantics, and acceptance criteria.
3. Before acting, describe files to add, modify, and delete, plus their expected effects.
4. After obtaining user confirmation, complete one cohesive module with a patch.
5. Write real implementation and real tests according to the specification.
6. Run tests, static checks, and necessary benchmarks.
7. Change `[ ]` to `[x]` in `tasks.md` only when every acceptance condition is actually met.
8. Make one commit per cohesive module to enable review and rollback.

It is forbidden to write code first and then alter specifications retroactively to
justify that code.

### 2.1 Batch Implementation and Layered Verification

- When consecutive modules have clear dependencies, unambiguous responsibility
  boundaries, and failures can be accurately located by module and dependency layer,
  several cohesive modules may be completed before a single batch-end execution of
  costly comprehensive verification.
- Batch implementation is not mixed implementation. Each module must retain
  independent file responsibility, unique business logic, a clear diff, and an
  independently reviewable input/output contract. Do not combine several semantics
  into an indistinguishable shared function or temporary entry point.
- After each module, run at least a proportionate quick static check, compilation
  check, or focused test to discover interface and dependency errors early. Expensive
  checks such as clean wheels, full regression, real Metal, installation
  revalidation, and remote version matrices may be run once after several
  non-interfering modules are complete.
- If a batch test fails, locate the root cause using module boundaries, call chains,
  and a minimal reproduction, then patch the corresponding module. Do not blindly
  modify code in a mixed-module state, add a second implementation, or hide
  responsibility boundaries with test special cases.
- Before all required batch-end tests genuinely pass, related tasks must not be
  checked, committed, pushed, released, or claimed complete. Batch verification
  must not be used to skip tests, use mocks, weaken assertions, or retain temporary
  bypasses.
- After a user approves a consecutive batch, first complete all code, comments, and
  test files in that batch, then enter consolidated verification. Do not repeatedly
  run the entire build, full test suite, or benchmark because an individual file
  finished during implementation.
- Verification layers that passed and are unaffected by later changes must not be
  repeated mechanically. A defect fix runs only focused checks covering its root
  cause and direct dependencies; after semantic implementation is complete, run one
  complete local batch-end verification, then one CI run after push for final
  cross-version and real-device validation. Formatting, comments, or documentation
  finishing must not trigger another complete test run.
- If compilation, real tests, or a performance gate fails, rerun the corresponding
  failed layer after a root-cause patch. This is required revalidation and must not
  expand into unrelated full repetition; nor may failed revalidation be skipped to
  reduce the number of runs.
- Do not run an absurd number of repeated builds, tests, installations, or benchmarks
  merely for the appearance of being safer. When explicit evidence of success exists
  and its code, input contract, dependencies, and runtime environment remain
  unaffected, that result must be reused and treated as valid. The amount of
  verification must follow actual risk and change impact.
- Within one approved batch, safely combinable reads, static checks, focused tests,
  and acceptance commands must be combined into one parallel or sequential
  execution and reported together. After the user confirms the file list and batch
  scope, ordinary patches and verification within that scope need no repeated
  per-file or per-command confirmation. Request authorization again only when
  expanding the scope, installing a new tool, performing a destructive action,
  publishing an external version, or requiring a new user decision.

## 3. File Modification Rules

- All added and modified files must use patches.
- Before modification, describe the file list, rationale, and expected effect, then
  wait for explicit confirmation.
- Normally use incremental patches. When more than 90% of a file must change, it may
  be deleted and rebuilt with a patch, but the reason must be stated in advance.
- Do not delete, overwrite, or revert user files without authorization.
- Do not create duplicate implementations, temporary bypasses, or a second business
  logic path for speed.

## 4. Implementation Quality

- Follow SOLID and DRY.
- Single responsibility: APIs, algorithms, device runtime, caching, serialization,
  and builds must not intrude upon one another.
- Dependency inversion: the training core depends on backend interfaces, not concrete
  device objects.
- Interface segregation: modules expose only the minimum interfaces that callers need.
- Open-closed principle: add objectives or backends through extension points, without
  modifying unrelated modules.
- Extract duplicate logic into one implementation; CPU and MPS mathematical semantics
  share the same specification and data structures.
- Do not add mocks, placeholder-success paths, fake performance data, or silent fallback.
- Do not make an incorrect implementation pass by modifying, skipping, weakening, or
  special-casing tests.
- Do not leave TODOs without an owner and task number. When deferral is necessary,
  enter it in the task list first and state its acceptance criteria.
- New or refactored code files should normally not exceed 200 lines; more than 300
  lines is a hard warning requiring a split. When above the target, split by
  responsibility into a package directory or several modules first. When native
  bindings, format declarations, or tightly coupled interfaces temporarily require
  more, explain why and schedule the split in the same development phase; do not
  continue adding functionality to long files.
- When adding functionality to an existing long file, extract stable boundaries into
  a new module before connecting the original entry point. Do not enlarge a giant
  file for convenience. File headers, class comments, function comments, and key
  implementation comments in split files must be in English.

## 5. Five Product Quality Principles

MPSBoost must aim to be the best deliverable product. The following five are hard
constraints; none may be sacrificed without evidence for another.

### 5.1 Fast

- Optimize end-to-end training and prediction, not only individual kernels.
- Performance measurement must include input conversion, binning, device
  initialization, synchronization, and cold-cache startup.
- Use unified memory, compact bins, batch scheduling, local histograms, and buffer
  reuse to reduce wasted overhead.
- Every performance optimization requires a real benchmark and correctness comparison;
  an optimization without evidence must not merge.
- When small data is unsuitable for a GPU, clearly state the applicability boundary;
  do not fabricate a general acceleration claim.

### 5.2 Easy to Install

- Supported platforms must provide prebuilt wheels, and the only default installation
  command is `python -m pip install mpsboost`.
- Ordinary users must not be required to install heavy frameworks, a system package
  manager, CMake, a compiler, or a shader toolchain.
- Wheels must include matching native extensions and shader resources and be validated
  on a clean machine.
- Installation failures must state the platform, version, or architecture cause;
  obscure dynamic-linking errors must not be exposed directly to users.

### 5.3 Easy to Use

- Public entry points retain an estimator style; reasonable defaults let common tasks
  run without low-level configuration.
- Users see only `device="mps"` and need not understand the division of work between
  internal MPS primitives and Metal kernels.
- Unknown or conflicting parameters fail early; silent ignoring is forbidden.
- Logging is concise by default. Diagnostic mode provides device, cache, and phase
  timings but must not expose sensitive information.

### 5.4 Small Footprint

- Do not introduce heavy runtime dependencies that serve only a small feature set.
- Native extensions, shaders, and the Python layer package only runtime requirements;
  tests, benchmarks, specifications, caches, and build artifacts must not enter wheels.
- Before release, record wheel size before and after extraction and audit the largest files.
- Eliminate duplicate resources; build specifications must define handling of debug
  symbols and release binaries.

### 5.5 Least Privilege

- Training and prediction must not require administrator privileges, system extensions,
  background services, network access, or additional entitlements.
- By default, read only data explicitly supplied by the user and write reconstructable
  caches only in the user cache directory.
- Importing the package, querying its version, and querying cache paths must not create
  files or access the network.
- Do not collect telemetry or upload data, models, device identifiers, or performance data.
- Release and CI credentials use least privilege and must never enter source, logs,
  wheels, or caches.

### 5.6 Acceptance Requirements

Every public release must provide and pass all of: an end-to-end performance report,
a clean-environment installation test, a minimal usage example, a wheel-size audit,
and a permission/network-behavior check. If any item fails, the release task must
not be marked complete.

## 6. Comment and Persisted-Language Rules

Chinese explanations in v0/v1 historical code and specifications may remain. Starting
with v2, new or substantially modified code comments, file-header descriptions,
public function documentation, key implementation descriptions, README files, release
notes, CI wording, and public error documentation must use English. Agent and user
conversation continues in Chinese.

Specifications are not required to be translated into English in place. The project
will later use MkDocs to establish a bilingual site; Chinese specifications may remain
as source or historical specifications. English versions must be maintained through
separate files, the site-generation process, or an explicit bilingual documentation
structure. Do not crudely translate all existing specifications in place and overwrite
them merely to meet the language rule.

Every code file must include a header comment stating at least the module intent,
responsibility boundary, key dependencies, and prohibitions.

Every public class, public function, and complex internal function must have a
documentation comment that states:

- its purpose and usage scenario;
- parameters, return values, and exceptions;
- ownership, thread safety, or side effects;
- key invariants and numerical semantics.

Complex control flow, long expressions, parallel reductions, cache invalidation,
memory synchronization, and non-obvious performance optimizations require key-point
comments explaining why; comments must not merely restate code mechanically. Simple
assignments do not need mechanical comments.

## 7. Test Rules

- Tests must validate real implementations, not only mocks.
- CPU references and device backends must use the same inputs and model semantics.
- Floating-point tolerances must have justification and must not be loosened
  arbitrarily to pass failed tests.
- Performance tests must include preprocessing, synchronization, and end-to-end time.
- A defect fix must preserve a reproducible test before fixing the implementation.
- When tests are not run or are blocked by the environment, state that truthfully and
  do not mark the work complete.

## 8. Tooling and Environment-Blocker Rules

- Before work begins, check the compiler, SDK, build system, dependencies, real device,
  permissions, and test tools required by the current module.
- When any required tool is missing, immediately stop related implementation or
  verification and explain to the user in Chinese what is missing, why it is needed,
  the recommended installation method, and how to validate after installation.
- Tool installation is performed by the user or after explicit authorization; agents
  must not independently install system tools or dependencies or alter the global environment.
- Do not lower specifications, delete features, use mocks, fabricate device results,
  skip tests, or submit unverified implementations to bypass missing tools.
- Do not replace the formal solution required by the specification with temporary
  scripts, hard-coded output, fake backends, or a second low-quality implementation.
- While the environment is blocked, the corresponding task remains `[ ]`; check it
  only after the real toolchain is restored and all acceptance criteria are met.
- When several formal tool options exist, state their size, stability, maintenance,
  and license impact for user selection; do not default to the fastest but lower-quality option.
- When a tool command fails, preserve the original error summary and identify the root
  cause first. Do not hide it by disabling validation, ignoring exit codes, or escalating privileges.

## 9. Git and Commits

- Git operations require explicit user authorization.
- Historical v0/v1 cleanup commits may use Chinese; v2 and later commit titles and
  bodies must use English.
- One commit contains only one cohesive module.
- Use an imperative title that clearly describes the result; do not use vague text
  such as "update code".
- The body states rationale, key design, and verification result; the total is at most 10 lines.
- Before committing, check ignored files, sensitive information, build artifacts, and the diff.
- `specs/` and `specs/AGENTS.md` are project-rule assets and may be committed with
  explicit user request; credentials, caches, local build artifacts, and temporary
  validation environments must not be committed.

Recommended format:

```text
Add compact binned matrix validation

Validate uint8/uint16 bin ownership and feature bounds.
Keep CPU and MPS paths on the same matrix contract.
Validation: targeted unit tests passed.
```

## 10. Release Rules

- Release artifacts must originate from the same tested build artifact; do not rebuild
  while uploading.
- Once public, a version must not be overwritten; use a new version for fixes.
- Complete release gates in the task list before formal release.
- Public README files, package metadata, and release notes use English; v2 and later
  code comments and commits also use English.
- Do not exaggerate functionality or performance or describe unimplemented capability as available.
- When authentication is already available, do not log in again; credentials must not
  be printed or written to files.

### 10.1 Large-Module Delivery and Milestone Releases

- Every large module defined by the task list must form one cohesive commit, be pushed
  to GitHub, and have CI preserve an installable wheel artifact. v2 and later commits
  use English.
- Do not upload internal modules to PyPI frequently merely for a record; PyPI releases
  only complete milestones users can perceive and use.
- Publish `0.2.0a0` after the first real S5 estimator passes acceptance; `0.2.0b0`
  after S6 performance goals pass; `0.2.0rc0` after S7 stability passes; and
  `0.2.0` after complete release gates pass.
- PyPI versions follow PEP 440; published versions are never overwritten or reused.
  For fixes, increment the corresponding prerelease number or patch version.
- GitHub commits, CI artifacts, and PyPI artifacts must be traceable to the same source
  state; do not rebuild during upload.
- Before upload, report the exact version, filename, size, SHA-256, and test results;
  after upload, install from formal PyPI in a clean environment and run real acceptance tests.
