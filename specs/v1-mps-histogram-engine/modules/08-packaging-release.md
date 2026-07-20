# Module Design: Build, Packaging, and Release

## 1. Build Target
Use `scikit-build-core`, CMake, Objective-C++, and build-time Metal shader
compilation to produce arm64 macOS wheels. Versioning has one authority and is
injected into Python, native code, and shader ABI.

## 2. Wheel Contents
Include only:
- public Python modules;
- one required native extension;
- matching `.metallib`;
- package metadata and license.

Do not include specifications, AGENTS, tests, benchmarks, source intermediates,
caches, debug logs, credentials, SDK files, or absolute build-machine paths.

## 3. Size
Release jobs record compressed/uncompressed wheel size, largest files, and dynamic
dependencies. Release builds handle debug symbols while retaining version information
needed for crash diagnostics. New dependencies must state their size and benefit.

## 4. CI
CI uses least-privilege `contents: read`. The test matrix covers supported Python
versions and supported macOS runners. Release workflows are separate from CI; only
version tags and protected environments receive release permission.

Prefer PyPI Trusted Publishing. If unavailable, tokens have minimum project scope,
exist only as secrets, and are never printed. Do not repeat interactive login or write
credentials to configuration files.

## 5. Git
- `specs/` and `specs/AGENTS.md` are project-rule assets in source; caches,
  `dist/`, and local build artifacts remain ignored;
- each cohesive module has an independent Chinese commit;
- commits total at most 10 lines and state result, rationale, and verification;
- before push, review status, diff, ignore rules, sensitive data, and test results;
- do not commit unverified artifacts.

## 6. Release Order
1. Complete every release task in reality;
2. build one artifact from a clean commit;
3. run metadata, content, link, size, and installation tests on that artifact;
4. record SHA-256;
5. validate release installation in an isolated environment;
6. obtain user confirmation of package name, version, and hash;
7. upload the same artifact to formal PyPI;
8. install from PyPI in a new environment and run real smoke `fit/predict`;
9. publish English release notes.

Stop on any failed step; do not reupload the same version or skip verification.

## 7. Rules for 0.2.0
Version 0.2.0 must contain real training capability. Do not release a placeholder
package containing only a namespace, fake API, or `NotImplementedError` training.
When only reserving a name, use an explicit prerelease and completely honest metadata,
but still require separate user confirmation before final upload.

## 8. Acceptance
- Installation completes with one pip command;
- no administrator privilege, network runtime dependency, or local compilation;
- wheel size and dependency audits pass;
- real training and prediction pass in a clean environment;
- artifact and test hashes agree exactly;
- GitHub and PyPI pages claim only implemented capabilities.

## 9. Large-Module Delivery and PyPI Milestones
Every accepted large module forms a GitHub module commit/push and has CI preserve a
verified wheel artifact. Do not upload internal implementation modules to PyPI
separately, avoiding versions users cannot directly use.

Fixed PyPI milestones are: release `0.2.0a0` for the first real S5 estimator;
`0.2.0b0` for S6 performance acceptance; `0.2.0rc0` for S7 stability
acceptance; and `0.2.0` for complete release gates. Every PyPI artifact originates
from the matching GitHub source state; audit hash, content, installation, and real
tests before upload, then retest from a clean formal-PyPI installation afterward.
