# Module Design: Layered Cache

## 1. Principle
Caches may improve performance only; they must not be a correctness prerequisite.
When a cache is deleted, corrupted, or unwritable, the system safely rebuilds it or
runs without caching and produces equivalent output.

## 2. L1 Process Cache
Contents: device objects, pipelines, read-only shader libraries, buffer pools, and
shape dispatch plans.
- Lifetime does not exceed the process;
- thread-safe;
- mutable training data is not shared across estimators;
- has explicit capacity limits and release policy;
- tests support isolated instances and must not depend on hidden global state.

## 3. L2 User Cache
The default root follows macOS user-cache conventions and permits explicit override
through `MPSBOOST_CACHE_DIR`. It contains:
- `pipelines/`: rebuildable binary archives;
- `quantization/`: user-explicitly enabled data-binning cache;
- `tuning/`: device-specific tuning results.

Keys contain at least cache-format version, package version, device registry/family,
system version, relevant parameters, and required data fingerprints. Cache files are
written to temporary files, validated, then atomically replaced.

## 4. L3 Build Cache
Contains compilation objects, CI wheel cache, and test-data downloads. It affects
development only and enters neither runtime APIs nor wheels. Keys cover source hash,
toolchain, SDK, Python ABI, and target architecture.

## 5. Permissions and Privacy
- Import, version queries, and path queries create no directories;
- create minimum-permission directories only on the first real write;
- do not cache labels, original paths, or credentials;
- do not access the network;
- cleanup operates only on parsed and verified MPSBoost-specific directories and
  rejects the root directory, home directory, and symlink escapes.

## 6. Single Interface
One cache service provides path planning, keys, reads, writes, validation, and cleanup.
Other modules must not assemble paths or define versions themselves. Training modules
request semantic objects only and do not read files.

## 7. Acceptance
- Cold, process-warm, and disk-warm paths return equivalent results;
- truncation, validation failure, old versions, and missing permissions are safe;
- concurrent writes to one key produce no partial file;
- cleanup APIs never leave the project cache root;
- wheels and public repositories include no cache.
