# Module Design: MPS Backend

## 1. Responsibility
Implement Apple GPU discovery, resource management, shader loading, kernel scheduling,
synchronization, and error conversion. Present a unified MPS backend to users;
internally, use suitable MPS optimization capabilities and custom Metal Compute kernels.

## 2. Least Privilege
The backend uses only public user-space GPU APIs. It needs no administrator privilege,
system extensions, background services, network, camera, file-access entitlements, or
telemetry. Package import creates no device; initialization occurs only in `fit()`
or explicit diagnostics.

## 3. Runtime Objects
- `MetalContext`: device, command queue, shader library, and capabilities;
- `PipelineRegistry`: caches pipelines by kernel, function constants, and ABI version;
- `BufferPool`: reuses temporary buffers managed by size class;
- `MPSBackend`: implements `ComputeBackend` and does not expose Objective-C objects to core.

Object ownership uses RAII wrappers. Before command completion, related buffers must
not return to pools or be destroyed.

## 4. Shader Release
- Compile `.metal → .air → .metallib` at build time;
- wheels include version-matched `.metallib`;
- ordinary user machines do not run shader compilers;
- validate native ABI, shader ABI, and package version;
- fail before training when a resource or function is missing.

## 5. Core Kernels
### Gradients
One thread handles one row; squared error outputs `float2(grad, hess)`. The host
checks length; kernels still guard grid boundaries.
### Histograms
The correctness baseline may use global atomics; production must use threadgroup-local
histograms and second-stage reduction to reduce hot contention. Freeze layout and tile
parameters using real benchmarks.
### Subsequent Hot Paths
Split scan, partition, histogram subtraction, and prediction update enter production
only after their tasks complete. Do not disguise a host fake implementation as a GPU
capability for absent kernels.

## 6. Memory
- Prefer shared storage for control data;
- freeze GPU-hot workspace after comparing shared/private storage;
- use `uint8` bins to reduce bandwidth;
- preallocate the maximum known workspace before training;
- clear only actual regions per level;
- record peak allocated size and estimate before OOM.

Unified memory is not free: synchronization, cache coherence, page residency, and
duplicate buffers still require profiling.

## 7. Scheduling and Synchronization
- Encode commands in level-wise batches, avoiding a submission for each node;
- synchronize CPU only for compact split results or phase boundaries;
- check status/error for every command buffer;
- enable validation in debug; retain boundaries and error context in release;
- do not fire-and-forget and then immediately release resources.

## 8. Capability
Validate arm64 macOS, a Metal device, required GPU-family/atomic capability, maximum
threadgroup memory, and shader resources. Do not infer capability from chip names
alone. Diagnostics return a non-sensitive capability summary.

## 9. Performance Gates
For every optimization, record data scale, chip, cold/warm cache, kernel and
end-to-end time, memory, and error. Enable by default only when at least two scales
are valid. Special-device parameters need a conservative general path and must enter
the L2 tuning-cache key.

## 10. Acceptance
- Run on real devices, not mock command queues;
- handle non-threadgroup multiples, skewed bins, maximum bins, and large gradients;
- command failures are reproducible and resources remain safe;
- repeated training leaks no buffers;
- wheel environments load shaders without development tools.
