// MPSBoost real GPU pipeline validation kernel.
//
// This kernel validates shader compilation, resource packaging, pipeline creation,
// shared memory, and command synchronization only; it has no training role. It must
// not replace a histogram implementation or be evidence that training is complete.
#include <metal_stdlib>

using namespace metal;

// Adds equally sized vectors element by element.
//
// The host guarantees that left, right, and output contain at least count floats.
// The grid and input length form a cross-language contract, so the shader must still
// independently prevent the final threadgroup from accessing out of bounds.
kernel void vector_add(device const float* left [[buffer(0)]],
                       device const float* right [[buffer(1)]],
                       device float* output [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
  output[gid] = left[gid] + right[gid];
}
