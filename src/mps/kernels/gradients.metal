// MPSBoost squared-error gradient/Hessian kernel.
//
// Responsibility: maps float32 labels and current predictions to float2(g, h). The
// C++ domain layer owns objective semantics; this file only executes that formula.
// Do not add loss selection, parameter defaults, or CPU fallback here.

#include <metal_stdlib>

using namespace metal;

kernel void squared_error_gradients(
    device const float* labels [[buffer(0)]],
    device const float* predictions [[buffer(1)]],
    device float2* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count) {
    return;
  }
  output[index] = float2(predictions[index] - labels[index], 1.0F);
}
