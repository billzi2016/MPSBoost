// MPSBoost 平方误差 gradient/Hessian kernel。
//
// 职责：把 float32 标签与当前预测映射为 float2(g, h)。目标函数语义由 C++ 领域层
// 冻结，本文件只执行同一公式；不得加入损失选择、参数默认值或 CPU 回退。

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
