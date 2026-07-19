// MPSBoost 真实 GPU 链路验证 kernel。
//
// 该 kernel 只验证 shader 编译、资源打包、pipeline、共享内存和 command 同步，不承担
// 训练功能。它不能被用作直方图实现的替代品，也不能作为训练任务完成的证据。
#include <metal_stdlib>

using namespace metal;

// 对等长向量执行逐元素相加。
//
// left/right/output 由 host 保证至少包含 count 个 float。gid 仍检查边界，因为 grid
// 调度与输入长度属于跨语言契约，shader 必须独立防止最后一个线程组越界。
kernel void vector_add(device const float* left [[buffer(0)]],
                       device const float* right [[buffer(1)]],
                       device float* output [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
  output[gid] = left[gid] + right[gid];
}
