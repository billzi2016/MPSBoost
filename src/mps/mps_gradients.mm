// MPSBoost Metal gradient and smoke-kernel execution.
//
// Intent: isolate simple vector kernels from tree-specific histogram and split
// code so backend lifetime and numerical kernels remain independently readable.

#include "mps_backend_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace mpsboost {

std::vector<GradientPair> MpsBackend::ComputeSquaredError(
    const std::vector<double>& labels,
    const std::vector<double>& predictions) const {
  if (labels.empty() || labels.size() != predictions.size()) {
    throw TrainingError(labels.empty() ? "标签不能为空" : "标签与预测长度不一致");
  }
  const std::uint32_t count = CheckedUInt32(labels.size(), "Gradient 样本数量");
  std::vector<float> labels_f(labels.size());
  std::vector<float> predictions_f(predictions.size());
  for (std::size_t index = 0; index < labels.size(); ++index) {
    labels_f[index] = CheckedFloat(labels[index], "标签");
    predictions_f[index] = CheckedFloat(predictions[index], "预测");
  }
  const std::size_t scalar_bytes = CheckedBytes(labels.size(), sizeof(float), "Gradient 输入");
  const std::size_t output_bytes =
      CheckedBytes(labels.size(), sizeof(float) * 2, "Gradient 输出");

  @autoreleasepool {
    id<MTLBuffer> labels_buffer = impl_->NewBuffer(labels_f.data(), scalar_bytes, "labels");
    id<MTLBuffer> predictions_buffer =
        impl_->NewBuffer(predictions_f.data(), scalar_bytes, "predictions");
    id<MTLBuffer> output_buffer = impl_->NewBuffer(nullptr, output_bytes, "gradients");
    id<MTLCommandBuffer> command = impl_->NewCommand("gradient");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) {
      throw BackendError("创建 gradient encoder 失败");
    }
    [encoder setComputePipelineState:impl_->gradients_];
    [encoder setBuffer:labels_buffer offset:0 atIndex:0];
    [encoder setBuffer:predictions_buffer offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBytes:&count length:sizeof(count) atIndex:3];
    const NSUInteger width = std::min<NSUInteger>(
        kThreadsPerGroup, [impl_->gradients_ maxTotalThreadsPerThreadgroup]);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    const auto gradient_started = std::chrono::steady_clock::now();
    Impl::Complete(command, "MPS gradient command 执行失败");
    impl_->timing_.gradient_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      gradient_started)
            .count();

    const auto* values = static_cast<const float*>([output_buffer contents]);
    std::vector<GradientPair> result(labels.size());
    for (std::size_t index = 0; index < labels.size(); ++index) {
      result[index] = GradientPair{values[index * 2], values[index * 2 + 1]};
    }
    return result;
  }
}

std::vector<float> MpsBackend::RunVectorAddForTest(
    const std::vector<float>& left,
    const std::vector<float>& right) const {
  if (left.size() != right.size()) {
    throw BackendError("GPU smoke 输入长度不一致");
  }
  if (left.empty()) {
    return {};
  }
  const std::uint32_t count = CheckedUInt32(left.size(), "GPU smoke 元素数量");
  const std::size_t bytes = CheckedBytes(left.size(), sizeof(float), "GPU smoke 输入");
  @autoreleasepool {
    id<MTLBuffer> left_buffer = impl_->NewBuffer(left.data(), bytes, "smoke left");
    id<MTLBuffer> right_buffer = impl_->NewBuffer(right.data(), bytes, "smoke right");
    id<MTLBuffer> output_buffer = impl_->NewBuffer(nullptr, bytes, "smoke output");
    id<MTLCommandBuffer> command = impl_->NewCommand("vector_add");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) {
      throw BackendError("创建 vector_add encoder 失败");
    }
    [encoder setComputePipelineState:impl_->vector_add_];
    [encoder setBuffer:left_buffer offset:0 atIndex:0];
    [encoder setBuffer:right_buffer offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBytes:&count length:sizeof(count) atIndex:3];
    const NSUInteger width = std::min<NSUInteger>(
        kThreadsPerGroup, [impl_->vector_add_ maxTotalThreadsPerThreadgroup]);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    Impl::Complete(command, "MPS vector_add command 执行失败");
    std::vector<float> output(left.size());
    std::memcpy(output.data(), [output_buffer contents], bytes);
    return output;
  }
}


}  // namespace mpsboost
