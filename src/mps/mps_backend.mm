// MPSBoost 真实 Metal 计算后端。
//
// 职责：统一持有 device/library/queue/pipeline，调度 gradient、两阶段 histogram 与内部
// smoke kernel，并把同步错误转换为 BackendError。树生长、split 和用户参数不得进入本文件。

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "mpsboost/backend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <utility>

namespace mpsboost {
namespace {

constexpr std::uint32_t kThreadsPerGroup = 256;
constexpr std::uint32_t kMaximumHistogramPartials = 16;

struct DeviceHistogramValue final {
  std::uint32_t count;
  float gradient;
  float hessian;
  std::uint32_t reserved;
};
static_assert(sizeof(DeviceHistogramValue) == 16);

std::string DescribeError(const char* stage, NSError* error) {
  std::ostringstream message;
  message << stage;
  if (error != nil) {
    message << ": " << [[error localizedDescription] UTF8String];
  }
  return message.str();
}

std::size_t CheckedBytes(std::size_t count,
                         std::size_t item_size,
                         const char* field) {
  if (item_size != 0 && count > std::numeric_limits<std::size_t>::max() / item_size) {
    throw BackendError(std::string(field) + "：字节数计算溢出");
  }
  return count * item_size;
}

std::size_t CheckedAddBytes(std::size_t left,
                            std::size_t right,
                            const char* field) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw BackendError(std::string(field) + "：工作区字节数溢出");
  }
  return left + right;
}

std::uint32_t CheckedUInt32(std::uint64_t value, const char* field) {
  if (value > std::numeric_limits<std::uint32_t>::max()) {
    throw BackendError(std::string(field) + " 超出当前 GPU ABI 的 uint32 范围");
  }
  return static_cast<std::uint32_t>(value);
}

float CheckedFloat(double value, const char* field) {
  if (!std::isfinite(value) ||
      value > static_cast<double>(std::numeric_limits<float>::max()) ||
      value < -static_cast<double>(std::numeric_limits<float>::max())) {
    throw BackendError(std::string(field) + " 无法安全转换为 GPU float32");
  }
  return static_cast<float>(value);
}

std::uint32_t ReductionWidth(id<MTLComputePipelineState> pipeline) {
  const NSUInteger limit =
      std::min<NSUInteger>(kThreadsPerGroup, [pipeline maxTotalThreadsPerThreadgroup]);
  std::uint32_t width = 1;
  while (static_cast<NSUInteger>(width * 2U) <= limit) {
    width *= 2U;
  }
  return width;
}

}  // namespace

class MpsBackend::Impl final {
 public:
  explicit Impl(const std::string& metallib_path) {
    @autoreleasepool {
      device_ = MTLCreateSystemDefaultDevice();
      if (device_ == nil) {
        throw BackendError("MPS 后端不可用：系统没有返回默认 Metal 设备");
      }
      NSString* path = [NSString stringWithUTF8String:metallib_path.c_str()];
      if (path == nil || ![[NSFileManager defaultManager] fileExistsAtPath:path]) {
        throw BackendError("MPS shader library 不存在或路径不是有效 UTF-8");
      }
      NSError* error = nil;
      library_ = [device_ newLibraryWithURL:[NSURL fileURLWithPath:path isDirectory:NO]
                                      error:&error];
      if (library_ == nil) {
        throw BackendError(DescribeError("加载 MPS shader library 失败", error));
      }
      queue_ = [device_ newCommandQueue];
      if (queue_ == nil) {
        throw BackendError("创建 MPS command queue 失败");
      }
      vector_add_ = MakePipeline(@"vector_add");
      gradients_ = MakePipeline(@"squared_error_gradients");
      histogram_u8_ = MakePipeline(@"histogram_partial_u8");
      histogram_u16_ = MakePipeline(@"histogram_partial_u16");
      histogram_baseline_u8_ = MakePipeline(@"histogram_baseline_u8");
      histogram_baseline_u16_ = MakePipeline(@"histogram_baseline_u16");
      histogram_reduce_ = MakePipeline(@"histogram_reduce");
    }
  }

  id<MTLComputePipelineState> MakePipeline(NSString* name) {
    id<MTLFunction> function = [library_ newFunctionWithName:name];
    if (function == nil) {
      throw BackendError("MPS shader library 缺少 kernel：" +
                         std::string([name UTF8String]));
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device_ newComputePipelineStateWithFunction:function
                                               error:&error];
    if (pipeline == nil) {
      throw BackendError(DescribeError("创建 MPS compute pipeline 失败", error));
    }
    return pipeline;
  }

  id<MTLBuffer> NewBuffer(const void* bytes,
                          std::size_t length,
                          const char* field) const {
    id<MTLBuffer> buffer = bytes == nullptr
                               ? [device_ newBufferWithLength:length
                                                      options:MTLResourceStorageModeShared]
                               : [device_ newBufferWithBytes:bytes
                                                      length:length
                                                     options:MTLResourceStorageModeShared];
    if (buffer == nil) {
      throw BackendError(std::string("分配 MPS buffer 失败：") + field);
    }
    return buffer;
  }

  void ValidateWorkingSet(std::initializer_list<std::size_t> lengths) const {
    std::size_t total = 0;
    for (const std::size_t length : lengths) {
      total = CheckedAddBytes(total, length, "MPS");
    }
    const std::uint64_t recommended =
        static_cast<std::uint64_t>([device_ recommendedMaxWorkingSetSize]);
    if (recommended != 0 && total > recommended) {
      throw BackendError("MPS 工作区超过设备建议上限；请减少行数、特征数或 max_bins");
    }
  }

  id<MTLCommandBuffer> NewCommand(const char* stage) const {
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    if (command == nil) {
      throw BackendError(std::string("创建 MPS command buffer 失败：") + stage);
    }
    return command;
  }

  static void Complete(id<MTLCommandBuffer> command, const char* stage) {
    [command commit];
    [command waitUntilCompleted];
    if ([command status] != MTLCommandBufferStatusCompleted) {
      throw BackendError(DescribeError(stage, [command error]));
    }
  }

  id<MTLDevice> device_;
  id<MTLLibrary> library_;
  id<MTLCommandQueue> queue_;
  id<MTLComputePipelineState> vector_add_;
  id<MTLComputePipelineState> gradients_;
  id<MTLComputePipelineState> histogram_u8_;
  id<MTLComputePipelineState> histogram_u16_;
  id<MTLComputePipelineState> histogram_baseline_u8_;
  id<MTLComputePipelineState> histogram_baseline_u16_;
  id<MTLComputePipelineState> histogram_reduce_;
  mutable BackendTiming timing_;
};

MpsBackend::MpsBackend(std::string metallib_path)
    : impl_(std::make_unique<Impl>(metallib_path)) {}
MpsBackend::~MpsBackend() = default;
MpsBackend::MpsBackend(MpsBackend&&) noexcept = default;
MpsBackend& MpsBackend::operator=(MpsBackend&&) noexcept = default;

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

NodeHistograms MpsBackend::BuildHistograms(
    const BinnedDataset& dataset,
    const std::vector<std::uint64_t>& rows,
    const std::vector<GradientPair>& gradients) const {
  return BuildHistogramsInternal(dataset, rows, gradients, false);
}

NodeHistograms MpsBackend::BuildBaselineHistogramsForTest(
    const BinnedDataset& dataset,
    const std::vector<std::uint64_t>& rows,
    const std::vector<GradientPair>& gradients) const {
  return BuildHistogramsInternal(dataset, rows, gradients, true);
}

NodeHistograms MpsBackend::BuildHistogramsInternal(
    const BinnedDataset& dataset,
    const std::vector<std::uint64_t>& rows,
    const std::vector<GradientPair>& gradients,
    bool baseline) const {
  if (rows.empty()) {
    throw TrainingError("节点行集合不能为空");
  }
  if (dataset.rows() != gradients.size()) {
    throw TrainingError("Gradient 数量与分箱数据行数不一致");
  }
  const std::uint32_t dataset_rows = CheckedUInt32(dataset.rows(), "数据行数");
  const std::uint32_t selected_rows = CheckedUInt32(rows.size(), "节点行数");
  const std::uint32_t features = dataset.features();
  std::vector<std::uint32_t> cell_features;
  std::vector<std::uint32_t> cell_bins;
  std::vector<std::uint32_t> feature_offsets;
  std::vector<std::uint32_t> feature_bin_counts;
  std::uint32_t maximum_feature_bins = 0;
  for (std::uint32_t feature = 0; feature < features; ++feature) {
    const std::uint32_t bin_count = dataset.feature_metadata()[feature].bin_count;
    if (cell_features.size() >
        std::numeric_limits<std::uint32_t>::max() - bin_count) {
      throw BackendError("Histogram cell 数量溢出");
    }
    feature_offsets.push_back(
        CheckedUInt32(cell_features.size(), "Histogram feature offset"));
    feature_bin_counts.push_back(bin_count);
    maximum_feature_bins = std::max(maximum_feature_bins, bin_count);
    for (std::uint32_t bin = 0; bin < bin_count; ++bin) {
      cell_features.push_back(feature);
      cell_bins.push_back(bin);
    }
  }
  const std::uint32_t cell_count =
      CheckedUInt32(cell_features.size(), "Histogram cell 数量");
  const std::size_t threadgroup_bytes = CheckedBytes(
      maximum_feature_bins, sizeof(std::uint32_t) * 3,
      "Histogram threadgroup 工作区");
  const bool effective_baseline =
      baseline || threadgroup_bytes > [impl_->device_ maxThreadgroupMemoryLength];
  id<MTLComputePipelineState> partial_pipeline =
      effective_baseline
          ? (dataset.storage() == BinStorage::kUInt8
                 ? impl_->histogram_baseline_u8_
                 : impl_->histogram_baseline_u16_)
          : (dataset.storage() == BinStorage::kUInt8 ? impl_->histogram_u8_
                                                     : impl_->histogram_u16_);
  const std::uint32_t reduction_width = ReductionWidth(partial_pipeline);
  const std::uint32_t partial_count = std::min<std::uint32_t>(
      kMaximumHistogramPartials,
      (selected_rows + reduction_width - 1) / reduction_width);

  std::vector<std::uint32_t> rows_u32(rows.size());
  for (std::size_t index = 0; index < rows.size(); ++index) {
    rows_u32[index] = CheckedUInt32(rows[index], "Histogram 行索引");
    if (rows_u32[index] >= dataset_rows) {
      throw TrainingError("Histogram 行索引越界");
    }
  }
  std::vector<float> gradient_values(gradients.size() * 2);
  for (std::size_t index = 0; index < gradients.size(); ++index) {
    gradient_values[index * 2] = CheckedFloat(gradients[index].gradient, "Gradient");
    gradient_values[index * 2 + 1] = CheckedFloat(gradients[index].hessian, "Hessian");
    if (gradient_values[index * 2 + 1] < 0.0F) {
      throw TrainingError("Hessian 必须非负");
    }
  }

  const std::size_t bin_item_size =
      dataset.storage() == BinStorage::kUInt8 ? sizeof(std::uint8_t) : sizeof(std::uint16_t);
  const std::size_t bin_bytes = CheckedBytes(
      static_cast<std::size_t>(dataset.bin_value_count()), bin_item_size, "分箱 buffer");
  const std::size_t row_bytes = CheckedBytes(rows_u32.size(), sizeof(std::uint32_t), "行索引 buffer");
  const std::size_t gradient_bytes =
      CheckedBytes(gradient_values.size(), sizeof(float), "Gradient buffer");
  const std::size_t cell_map_bytes =
      CheckedBytes(cell_features.size(), sizeof(std::uint32_t), "Histogram cell map");
  const std::size_t feature_map_bytes =
      CheckedBytes(features, sizeof(std::uint32_t), "Histogram feature map");
  const std::size_t partial_values =
      CheckedBytes(cell_count, partial_count, "Histogram partial 数量");
  const std::size_t partial_bytes =
      CheckedBytes(partial_values, sizeof(DeviceHistogramValue), "Histogram partial");
  const std::size_t output_bytes =
      CheckedBytes(cell_count, sizeof(DeviceHistogramValue), "Histogram 输出");
  impl_->ValidateWorkingSet(
      {bin_bytes, row_bytes, gradient_bytes, cell_map_bytes, cell_map_bytes,
       feature_map_bytes, feature_map_bytes,
       effective_baseline ? 0 : partial_bytes, output_bytes});

  @autoreleasepool {
    id<MTLBuffer> bins_buffer = impl_->NewBuffer(dataset.bin_data(), bin_bytes, "bins");
    id<MTLBuffer> rows_buffer = impl_->NewBuffer(rows_u32.data(), row_bytes, "rows");
    id<MTLBuffer> gradients_buffer =
        impl_->NewBuffer(gradient_values.data(), gradient_bytes, "gradients");
    id<MTLBuffer> cell_features_buffer =
        impl_->NewBuffer(cell_features.data(), cell_map_bytes, "cell features");
    id<MTLBuffer> cell_bins_buffer =
        impl_->NewBuffer(cell_bins.data(), cell_map_bytes, "cell bins");
    id<MTLBuffer> feature_offsets_buffer =
        impl_->NewBuffer(feature_offsets.data(), feature_map_bytes, "feature offsets");
    id<MTLBuffer> feature_bin_counts_buffer = impl_->NewBuffer(
        feature_bin_counts.data(), feature_map_bytes, "feature bin counts");
    id<MTLBuffer> partial_buffer =
        effective_baseline ? nil : impl_->NewBuffer(nullptr, partial_bytes, "partials");
    id<MTLBuffer> output_buffer = impl_->NewBuffer(nullptr, output_bytes, "histogram");

    id<MTLCommandBuffer> histogram_command = impl_->NewCommand("histogram");
    const auto encoding_started = std::chrono::steady_clock::now();
    id<MTLComputeCommandEncoder> partial_encoder =
        [histogram_command computeCommandEncoder];
    if (partial_encoder == nil) {
      throw BackendError("创建 histogram partial encoder 失败");
    }
    [partial_encoder setComputePipelineState:partial_pipeline];
    [partial_encoder setBuffer:bins_buffer offset:0 atIndex:0];
    [partial_encoder setBuffer:rows_buffer offset:0 atIndex:1];
    [partial_encoder setBuffer:gradients_buffer offset:0 atIndex:2];
    [partial_encoder setBuffer:effective_baseline ? output_buffer : partial_buffer
                        offset:0
                       atIndex:3];
    [partial_encoder setBuffer:effective_baseline ? cell_features_buffer
                                                  : feature_offsets_buffer
                        offset:0
                       atIndex:4];
    [partial_encoder setBuffer:effective_baseline ? cell_bins_buffer
                                                  : feature_bin_counts_buffer
                        offset:0
                       atIndex:5];
    [partial_encoder setBytes:&dataset_rows length:sizeof(dataset_rows) atIndex:6];
    [partial_encoder setBytes:&selected_rows length:sizeof(selected_rows) atIndex:7];
    if (effective_baseline) {
      [partial_encoder setBytes:&cell_count length:sizeof(cell_count) atIndex:8];
      const NSUInteger baseline_width = std::min<NSUInteger>(
          kThreadsPerGroup, [partial_pipeline maxTotalThreadsPerThreadgroup]);
      [partial_encoder dispatchThreads:MTLSizeMake(cell_count, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(baseline_width, 1, 1)];
    } else {
      [partial_encoder setBytes:&partial_count length:sizeof(partial_count) atIndex:8];
      [partial_encoder setBytes:&cell_count length:sizeof(cell_count) atIndex:9];
      [partial_encoder setThreadgroupMemoryLength:maximum_feature_bins *
                                                  sizeof(std::uint32_t)
                                          atIndex:0];
      [partial_encoder setThreadgroupMemoryLength:maximum_feature_bins *
                                                  sizeof(std::uint32_t)
                                          atIndex:1];
      [partial_encoder setThreadgroupMemoryLength:maximum_feature_bins *
                                                  sizeof(std::uint32_t)
                                          atIndex:2];
      [partial_encoder dispatchThreadgroups:MTLSizeMake(features, partial_count, 1)
                      threadsPerThreadgroup:MTLSizeMake(reduction_width, 1, 1)];
    }
    [partial_encoder endEncoding];
    if (!effective_baseline) {
      // 同一 command buffer 中后一个 encoder 会看到前一个 encoder 的完整写入结果，
      // 无需 host 中间同步；这消除每个节点一次额外提交与 wait。
      id<MTLComputeCommandEncoder> reduce_encoder =
          [histogram_command computeCommandEncoder];
      if (reduce_encoder == nil) {
        throw BackendError("创建 histogram reduction encoder 失败");
      }
      [reduce_encoder setComputePipelineState:impl_->histogram_reduce_];
      [reduce_encoder setBuffer:partial_buffer offset:0 atIndex:0];
      [reduce_encoder setBuffer:output_buffer offset:0 atIndex:1];
      [reduce_encoder setBytes:&cell_count length:sizeof(cell_count) atIndex:2];
      [reduce_encoder setBytes:&partial_count length:sizeof(partial_count) atIndex:3];
      const NSUInteger width = std::min<NSUInteger>(
          kThreadsPerGroup, [impl_->histogram_reduce_ maxTotalThreadsPerThreadgroup]);
      [reduce_encoder dispatchThreads:MTLSizeMake(cell_count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
      [reduce_encoder endEncoding];
    }
    impl_->timing_.histogram_encode_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      encoding_started)
            .count();
    const auto command_started = std::chrono::steady_clock::now();
    Impl::Complete(histogram_command, "MPS histogram command 执行失败");
    impl_->timing_.histogram_command_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      command_started)
            .count();

    const auto* values =
        static_cast<const DeviceHistogramValue*>([output_buffer contents]);
    NodeHistograms result;
    result.reserve(features);
    std::size_t cell_offset = 0;
    for (std::uint32_t feature = 0; feature < features; ++feature) {
      const std::uint32_t bin_count = dataset.feature_metadata()[feature].bin_count;
      FeatureHistogram feature_histogram;
      feature_histogram.reserve(bin_count);
      for (std::uint32_t bin = 0; bin < bin_count; ++bin) {
        const DeviceHistogramValue& value = values[cell_offset + bin];
        feature_histogram.push_back(HistogramBin{
            value.count, value.gradient, value.hessian});
      }
      result.push_back(std::move(feature_histogram));
      cell_offset += bin_count;
    }
    return result;
  }
}

BackendTiming MpsBackend::last_timing() const noexcept { return impl_->timing_; }

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
