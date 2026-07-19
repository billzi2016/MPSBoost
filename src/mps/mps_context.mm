// MPSBoost Metal context, pipeline, and buffer pool implementation.
//
// Intent: centralize Objective-C/Metal resource ownership so compute kernels can
// stay focused on gradient, histogram, split, and partition work.

#include "mps_backend_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <utility>

namespace mpsboost {

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

std::vector<std::uint32_t> MakeRowsU32(const std::vector<std::uint64_t>& rows,
                                       std::uint32_t dataset_rows,
                                       const char* context) {
  std::vector<std::uint32_t> rows_u32(rows.size());
  for (std::size_t index = 0; index < rows.size(); ++index) {
    rows_u32[index] = CheckedUInt32(rows[index], context);
    if (rows_u32[index] >= dataset_rows) {
      throw TrainingError("GPU 行索引越界");
    }
  }
  return rows_u32;
}

std::vector<float> MakeGradientValues(
    const BinnedDataset& dataset,
    const std::vector<GradientPair>& gradients) {
  if (dataset.rows() != gradients.size()) {
    throw TrainingError("Gradient 数量与分箱数据行数不一致");
  }
  std::vector<float> gradient_values(gradients.size() * 2);
  for (std::size_t index = 0; index < gradients.size(); ++index) {
    gradient_values[index * 2] = CheckedFloat(gradients[index].gradient, "Gradient");
    gradient_values[index * 2 + 1] = CheckedFloat(gradients[index].hessian, "Hessian");
    if (gradient_values[index * 2 + 1] < 0.0F) {
      throw TrainingError("Hessian 必须非负");
    }
  }
  return gradient_values;
}

void BuildHistogramLayout(const BinnedDataset& dataset,
                          std::vector<std::uint32_t>* cell_features,
                          std::vector<std::uint32_t>* cell_bins,
                          std::vector<std::uint32_t>* feature_offsets,
                          std::vector<std::uint32_t>* feature_bin_counts,
                          std::uint32_t* maximum_feature_bins) {
  cell_features->clear();
  cell_bins->clear();
  feature_offsets->clear();
  feature_bin_counts->clear();
  *maximum_feature_bins = 0;
  for (std::uint32_t feature = 0; feature < dataset.features(); ++feature) {
    const std::uint32_t bin_count = dataset.feature_metadata()[feature].bin_count;
    if (cell_features->size() >
        std::numeric_limits<std::uint32_t>::max() - bin_count) {
      throw BackendError("Histogram cell 数量溢出");
    }
    feature_offsets->push_back(
        CheckedUInt32(cell_features->size(), "Histogram feature offset"));
    feature_bin_counts->push_back(bin_count);
    *maximum_feature_bins = std::max(*maximum_feature_bins, bin_count);
    for (std::uint32_t bin = 0; bin < bin_count; ++bin) {
      cell_features->push_back(feature);
      cell_bins->push_back(bin);
    }
  }
}

NodeHistograms DecodeHistograms(const BinnedDataset& dataset,
                                const DeviceHistogramValue* values) {
  NodeHistograms result;
  result.reserve(dataset.features());
  std::size_t cell_offset = 0;
  for (std::uint32_t feature = 0; feature < dataset.features(); ++feature) {
    const std::uint32_t bin_count = dataset.feature_metadata()[feature].bin_count;
    FeatureHistogram feature_histogram;
    feature_histogram.reserve(bin_count);
    for (std::uint32_t bin = 0; bin < bin_count; ++bin) {
      const DeviceHistogramValue& value = values[cell_offset + bin];
      feature_histogram.push_back(
          HistogramBin{value.count, value.gradient, value.hessian});
    }
    result.push_back(std::move(feature_histogram));
    cell_offset += bin_count;
  }
  return result;
}

MpsBackend::Impl::Impl(const std::string& metallib_path) {
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
      split_scan_ = MakePipeline(@"split_scan_features");
      partition_u8_ = MakePipeline(@"partition_rows_u8");
      partition_u16_ = MakePipeline(@"partition_rows_u16");
    }
  }

id<MTLComputePipelineState> MpsBackend::Impl::MakePipeline(NSString* name) {
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

id<MTLBuffer> MpsBackend::Impl::NewBuffer(const void* bytes,
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

id<MTLBuffer> MpsBackend::Impl::NewScratchBuffer(std::size_t length, const char* field) const {
    auto iterator = pooled_buffers_.lower_bound(length);
    if (iterator != pooled_buffers_.end()) {
      id<MTLBuffer> buffer = iterator->second;
      pooled_buffers_.erase(iterator);
      ++timing_.pooled_buffer_reuse_count;
      return buffer;
    }
    ++timing_.pooled_buffer_allocation_count;
    return NewBuffer(nullptr, length, field);
  }

void MpsBackend::Impl::ReturnScratchBuffer(id<MTLBuffer> buffer) const {
    if (buffer == nil) {
      return;
    }
    // L1 buffer pool 只复用临时输出/partial 工作区；输入数据 buffer 仍由调用点明确
    // 拥有，避免把可变训练数据跨节点或跨 estimator 意外共享。
    pooled_buffers_.emplace(static_cast<std::size_t>([buffer length]), buffer);
  }

void MpsBackend::Impl::ValidateWorkingSet(std::initializer_list<std::size_t> lengths) const {
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

id<MTLCommandBuffer> MpsBackend::Impl::NewCommand(const char* stage) const {
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    if (command == nil) {
      throw BackendError(std::string("创建 MPS command buffer 失败：") + stage);
    }
    return command;
  }

void MpsBackend::Impl::Complete(id<MTLCommandBuffer> command, const char* stage) {
    [command commit];
    [command waitUntilCompleted];
    if ([command status] != MTLCommandBufferStatusCompleted) {
      throw BackendError(DescribeError(stage, [command error]));
    }
  }


}  // namespace mpsboost
