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
    throw BackendError(std::string(field) + ": byte-count calculation overflow");
  }
  return count * item_size;
}

std::size_t CheckedAddBytes(std::size_t left,
                            std::size_t right,
                            const char* field) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw BackendError(std::string(field) + ": workspace byte-count overflow");
  }
  return left + right;
}

std::uint32_t CheckedUInt32(std::uint64_t value, const char* field) {
  if (value > std::numeric_limits<std::uint32_t>::max()) {
    throw BackendError(std::string(field) + " exceeds the uint32 range of the current GPU ABI");
  }
  return static_cast<std::uint32_t>(value);
}

float CheckedFloat(double value, const char* field) {
  if (!std::isfinite(value) ||
      value > static_cast<double>(std::numeric_limits<float>::max()) ||
      value < -static_cast<double>(std::numeric_limits<float>::max())) {
    throw BackendError(std::string(field) + " cannot be safely converted to GPU float32");
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
      throw TrainingError("GPU row index is out of bounds");
    }
  }
  return rows_u32;
}

std::vector<float> MakeGradientValues(
    const BinnedDataset& dataset,
    const std::vector<GradientPair>& gradients) {
  if (dataset.rows() != gradients.size()) {
    throw TrainingError("Gradient count does not match binned dataset row count");
  }
  std::vector<float> gradient_values(gradients.size() * 2);
  for (std::size_t index = 0; index < gradients.size(); ++index) {
    gradient_values[index * 2] = CheckedFloat(gradients[index].gradient, "Gradient");
    gradient_values[index * 2 + 1] = CheckedFloat(gradients[index].hessian, "Hessian");
    if (gradient_values[index * 2 + 1] < 0.0F) {
      throw TrainingError("Hessian must be non-negative");
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
      throw BackendError("Histogram cell count overflow");
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
        throw BackendError("MPS backend is unavailable: the system did not return a default Metal device");
      }
      NSString* path = [NSString stringWithUTF8String:metallib_path.c_str()];
      if (path == nil || ![[NSFileManager defaultManager] fileExistsAtPath:path]) {
        throw BackendError("MPS shader library does not exist or its path is not valid UTF-8");
      }
      NSError* error = nil;
      library_ = [device_ newLibraryWithURL:[NSURL fileURLWithPath:path isDirectory:NO]
                                      error:&error];
      if (library_ == nil) {
        throw BackendError(DescribeError("Failed to load MPS shader library", error));
      }
      queue_ = [device_ newCommandQueue];
      if (queue_ == nil) {
        throw BackendError("Failed to create MPS command queue");
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
      throw BackendError("MPS shader library is missing kernel: " +
                         std::string([name UTF8String]));
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device_ newComputePipelineStateWithFunction:function
                                               error:&error];
    if (pipeline == nil) {
      throw BackendError(DescribeError("Failed to create MPS compute pipeline", error));
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
      throw BackendError(std::string("Failed to allocate MPS buffer: ") + field);
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
    // The L1 buffer pool reuses only temporary output/partial workspace. Call sites
    // retain explicit ownership of input buffers to prevent mutable training data
    // from being shared accidentally across nodes or estimators.
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
      throw BackendError("MPS workspace exceeds the device-recommended limit; reduce rows, features, or max_bins");
    }
  }

id<MTLCommandBuffer> MpsBackend::Impl::NewCommand(const char* stage) const {
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    if (command == nil) {
      throw BackendError(std::string("Failed to create MPS command buffer: ") + stage);
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
