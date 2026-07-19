// MPSBoost Metal backend internal implementation details.
//
// Intent: keep Objective-C/Metal types out of public headers while allowing the
// backend implementation to be split by responsibility. These declarations are
// private to the native extension and preserve one shared MpsBackend::Impl.
#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "mpsboost/backend.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mpsboost {

constexpr std::uint32_t kThreadsPerGroup = 256;
constexpr std::uint32_t kMaximumHistogramPartials = 16;

struct DeviceHistogramValue final {
  std::uint32_t count;
  float gradient;
  float hessian;
  std::uint32_t reserved;
};
static_assert(sizeof(DeviceHistogramValue) == 16);

struct DeviceSplitCandidate final {
  std::uint32_t valid;
  std::uint32_t feature;
  std::uint32_t threshold_bin;
  std::uint32_t left_count;
  std::uint32_t right_count;
  float left_gradient;
  float left_hessian;
  float right_gradient;
  float right_hessian;
  float gain;
};
static_assert(sizeof(DeviceSplitCandidate) == 40);

std::string DescribeError(const char* stage, NSError* error);
std::size_t CheckedBytes(std::size_t count, std::size_t item_size, const char* field);
std::size_t CheckedAddBytes(std::size_t left, std::size_t right, const char* field);
std::uint32_t CheckedUInt32(std::uint64_t value, const char* field);
float CheckedFloat(double value, const char* field);
std::uint32_t ReductionWidth(id<MTLComputePipelineState> pipeline);
std::vector<std::uint32_t> MakeRowsU32(const std::vector<std::uint64_t>& rows,
                                       std::uint32_t dataset_rows,
                                       const char* context);
std::vector<float> MakeGradientValues(const BinnedDataset& dataset,
                                      const std::vector<GradientPair>& gradients);
void BuildHistogramLayout(const BinnedDataset& dataset,
                          std::vector<std::uint32_t>* cell_features,
                          std::vector<std::uint32_t>* cell_bins,
                          std::vector<std::uint32_t>* feature_offsets,
                          std::vector<std::uint32_t>* feature_bin_counts,
                          std::uint32_t* maximum_feature_bins);
NodeHistograms DecodeHistograms(const BinnedDataset& dataset,
                                const DeviceHistogramValue* values);

class MpsBackend::Impl final {
 public:
  explicit Impl(const std::string& metallib_path);
  id<MTLComputePipelineState> MakePipeline(NSString* name);
  id<MTLBuffer> NewBuffer(const void* bytes, std::size_t length, const char* field) const;
  id<MTLBuffer> NewScratchBuffer(std::size_t length, const char* field) const;
  void ReturnScratchBuffer(id<MTLBuffer> buffer) const;
  void ValidateWorkingSet(std::initializer_list<std::size_t> lengths) const;
  id<MTLCommandBuffer> NewCommand(const char* stage) const;
  static void Complete(id<MTLCommandBuffer> command, const char* stage);

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
  id<MTLComputePipelineState> split_scan_;
  id<MTLComputePipelineState> partition_u8_;
  id<MTLComputePipelineState> partition_u16_;
  mutable BackendTiming timing_;
  mutable std::multimap<std::size_t, id<MTLBuffer>> pooled_buffers_;
};

}  // namespace mpsboost
