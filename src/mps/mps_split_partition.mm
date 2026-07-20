// MPSBoost Metal split scan and row partition execution.
//
// Intent: keep hot path split helpers together because they share row validation,
// working-set checks, and GPU ABI structs but do not own tree growth policy.

#include "mps_backend_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace mpsboost {

std::vector<SplitScanCandidate> MpsBackend::ScanSplitsForTest(
    const BinnedDataset& dataset,
    const std::vector<std::uint64_t>& rows,
    const std::vector<GradientPair>& gradients,
    std::uint64_t min_samples_leaf,
    double min_child_weight,
    double reg_lambda,
    double gamma) const {
  if (rows.empty()) {
    throw TrainingError("Split-scan row set must not be empty");
  }
  if (min_samples_leaf == 0 || !std::isfinite(min_child_weight) ||
      min_child_weight < 0.0 || !std::isfinite(reg_lambda) ||
      reg_lambda < 0.0 || !std::isfinite(gamma) || gamma < 0.0) {
    throw TrainingError("Split-scan parameters are invalid");
  }
  const NodeHistograms histograms = BuildHistograms(dataset, rows, gradients);
  std::vector<std::uint32_t> cell_features;
  std::vector<std::uint32_t> cell_bins;
  std::vector<std::uint32_t> feature_offsets;
  std::vector<std::uint32_t> feature_bin_counts;
  std::uint32_t maximum_feature_bins = 0;
  BuildHistogramLayout(dataset, &cell_features, &cell_bins, &feature_offsets,
                       &feature_bin_counts, &maximum_feature_bins);
  const std::uint32_t features = dataset.features();
  const std::uint32_t cell_count =
      CheckedUInt32(cell_features.size(), "split-scan cell count");

  std::vector<DeviceHistogramValue> flat_histogram;
  flat_histogram.reserve(cell_count);
  for (const FeatureHistogram& feature : histograms) {
    for (const HistogramBin& bin : feature) {
      flat_histogram.push_back(DeviceHistogramValue{
          CheckedUInt32(bin.count, "Split scan bin count"),
          CheckedFloat(bin.gradient_sum, "Split scan gradient"),
          CheckedFloat(bin.hessian_sum, "Split scan hessian"),
          0});
    }
  }

  const std::size_t histogram_bytes =
      CheckedBytes(flat_histogram.size(), sizeof(DeviceHistogramValue),
                   "Split scan histogram");
  const std::size_t feature_map_bytes =
      CheckedBytes(features, sizeof(std::uint32_t), "Split scan feature map");
  const std::size_t output_bytes =
      CheckedBytes(features, sizeof(DeviceSplitCandidate),
                   "Split scan output");
  impl_->ValidateWorkingSet({histogram_bytes, feature_map_bytes,
                             feature_map_bytes, output_bytes});

  @autoreleasepool {
    id<MTLBuffer> histogram_buffer =
        impl_->NewBuffer(flat_histogram.data(), histogram_bytes, "split histogram");
    id<MTLBuffer> feature_offsets_buffer =
        impl_->NewBuffer(feature_offsets.data(), feature_map_bytes,
                         "split feature offsets");
    id<MTLBuffer> feature_bin_counts_buffer =
        impl_->NewBuffer(feature_bin_counts.data(), feature_map_bytes,
                         "split feature bin counts");
    id<MTLBuffer> output_buffer =
        impl_->NewScratchBuffer(output_bytes, "split candidates");

    const std::uint32_t min_samples_leaf_u32 =
        CheckedUInt32(min_samples_leaf, "Split scan min_samples_leaf");
    const float min_child_weight_f =
        CheckedFloat(min_child_weight, "Split scan min_child_weight");
    const float reg_lambda_f = CheckedFloat(reg_lambda, "Split scan reg_lambda");
    const float gamma_f = CheckedFloat(gamma, "Split scan gamma");

    id<MTLCommandBuffer> command = impl_->NewCommand("split_scan");
    const auto encoding_started = std::chrono::steady_clock::now();
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) {
      throw BackendError("Failed to create split-scan encoder");
    }
    [encoder setComputePipelineState:impl_->split_scan_];
    [encoder setBuffer:histogram_buffer offset:0 atIndex:0];
    [encoder setBuffer:feature_offsets_buffer offset:0 atIndex:1];
    [encoder setBuffer:feature_bin_counts_buffer offset:0 atIndex:2];
    [encoder setBuffer:output_buffer offset:0 atIndex:3];
    [encoder setBytes:&features length:sizeof(features) atIndex:4];
    [encoder setBytes:&min_samples_leaf_u32
               length:sizeof(min_samples_leaf_u32)
              atIndex:5];
    [encoder setBytes:&min_child_weight_f
               length:sizeof(min_child_weight_f)
              atIndex:6];
    [encoder setBytes:&reg_lambda_f length:sizeof(reg_lambda_f) atIndex:7];
    [encoder setBytes:&gamma_f length:sizeof(gamma_f) atIndex:8];
    const NSUInteger width = std::min<NSUInteger>(
        kThreadsPerGroup, [impl_->split_scan_ maxTotalThreadsPerThreadgroup]);
    [encoder dispatchThreads:MTLSizeMake(features, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    impl_->timing_.hot_path_encode_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      encoding_started)
            .count();
    const auto command_started = std::chrono::steady_clock::now();
    Impl::Complete(command, "MPS split-scan command failed");
    impl_->timing_.hot_path_command_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      command_started)
            .count();

    const auto* device_candidates =
        static_cast<const DeviceSplitCandidate*>([output_buffer contents]);
    std::vector<SplitScanCandidate> result;
    result.reserve(features);
    for (std::uint32_t feature = 0; feature < features; ++feature) {
      const DeviceSplitCandidate& candidate = device_candidates[feature];
      result.push_back(SplitScanCandidate{
          candidate.valid != 0,
          candidate.feature,
          candidate.threshold_bin,
          candidate.left_count,
          candidate.right_count,
          candidate.left_gradient,
          candidate.left_hessian,
          candidate.right_gradient,
          candidate.right_hessian,
          candidate.gain});
    }
    impl_->ReturnScratchBuffer(output_buffer);
    return result;
  }
}

std::pair<std::vector<std::uint64_t>, std::vector<std::uint64_t>>
MpsBackend::PartitionRowsForTest(const BinnedDataset& dataset,
                                 const std::vector<std::uint64_t>& rows,
                                 std::uint32_t feature,
                                 std::uint32_t threshold_bin) const {
  if (rows.empty()) {
    throw TrainingError("Partition row set must not be empty");
  }
  if (feature >= dataset.features() ||
      threshold_bin >= dataset.feature_metadata()[feature].bin_count) {
    throw TrainingError("Partition feature or threshold is out of bounds");
  }
  const std::uint32_t dataset_rows = CheckedUInt32(dataset.rows(), "partition dataset row count");
  const std::uint32_t selected_rows = CheckedUInt32(rows.size(), "partition row count");
  const std::vector<std::uint32_t> rows_u32 =
      MakeRowsU32(rows, dataset_rows, "partition row index");
  const std::size_t bin_item_size =
      dataset.storage() == BinStorage::kUInt8 ? sizeof(std::uint8_t)
                                              : sizeof(std::uint16_t);
  const std::size_t bin_bytes =
      CheckedBytes(static_cast<std::size_t>(dataset.bin_value_count()),
                   bin_item_size, "partition binned-data buffer");
  const std::size_t row_bytes =
      CheckedBytes(rows_u32.size(), sizeof(std::uint32_t), "Partition rows");
  const std::size_t count_bytes =
      CheckedBytes(2, sizeof(std::uint32_t), "Partition counts");
  impl_->ValidateWorkingSet({bin_bytes, row_bytes, row_bytes, row_bytes,
                             count_bytes});

  @autoreleasepool {
    std::uint32_t zero_counts[2] = {0, 0};
    id<MTLBuffer> bins_buffer =
        impl_->NewBuffer(dataset.bin_data(), bin_bytes, "partition bins");
    id<MTLBuffer> rows_buffer =
        impl_->NewBuffer(rows_u32.data(), row_bytes, "partition input rows");
    id<MTLBuffer> left_buffer =
        impl_->NewScratchBuffer(row_bytes, "partition left rows");
    id<MTLBuffer> right_buffer =
        impl_->NewScratchBuffer(row_bytes, "partition right rows");
    id<MTLBuffer> counts_buffer =
        impl_->NewBuffer(zero_counts, count_bytes, "partition counts");

    id<MTLCommandBuffer> command = impl_->NewCommand("partition");
    const auto encoding_started = std::chrono::steady_clock::now();
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (encoder == nil) {
      throw BackendError("Failed to create partition encoder");
    }
    id<MTLComputePipelineState> pipeline =
        dataset.storage() == BinStorage::kUInt8 ? impl_->partition_u8_
                                                : impl_->partition_u16_;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:bins_buffer offset:0 atIndex:0];
    [encoder setBuffer:rows_buffer offset:0 atIndex:1];
    [encoder setBuffer:left_buffer offset:0 atIndex:2];
    [encoder setBuffer:right_buffer offset:0 atIndex:3];
    [encoder setBuffer:counts_buffer offset:0 atIndex:4];
    [encoder setBytes:&dataset_rows length:sizeof(dataset_rows) atIndex:5];
    [encoder setBytes:&selected_rows length:sizeof(selected_rows) atIndex:6];
    [encoder setBytes:&feature length:sizeof(feature) atIndex:7];
    [encoder setBytes:&threshold_bin length:sizeof(threshold_bin) atIndex:8];
    const NSUInteger width =
        std::min<NSUInteger>(kThreadsPerGroup,
                             [pipeline maxTotalThreadsPerThreadgroup]);
    [encoder dispatchThreads:MTLSizeMake(selected_rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    impl_->timing_.hot_path_encode_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      encoding_started)
            .count();
    const auto command_started = std::chrono::steady_clock::now();
    Impl::Complete(command, "MPS partition command failed");
    impl_->timing_.hot_path_command_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      command_started)
            .count();

    const auto* counts =
        static_cast<const std::uint32_t*>([counts_buffer contents]);
    if (static_cast<std::uint64_t>(counts[0]) + counts[1] != rows.size()) {
      throw BackendError("Partition output count does not match input row count");
    }
    const auto* left_u32 =
        static_cast<const std::uint32_t*>([left_buffer contents]);
    const auto* right_u32 =
        static_cast<const std::uint32_t*>([right_buffer contents]);
    std::vector<std::uint64_t> left(left_u32, left_u32 + counts[0]);
    std::vector<std::uint64_t> right(right_u32, right_u32 + counts[1]);
    impl_->ReturnScratchBuffer(left_buffer);
    impl_->ReturnScratchBuffer(right_buffer);
    return {std::move(left), std::move(right)};
  }
}

}  // namespace mpsboost
