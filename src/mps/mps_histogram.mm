// MPSBoost Metal histogram execution.
//
// Intent: own the two-stage histogram path, baseline fallback path, and layer
// batching entry while leaving tree growth and split semantics outside MPS code.

#include "mps_backend_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace mpsboost {

namespace {

bool HistogramOutputIsValid(const NodeHistograms& histograms,
                            const BinnedDataset& dataset,
                            std::uint64_t expected_rows) {
  if (histograms.size() != dataset.features()) {
    return false;
  }
  for (std::uint32_t feature = 0; feature < dataset.features(); ++feature) {
    if (histograms[feature].size() !=
        dataset.feature_metadata()[feature].bin_count) {
      return false;
    }
    std::uint64_t observed_rows = 0;
    for (const HistogramBin& bin : histograms[feature]) {
      if (bin.count > std::numeric_limits<std::uint64_t>::max() - observed_rows) {
        return false;
      }
      observed_rows += bin.count;
      if (!std::isfinite(bin.gradient_sum) ||
          !std::isfinite(bin.hessian_sum) || bin.hessian_sum < 0.0) {
        return false;
      }
    }
    if (observed_rows != expected_rows) {
      return false;
    }
  }
  return true;
}

}  // namespace

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

std::vector<NodeHistograms> MpsBackend::BuildLayerHistograms(
    const BinnedDataset& dataset,
    const std::vector<std::vector<std::uint64_t>>& node_rows,
    const std::vector<GradientPair>& gradients) const {
  if (node_rows.empty()) {
    return {};
  }
  // 现阶段每个节点仍保持独立 histogram 输出，保证训练核心的 split 选择和分区校验
  // 不变；同层入口让调用方避免逐节点动态分派，并让后端 L1 buffer pool 跨节点复用
  // partial/output 工作区。后续可以在此函数内部进一步合并 command，而不改核心语义。
  std::vector<NodeHistograms> result;
  result.reserve(node_rows.size());
  for (const std::vector<std::uint64_t>& rows : node_rows) {
    result.push_back(BuildHistogramsInternal(dataset, rows, gradients, false));
  }
  return result;
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
  BuildHistogramLayout(dataset, &cell_features, &cell_bins, &feature_offsets,
                       &feature_bin_counts, &maximum_feature_bins);
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

  std::vector<std::uint32_t> rows_u32 =
      MakeRowsU32(rows, dataset_rows, "Histogram 行索引");
  std::vector<float> gradient_values = MakeGradientValues(dataset, gradients);

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
        effective_baseline ? nil : impl_->NewScratchBuffer(partial_bytes, "partials");
    id<MTLBuffer> output_buffer =
        impl_->NewScratchBuffer(output_bytes, "histogram");

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
    NodeHistograms result = DecodeHistograms(dataset, values);
    impl_->ReturnScratchBuffer(partial_buffer);
    impl_->ReturnScratchBuffer(output_buffer);
    if (!HistogramOutputIsValid(result, dataset, rows.size())) {
      if (effective_baseline) {
        throw TrainingError("MPS baseline histogram produced invalid statistics");
      }
      return BuildHistogramsInternal(dataset, rows, gradients, true);
    }
    return result;
  }
}

}  // namespace mpsboost
