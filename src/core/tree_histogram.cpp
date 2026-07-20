// Histogram orchestration helpers for layer-wise native tree growth.
//
// The functions in this unit preserve the shared CPU/MPS builder contract and
// implement histogram subtraction without duplicating the training loop.

#include "tree_internal.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace mpsboost::tree_internal {

namespace {

double SubtractHistogramHessian(double parent, double child) {
  if (!std::isfinite(parent) || !std::isfinite(child) || parent < 0.0 ||
      child < 0.0) {
    throw TrainingError("Histogram subtraction Hessian values must be finite and non-negative");
  }
  const double difference = parent - child;
  const double tolerance = 1e-5 * std::max({1.0, std::abs(parent), std::abs(child)});
  if (difference < -tolerance) {
    throw TrainingError("Histogram subtraction child Hessian exceeded parent");
  }
  return difference < 0.0 ? 0.0 : difference;
}

}  // namespace

NodeHistograms SubtractHistograms(const NodeHistograms& parent,
                                  const NodeHistograms& child) {
  if (parent.size() != child.size()) {
    throw TrainingError("Histogram subtraction 特征数量不一致");
  }
  NodeHistograms result;
  result.reserve(parent.size());
  for (std::size_t feature = 0; feature < parent.size(); ++feature) {
    if (parent[feature].size() != child[feature].size()) {
      throw TrainingError("Histogram subtraction bin 数量不一致");
    }
    FeatureHistogram feature_histogram;
    feature_histogram.reserve(parent[feature].size());
    for (std::size_t bin = 0; bin < parent[feature].size(); ++bin) {
      const HistogramBin& parent_bin = parent[feature][bin];
      const HistogramBin& child_bin = child[feature][bin];
      if (child_bin.count > parent_bin.count) {
        throw TrainingError("Histogram subtraction 子节点计数超过父节点");
      }
      const double gradient_sum =
          parent_bin.gradient_sum - child_bin.gradient_sum;
      if (!std::isfinite(parent_bin.gradient_sum) ||
          !std::isfinite(child_bin.gradient_sum) ||
          !std::isfinite(gradient_sum)) {
        throw TrainingError("Histogram subtraction gradient values must be finite");
      }
      feature_histogram.push_back(HistogramBin{
          parent_bin.count - child_bin.count,
          gradient_sum,
          SubtractHistogramHessian(parent_bin.hessian_sum,
                                   child_bin.hessian_sum)});
    }
    result.push_back(std::move(feature_histogram));
  }
  return result;
}

std::vector<NodeHistograms> BuildCurrentLayerHistograms(
    const BinnedDataset& dataset,
    const std::vector<ActiveNode>& current_layer,
    const std::vector<GradientPair>& gradients,
    const HistogramBuilder& histogram_builder) {
  std::vector<std::vector<std::uint64_t>> layer_rows;
  layer_rows.reserve(current_layer.size());
  for (const ActiveNode& active : current_layer) {
    if (active.cached_histograms.empty()) {
      layer_rows.push_back(active.rows);
    }
  }

  std::vector<NodeHistograms> computed_histograms;
  if (!layer_rows.empty()) {
    if (const auto* layer_builder =
            dynamic_cast<const LayerHistogramBuilder*>(&histogram_builder)) {
      computed_histograms =
          layer_builder->BuildLayerHistograms(dataset, layer_rows, gradients);
      if (computed_histograms.size() != layer_rows.size()) {
        throw TrainingError("按层 histogram 返回节点数量与活跃层不一致");
      }
    } else {
      computed_histograms.reserve(layer_rows.size());
      for (const std::vector<std::uint64_t>& rows : layer_rows) {
        computed_histograms.push_back(
            histogram_builder.BuildHistograms(dataset, rows, gradients));
      }
    }
  }

  std::vector<NodeHistograms> result;
  result.reserve(current_layer.size());
  std::size_t computed_index = 0;
  for (const ActiveNode& active : current_layer) {
    if (!active.cached_histograms.empty()) {
      result.push_back(active.cached_histograms);
    } else {
      result.push_back(std::move(computed_histograms[computed_index++]));
    }
  }
  return result;
}

std::vector<NodeHistograms> BuildPendingChildHistograms(
    const BinnedDataset& dataset,
    const std::vector<PendingChildHistogram>& pending,
    const std::vector<GradientPair>& gradients,
    const HistogramBuilder& histogram_builder) {
  std::vector<std::vector<std::uint64_t>> rows_to_build;
  rows_to_build.reserve(pending.size());
  for (const PendingChildHistogram& item : pending) {
    rows_to_build.push_back(item.rows);
  }
  if (const auto* layer_builder =
          dynamic_cast<const LayerHistogramBuilder*>(&histogram_builder)) {
    return layer_builder->BuildLayerHistograms(dataset, rows_to_build,
                                               gradients);
  }
  std::vector<NodeHistograms> result;
  result.reserve(rows_to_build.size());
  for (const std::vector<std::uint64_t>& rows : rows_to_build) {
    result.push_back(histogram_builder.BuildHistograms(dataset, rows, gradients));
  }
  return result;
}

}  // namespace mpsboost::tree_internal
