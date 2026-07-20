// Device-independent orchestration for one depth-limited regression tree.
//
// This unit owns growth, prediction, and restore entry points. Focused internal
// units keep split, histogram, and structural policy identical on CPU and MPS.

#include "mpsboost/tree.hpp"

#include <cmath>
#include <cstddef>
#include <numeric>
#include <utility>

#include "mpsboost/backend.hpp"
#include "mpsboost/binned_dataset.hpp"
#include "tree_internal.hpp"

namespace mpsboost {
using tree_internal::ActiveNode;
using tree_internal::BuildCurrentLayerHistograms;
using tree_internal::BuildPendingChildHistograms;
using tree_internal::EffectiveMaxLeaves;
using tree_internal::ExtendInteractionPath;
using tree_internal::NodeStatistics;
using tree_internal::PendingChildHistogram;
using tree_internal::PreparedSplit;
using tree_internal::PrepareSplitRows;
using tree_internal::SubtractHistograms;
using tree_internal::SumRows;
using tree_internal::TreeTrainingAccess;
using tree_internal::TrainLeafWiseRegressionTree;
using tree_internal::ValidateInteractionConstraints;
using tree_internal::ValidateParameters;
using tree_internal::ValidateTreeStructure;

namespace {

RegressionTree InitializeTree(const BinnedDataset& dataset,
                              std::vector<std::uint64_t>* root_rows,
                              NodeStatistics* root_statistics,
                              const std::vector<GradientPair>& gradients,
                              const TreeTrainingParameters& parameters) {
  root_rows->resize(static_cast<std::size_t>(dataset.rows()));
  std::iota(root_rows->begin(), root_rows->end(), std::uint64_t{0});
  *root_statistics = SumRows(*root_rows, gradients);
  return TreeTrainingAccess::Create(dataset.features(), *root_statistics,
                                    parameters);
}

RegressionTree TrainLevelWiseRegressionTree(
    const BinnedDataset& dataset,
    const std::vector<GradientPair>& gradients,
    const TreeTrainingParameters& parameters,
    const HistogramBuilder& histogram_builder) {
  std::vector<std::uint64_t> root_rows;
  NodeStatistics root_statistics;
  RegressionTree tree =
      InitializeTree(dataset, &root_rows, &root_statistics, gradients, parameters);

  std::vector<ActiveNode> current_layer;
  current_layer.push_back(
      ActiveNode{0, 0, std::move(root_rows), root_statistics, {},
                 -std::numeric_limits<double>::infinity(),
                 std::numeric_limits<double>::infinity(), {}});
  std::uint32_t leaf_count = 1;
  const std::uint32_t max_leaves = EffectiveMaxLeaves(parameters);

  while (!current_layer.empty() && leaf_count < max_leaves) {
    if (current_layer.front().depth >= parameters.max_depth) {
      break;
    }
    std::vector<ActiveNode> next_layer;
    std::vector<PendingChildHistogram> pending_child_histograms;
    const std::vector<NodeHistograms> layer_histograms =
        BuildCurrentLayerHistograms(dataset, current_layer, gradients,
                                    histogram_builder);
    for (std::size_t active_index = 0; active_index < current_layer.size();
         ++active_index) {
      ActiveNode& active = current_layer[active_index];
      if (active.depth >= parameters.max_depth) {
        continue;
      }
      if (leaf_count >= max_leaves) {
        break;
      }
      const PreparedSplit prepared =
          PrepareSplitRows(dataset, active, layer_histograms[active_index],
                           gradients, parameters);
      if (!prepared.valid) {
        continue;
      }

      std::uint32_t left_index = 0;
      std::uint32_t right_index = 0;
      TreeTrainingAccess::ApplySplit(&tree, active, prepared, parameters,
                                     &left_index, &right_index);
      ++leaf_count;

      const std::uint32_t child_depth = active.depth + 1;
      const std::vector<std::uint32_t> child_path =
          ExtendInteractionPath(active.path_features, prepared.split.feature);
      next_layer.push_back(ActiveNode{
          left_index, child_depth, std::move(prepared.left_rows),
          prepared.split.left, {}, prepared.split.left_lower_bound,
          prepared.split.left_upper_bound, child_path});
      next_layer.push_back(ActiveNode{
          right_index, child_depth, std::move(prepared.right_rows),
          prepared.split.right, {}, prepared.split.right_lower_bound,
          prepared.split.right_upper_bound, child_path});
      if (child_depth < parameters.max_depth) {
        const bool build_left =
            prepared.split.left.count <= prepared.split.right.count;
        const std::size_t child_index =
            next_layer.size() - (build_left ? 2U : 1U);
        pending_child_histograms.push_back(PendingChildHistogram{
            child_index, next_layer[child_index].rows,
            layer_histograms[active_index]});
      }
    }
    if (!pending_child_histograms.empty()) {
      const std::vector<NodeHistograms> built_child_histograms =
          BuildPendingChildHistograms(dataset, pending_child_histograms, gradients,
                                      histogram_builder);
      if (built_child_histograms.size() != pending_child_histograms.size()) {
        throw TrainingError("Histogram subtraction child count does not match");
      }
      for (std::size_t index = 0; index < pending_child_histograms.size();
           ++index) {
        const PendingChildHistogram& pending = pending_child_histograms[index];
        if (pending.next_layer_index >= next_layer.size()) {
          throw TrainingError("Histogram subtraction child index is out of bounds");
        }
        next_layer[pending.next_layer_index].cached_histograms =
            built_child_histograms[index];
        const std::size_t sibling_index =
            pending.next_layer_index ^ std::size_t{1};
        if (sibling_index >= next_layer.size()) {
          throw TrainingError("Histogram subtraction sibling index is out of bounds");
        }
        next_layer[sibling_index].cached_histograms =
            SubtractHistograms(pending.parent_histograms,
                               built_child_histograms[index]);
      }
    }
    current_layer = std::move(next_layer);
  }
  return tree;
}

}  // namespace

RegressionTree TrainSingleRegressionTree(
    const BinnedDataset& dataset,
    const std::vector<GradientPair>& gradients,
    const TreeTrainingParameters& parameters,
    const HistogramBuilder& histogram_builder) {
  ValidateParameters(parameters);
  if (!parameters.monotonic_constraints.empty() &&
      parameters.monotonic_constraints.size() != dataset.features()) {
    throw TrainingError("monotonic_constraints length must match feature count");
  }
  for (const std::int8_t constraint : parameters.monotonic_constraints) {
    if (constraint < -1 || constraint > 1) {
      throw TrainingError("monotonic_constraints values must be -1, 0, or 1");
    }
  }
  ValidateInteractionConstraints(parameters, dataset.features());
  if (dataset.rows() != static_cast<std::uint64_t>(gradients.size())) {
    throw TrainingError("Gradient count does not match binned dataset row count");
  }
  if (parameters.growth_strategy ==
      TreeTrainingParameters::GrowthStrategy::kLeafWise) {
    return TrainLeafWiseRegressionTree(dataset, gradients, parameters,
                                       histogram_builder);
  }
  return TrainLevelWiseRegressionTree(dataset, gradients, parameters,
                                      histogram_builder);
}

std::vector<double> RegressionTree::Predict(const BinnedDataset& dataset) const {
  if (nodes_.empty()) {
    throw TrainingError("Tree model has no root node");
  }
  if (dataset.features() != feature_count_) {
    throw TrainingError("Prediction feature count does not match the tree model");
  }

  std::vector<double> predictions(static_cast<std::size_t>(dataset.rows()));
  for (std::uint64_t row = 0; row < dataset.rows(); ++row) {
    std::uint32_t node_index = 0;
    std::size_t visited = 0;
    while (true) {
      if (node_index >= nodes_.size() || ++visited > nodes_.size()) {
        throw TrainingError("Tree node index is out of bounds or the structure contains a cycle");
      }
      const TreeNode& node = nodes_[node_index];
      if (node.IsLeaf()) {
        if (!std::isfinite(node.leaf_value)) {
          throw TrainingError("Tree leaf value is not finite");
        }
        predictions[static_cast<std::size_t>(row)] = node.leaf_value;
        break;
      }
      if (node.feature_index >= feature_count_) {
        throw TrainingError("Tree branch feature index is out of bounds");
      }
      const bool goes_left =
          dataset.IsMissing(row, node.feature_index)
              ? node.default_left
              : dataset.GetBin(row, node.feature_index) <= node.threshold_bin;
      node_index = goes_left ? node.left_child : node.right_child;
    }
  }
  return predictions;
}

RegressionTree RegressionTree::Restore(std::uint32_t feature_count,
                                       std::vector<TreeNode> nodes) {
  ValidateTreeStructure(feature_count, nodes);
  RegressionTree tree;
  tree.feature_count_ = feature_count;
  tree.nodes_ = std::move(nodes);
  return tree;
}

}  // namespace mpsboost
