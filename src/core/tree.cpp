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
using tree_internal::AppendNode;
using tree_internal::BuildCurrentLayerHistograms;
using tree_internal::BuildPendingChildHistograms;
using tree_internal::FindBestSplit;
using tree_internal::MakeLeaf;
using tree_internal::NodeStatistics;
using tree_internal::PendingChildHistogram;
using tree_internal::SplitCandidate;
using tree_internal::SubtractHistograms;
using tree_internal::SumRows;
using tree_internal::ValidateParameters;
using tree_internal::ValidateTreeStructure;

RegressionTree TrainSingleRegressionTree(
    const BinnedDataset& dataset,
    const std::vector<GradientPair>& gradients,
    const TreeTrainingParameters& parameters,
    const HistogramBuilder& histogram_builder) {
  ValidateParameters(parameters);
  if (dataset.rows() != static_cast<std::uint64_t>(gradients.size())) {
    throw TrainingError("Gradient 数量与分箱数据行数不一致");
  }

  std::vector<std::uint64_t> root_rows(static_cast<std::size_t>(dataset.rows()));
  std::iota(root_rows.begin(), root_rows.end(), std::uint64_t{0});
  const NodeStatistics root_statistics = SumRows(root_rows, gradients);

  RegressionTree tree;
  tree.feature_count_ = dataset.features();
  tree.nodes_.reserve(1);
  AppendNode(&tree.nodes_, MakeLeaf(root_statistics, parameters.reg_lambda));

  std::vector<ActiveNode> current_layer;
  current_layer.push_back(
      ActiveNode{0, 0, std::move(root_rows), root_statistics, {}});

  while (!current_layer.empty()) {
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
      const NodeHistograms& histograms = layer_histograms[active_index];
      if (histograms.size() != dataset.features()) {
        throw TrainingError("Histogram 特征数量与数据集不一致");
      }
      const SplitCandidate split =
          FindBestSplit(histograms, dataset, active.statistics, active.node_index,
                        active.depth, parameters);
      if (!split.valid) {
        continue;
      }

      std::vector<std::uint64_t> left_rows;
      std::vector<std::uint64_t> right_rows;
      left_rows.reserve(static_cast<std::size_t>(split.left.count));
      right_rows.reserve(static_cast<std::size_t>(split.right.count));
      for (const std::uint64_t row : active.rows) {
        // Binning uses lower-bound semantics, so values equal to a boundary
        // stay in the lower bin. Training and prediction must both route
        // bin <= threshold left to keep boundary samples on one stable path.
        if (dataset.GetBin(row, split.feature) <= split.threshold_bin) {
          left_rows.push_back(row);
        } else {
          right_rows.push_back(row);
        }
      }
      if (left_rows.size() != split.left.count ||
          right_rows.size() != split.right.count) {
        throw TrainingError("样本分区数量与 histogram 统计不一致");
      }

      const std::uint32_t left_index =
          AppendNode(&tree.nodes_, MakeLeaf(split.left, parameters.reg_lambda));
      const std::uint32_t right_index =
          AppendNode(&tree.nodes_, MakeLeaf(split.right, parameters.reg_lambda));
      TreeNode& parent = tree.nodes_[active.node_index];
      parent.feature_index = split.feature;
      parent.threshold_bin = split.threshold_bin;
      parent.left_child = left_index;
      parent.right_child = right_index;
      parent.leaf_value = 0.0;
      parent.gain = split.gain;
      parent.flags = 0;

      const std::uint32_t child_depth = active.depth + 1;
      next_layer.push_back(ActiveNode{
          left_index, child_depth, std::move(left_rows), split.left, {}});
      next_layer.push_back(ActiveNode{
          right_index, child_depth, std::move(right_rows), split.right, {}});
      if (child_depth < parameters.max_depth) {
        const bool build_left = split.left.count <= split.right.count;
        const std::size_t child_index =
            next_layer.size() - (build_left ? 2U : 1U);
        pending_child_histograms.push_back(PendingChildHistogram{
            child_index, next_layer[child_index].rows, histograms});
      }
    }
    if (!pending_child_histograms.empty()) {
      const std::vector<NodeHistograms> built_child_histograms =
          BuildPendingChildHistograms(dataset, pending_child_histograms, gradients,
                                      histogram_builder);
      if (built_child_histograms.size() != pending_child_histograms.size()) {
        throw TrainingError("Histogram subtraction 子节点数量不一致");
      }
      for (std::size_t index = 0; index < pending_child_histograms.size();
           ++index) {
        const PendingChildHistogram& pending = pending_child_histograms[index];
        if (pending.next_layer_index >= next_layer.size()) {
          throw TrainingError("Histogram subtraction 子节点索引越界");
        }
        next_layer[pending.next_layer_index].cached_histograms =
            built_child_histograms[index];
        const std::size_t sibling_index =
            pending.next_layer_index ^ std::size_t{1};
        if (sibling_index >= next_layer.size()) {
          throw TrainingError("Histogram subtraction 兄弟节点索引越界");
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

std::vector<double> RegressionTree::Predict(const BinnedDataset& dataset) const {
  if (nodes_.empty()) {
    throw TrainingError("树模型没有根节点");
  }
  if (dataset.features() != feature_count_) {
    throw TrainingError("预测特征数量与树模型不一致");
  }

  std::vector<double> predictions(static_cast<std::size_t>(dataset.rows()));
  for (std::uint64_t row = 0; row < dataset.rows(); ++row) {
    std::uint32_t node_index = 0;
    std::size_t visited = 0;
    while (true) {
      if (node_index >= nodes_.size() || ++visited > nodes_.size()) {
        throw TrainingError("树节点索引越界或结构包含环");
      }
      const TreeNode& node = nodes_[node_index];
      if (node.IsLeaf()) {
        if (!std::isfinite(node.leaf_value)) {
          throw TrainingError("树叶值不是有限数");
        }
        predictions[static_cast<std::size_t>(row)] = node.leaf_value;
        break;
      }
      if (node.feature_index >= feature_count_) {
        throw TrainingError("树分支特征索引越界");
      }
      node_index = dataset.GetBin(row, node.feature_index) <= node.threshold_bin
                       ? node.left_child
                       : node.right_child;
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
