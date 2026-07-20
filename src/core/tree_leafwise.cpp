// Leaf-wise native regression tree growth strategy.
//
// This unit owns the controlled LightGBM-like growth loop. Split scoring,
// histogram construction, and flat-tree mutation still use the shared helpers
// so leaf-wise and level-wise training keep one numerical contract.

#include "tree_internal.hpp"

#include <cstddef>
#include <limits>
#include <utility>

namespace mpsboost::tree_internal {
namespace {

bool IsBetterLeafWiseCandidate(const PreparedSplit& candidate,
                               const ActiveNode& candidate_node,
                               const PreparedSplit& incumbent,
                               const ActiveNode& incumbent_node) {
  if (!incumbent.valid) {
    return true;
  }
  if (candidate.split.gain != incumbent.split.gain) {
    return candidate.split.gain > incumbent.split.gain;
  }
  return candidate_node.node_index < incumbent_node.node_index;
}

void CacheChildHistograms(const BinnedDataset& dataset,
                          const std::vector<GradientPair>& gradients,
                          const HistogramBuilder& histogram_builder,
                          const TreeTrainingParameters& parameters,
                          const PreparedSplit& prepared,
                          const NodeHistograms& parent_histograms,
                          ActiveNode* left,
                          ActiveNode* right) {
  if (left == nullptr || right == nullptr ||
      left->depth >= parameters.max_depth) {
    return;
  }
  if (prepared.split.left.count <= prepared.split.right.count) {
    left->cached_histograms =
        histogram_builder.BuildHistograms(dataset, left->rows, gradients);
    right->cached_histograms =
        SubtractHistograms(parent_histograms, left->cached_histograms);
  } else {
    right->cached_histograms =
        histogram_builder.BuildHistograms(dataset, right->rows, gradients);
    left->cached_histograms =
        SubtractHistograms(parent_histograms, right->cached_histograms);
  }
}

}  // namespace

RegressionTree TrainLeafWiseRegressionTree(
    const BinnedDataset& dataset,
    const std::vector<GradientPair>& gradients,
    const TreeTrainingParameters& parameters,
    const HistogramBuilder& histogram_builder) {
  std::vector<std::uint64_t> root_rows(static_cast<std::size_t>(dataset.rows()));
  for (std::uint64_t row = 0; row < dataset.rows(); ++row) {
    root_rows[static_cast<std::size_t>(row)] = row;
  }
  const NodeStatistics root_statistics = SumRows(root_rows, gradients);
  RegressionTree tree = TreeTrainingAccess::Create(
      dataset.features(), root_statistics, parameters);

  std::vector<ActiveNode> active_leaves;
  active_leaves.push_back(
      ActiveNode{0, 0, std::move(root_rows), root_statistics, {},
                 -std::numeric_limits<double>::infinity(),
                 std::numeric_limits<double>::infinity(), {}});
  std::uint32_t leaf_count = 1;
  const std::uint32_t max_leaves = EffectiveMaxLeaves(parameters);
  const std::uint32_t max_active_leaves = EffectiveMaxActiveLeaves(parameters);

  while (!active_leaves.empty() && leaf_count < max_leaves) {
    if (active_leaves.size() > max_active_leaves) {
      throw TrainingError("active leaf queue exceeded max_active_leaves");
    }
    std::size_t best_index = 0;
    PreparedSplit best_prepared;
    NodeHistograms best_histograms;
    for (std::size_t index = 0; index < active_leaves.size(); ++index) {
      ActiveNode& active = active_leaves[index];
      if (active.depth >= parameters.max_depth) {
        continue;
      }
      NodeHistograms histograms =
          active.cached_histograms.empty()
              ? histogram_builder.BuildHistograms(dataset, active.rows, gradients)
              : std::move(active.cached_histograms);
      PreparedSplit prepared =
          PrepareSplitRows(dataset, active, histograms, gradients, parameters);
      if (prepared.valid &&
          IsBetterLeafWiseCandidate(prepared, active, best_prepared,
                                    active_leaves[best_index])) {
        best_index = index;
        best_prepared = std::move(prepared);
        best_histograms = std::move(histograms);
      } else {
        active.cached_histograms = std::move(histograms);
      }
    }
    if (!best_prepared.valid) {
      break;
    }

    ActiveNode active = std::move(active_leaves[best_index]);
    active_leaves.erase(active_leaves.begin() +
                        static_cast<std::ptrdiff_t>(best_index));
    std::uint32_t left_index = 0;
    std::uint32_t right_index = 0;
    TreeTrainingAccess::ApplySplit(&tree, active, best_prepared, parameters,
                                   &left_index, &right_index);
    ++leaf_count;

    const std::uint32_t child_depth = active.depth + 1;
    const std::vector<std::uint32_t> child_path =
        ExtendInteractionPath(active.path_features, best_prepared.split.feature);
    ActiveNode left{left_index, child_depth, std::move(best_prepared.left_rows),
                    best_prepared.split.left, {},
                    best_prepared.split.left_lower_bound,
                    best_prepared.split.left_upper_bound, child_path};
    ActiveNode right{right_index, child_depth, std::move(best_prepared.right_rows),
                     best_prepared.split.right, {},
                     best_prepared.split.right_lower_bound,
                     best_prepared.split.right_upper_bound, child_path};
    CacheChildHistograms(dataset, gradients, histogram_builder, parameters,
                         best_prepared,
                         best_histograms, &left, &right);
    const bool can_add_both =
        active_leaves.size() + 2 <= max_active_leaves || leaf_count >= max_leaves;
    if (!can_add_both) {
      throw TrainingError("leaf-wise active queue reached max_active_leaves");
    }
    active_leaves.push_back(std::move(left));
    active_leaves.push_back(std::move(right));
  }
  return tree;
}

}  // namespace mpsboost::tree_internal
