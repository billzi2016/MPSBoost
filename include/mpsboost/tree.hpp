// MPSBoost single-tree domain model and device-independent training contract.
//
// Responsibility: defines compact flat nodes, the sole level-wise growth entry,
// and deterministic prediction. Implementations consume only binned data,
// gradient/Hessian, and histogram abstractions, not Python, Metal, caches, or files.
#pragma once

#include <cstdint>
#include <cstddef>
#include <limits>
#include <vector>

#include "mpsboost/objective.hpp"

namespace mpsboost {

class BinnedDataset;
class HistogramBuilder;
class LayerHistogramBuilder;
namespace tree_internal {
class TreeTrainingAccess;
}

constexpr std::uint32_t kInvalidNodeIndex =
    std::numeric_limits<std::uint32_t>::max();
constexpr std::uint8_t kTreeNodeLeafFlag = 1U;

// Flat tree node. Branches use feature_index/threshold_bin/children/gain; leaves
// use only leaf_value. flags explicitly identify leaves; do not encode node type
// with special values such as NaN or negative indices.
struct TreeNode final {
  std::uint32_t feature_index{0};
  std::uint32_t threshold_bin{0};
  std::uint32_t left_child{kInvalidNodeIndex};
  std::uint32_t right_child{kInvalidNodeIndex};
  double leaf_value{0.0};
  double gain{0.0};
  bool default_left{true};
  std::uint8_t flags{kTreeNodeLeafFlag};

  bool IsLeaf() const noexcept { return (flags & kTreeNodeLeafFlag) != 0; }
};

// Single-tree training parameters. min_child_weight constrains child Hessian sums.
// Squared error has Hessian 1, but a separate field preserves the growth contract
// for later objectives.
struct TreeTrainingParameters final {
  enum class SplitStrategy : std::uint32_t {
    kBestGain = 0,
    kRandomThreshold = 1,
  };
  enum class GrowthStrategy : std::uint32_t {
    kLevelWise = 0,
    kLeafWise = 1,
  };

  std::uint32_t max_depth{6};
  std::uint32_t max_leaves{0};
  std::uint32_t max_active_leaves{0};
  std::uint64_t min_samples_leaf{1};
  double min_child_weight{0.0};
  double reg_lambda{1.0};
  double reg_alpha{0.0};
  double max_delta_step{0.0};
  double gamma{0.0};
  double min_gain_to_split{0.0};
  SplitStrategy split_strategy{SplitStrategy::kBestGain};
  GrowthStrategy growth_strategy{GrowthStrategy::kLevelWise};
  std::uint32_t random_seed{0};
  std::vector<std::int8_t> monotonic_constraints;
  std::vector<std::vector<std::uint32_t>> interaction_constraints;
};

class RegressionTree final {
 public:
  std::uint32_t feature_count() const noexcept { return feature_count_; }
  const std::vector<TreeNode>& nodes() const noexcept { return nodes_; }

  // Predict deterministically row by row over quantized data. Feature-count or
  // internal-index mismatches fail explicitly. This function does not mutate the
  // tree or dataset, so multiple read-only callers can safely share one model.
  std::vector<double> Predict(const BinnedDataset& dataset) const;

  // Restore a flat tree from model-file fields. This entry validates root, indices,
  // cycles, reachability, and leaf values; only validated trees enter RegressionModel.
  static RegressionTree Restore(std::uint32_t feature_count,
                                std::vector<TreeNode> nodes);

 private:
  friend class tree_internal::TreeTrainingAccess;
  friend RegressionTree TrainSingleRegressionTree(
      const BinnedDataset&,
      const std::vector<GradientPair>&,
      const TreeTrainingParameters&,
      const HistogramBuilder&);

  std::uint32_t feature_count_{0};
  std::vector<TreeNode> nodes_;
};

// Train one depth-limited regression tree with the sole level-wise strategy. The
// training core owns control flow and stable split selection; HistogramBuilder only
// computes statistics. Exceptions discard local results and never return a partial model.
RegressionTree TrainSingleRegressionTree(
    const BinnedDataset& dataset,
    const std::vector<GradientPair>& gradients,
    const TreeTrainingParameters& parameters,
    const HistogramBuilder& histogram_builder);

}  // namespace mpsboost
