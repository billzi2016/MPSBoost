// Internal contracts shared by the native tree implementation units.
//
// This header is intentionally private to src/core. It keeps training state and
// helper boundaries out of the public API while allowing each implementation
// unit to own one coherent responsibility.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mpsboost/backend.hpp"
#include "mpsboost/binned_dataset.hpp"
#include "mpsboost/tree.hpp"

namespace mpsboost::tree_internal {

struct NodeStatistics final {
  std::uint64_t count{0};
  double gradient_sum{0.0};
  double hessian_sum{0.0};
};

struct SplitCandidate final {
  bool valid{false};
  std::uint32_t feature{0};
  std::uint32_t threshold_bin{0};
  double gain{0.0};
  NodeStatistics left;
  NodeStatistics right;
};

struct ActiveNode final {
  std::uint32_t node_index{0};
  std::uint32_t depth{0};
  std::vector<std::uint64_t> rows;
  NodeStatistics statistics;
  NodeHistograms cached_histograms;
};

struct PendingChildHistogram final {
  std::size_t next_layer_index{0};
  std::vector<std::uint64_t> rows;
  NodeHistograms parent_histograms;
};

struct PreparedSplit final {
  bool valid{false};
  SplitCandidate split;
  std::vector<std::uint64_t> left_rows;
  std::vector<std::uint64_t> right_rows;
};

class TreeTrainingAccess final {
 public:
  static RegressionTree Create(std::uint32_t feature_count,
                               const NodeStatistics& root_statistics,
                               double reg_lambda);
  static void ApplySplit(RegressionTree* tree,
                         const ActiveNode& active,
                         const PreparedSplit& prepared,
                         const TreeTrainingParameters& parameters,
                         std::uint32_t* left_index,
                         std::uint32_t* right_index);
};

void ValidateParameters(const TreeTrainingParameters& parameters);

NodeStatistics SumRows(const std::vector<std::uint64_t>& rows,
                       const std::vector<GradientPair>& gradients);

SplitCandidate FindBestSplit(const NodeHistograms& histograms,
                             const BinnedDataset& dataset,
                             const NodeStatistics& parent,
                             std::uint32_t node_index,
                             std::uint32_t depth,
                             const TreeTrainingParameters& parameters);

NodeHistograms SubtractHistograms(const NodeHistograms& parent,
                                  const NodeHistograms& child);

TreeNode MakeLeaf(const NodeStatistics& statistics, double reg_lambda);

std::uint32_t AppendNode(std::vector<TreeNode>* nodes, TreeNode node);

void ValidateTreeStructure(std::uint32_t feature_count,
                           const std::vector<TreeNode>& nodes);

std::uint32_t EffectiveMaxLeaves(const TreeTrainingParameters& parameters);

std::vector<NodeHistograms> BuildCurrentLayerHistograms(
    const BinnedDataset& dataset,
    const std::vector<ActiveNode>& current_layer,
    const std::vector<GradientPair>& gradients,
    const HistogramBuilder& histogram_builder);

std::vector<NodeHistograms> BuildPendingChildHistograms(
    const BinnedDataset& dataset,
    const std::vector<PendingChildHistogram>& pending,
    const std::vector<GradientPair>& gradients,
    const HistogramBuilder& histogram_builder);

PreparedSplit PrepareSplitRows(const BinnedDataset& dataset,
                               const ActiveNode& active,
                               const NodeHistograms& histograms,
                               const TreeTrainingParameters& parameters);

RegressionTree TrainLeafWiseRegressionTree(
    const BinnedDataset& dataset,
    const std::vector<GradientPair>& gradients,
    const TreeTrainingParameters& parameters,
    const HistogramBuilder& histogram_builder);

}  // namespace mpsboost::tree_internal
