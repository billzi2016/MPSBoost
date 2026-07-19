// Parameter validation and node construction for native regression trees.
//
// Keeping these structural operations together prevents the growth loop from
// mixing policy checks with mutation of the flat node representation.

#include "tree_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace mpsboost::tree_internal {

void ValidateParameters(const TreeTrainingParameters& parameters) {
  if (parameters.min_samples_leaf == 0) {
    throw TrainingError("min_samples_leaf 必须至少为 1");
  }
  if (parameters.max_leaves == 1) {
    throw TrainingError("max_leaves must be zero or at least 2");
  }
  if (!std::isfinite(parameters.min_child_weight) ||
      parameters.min_child_weight < 0.0) {
    throw TrainingError("min_child_weight 必须是有限非负数");
  }
  if (!std::isfinite(parameters.reg_lambda) || parameters.reg_lambda < 0.0) {
    throw TrainingError("reg_lambda 必须是有限非负数");
  }
  if (!std::isfinite(parameters.gamma) || parameters.gamma < 0.0) {
    throw TrainingError("gamma 必须是有限非负数");
  }
}

std::uint32_t EffectiveMaxLeaves(const TreeTrainingParameters& parameters) {
  if (parameters.max_leaves != 0) {
    return parameters.max_leaves;
  }
  if (parameters.max_depth >= 31) {
    return std::numeric_limits<std::uint32_t>::max();
  }
  return std::uint32_t{1} << parameters.max_depth;
}

TreeNode MakeLeaf(const NodeStatistics& statistics, double reg_lambda) {
  TreeNode node;
  node.leaf_value =
      LeafWeight(statistics.gradient_sum, statistics.hessian_sum, reg_lambda);
  return node;
}

std::uint32_t AppendNode(std::vector<TreeNode>* nodes, TreeNode node) {
  if (nodes->size() >= kInvalidNodeIndex) {
    throw TrainingError("树节点数量超出 uint32 索引范围");
  }
  const auto index = static_cast<std::uint32_t>(nodes->size());
  nodes->push_back(std::move(node));
  return index;
}

RegressionTree TreeTrainingAccess::Create(std::uint32_t feature_count,
                                          const NodeStatistics& root_statistics,
                                          double reg_lambda) {
  RegressionTree tree;
  tree.feature_count_ = feature_count;
  tree.nodes_.reserve(1);
  AppendNode(&tree.nodes_, MakeLeaf(root_statistics, reg_lambda));
  return tree;
}

void TreeTrainingAccess::ApplySplit(RegressionTree* tree,
                                    const ActiveNode& active,
                                    const PreparedSplit& prepared,
                                    const TreeTrainingParameters& parameters,
                                    std::uint32_t* left_index,
                                    std::uint32_t* right_index) {
  if (!prepared.valid || left_index == nullptr || right_index == nullptr) {
    throw TrainingError("internal split application contract failed");
  }
  *left_index = AppendNode(&tree->nodes_,
                           MakeLeaf(prepared.split.left, parameters.reg_lambda));
  *right_index = AppendNode(
      &tree->nodes_, MakeLeaf(prepared.split.right, parameters.reg_lambda));
  TreeNode& parent = tree->nodes_[active.node_index];
  parent.feature_index = prepared.split.feature;
  parent.threshold_bin = prepared.split.threshold_bin;
  parent.left_child = *left_index;
  parent.right_child = *right_index;
  parent.leaf_value = 0.0;
  parent.gain = prepared.split.gain;
  parent.flags = 0;
}

void ValidateTreeStructure(std::uint32_t feature_count,
                           const std::vector<TreeNode>& nodes) {
  if (feature_count == 0 || nodes.empty() || nodes.size() >= kInvalidNodeIndex) {
    throw TrainingError("模型树的特征数或节点数不合法");
  }
  std::vector<std::uint8_t> state(nodes.size(), 0);
  const auto visit = [&](const auto& self, std::uint32_t index) -> void {
    if (index >= nodes.size()) {
      throw TrainingError("模型树的子节点索引越界");
    }
    if (state[index] == 1) {
      throw TrainingError("模型树结构包含环");
    }
    if (state[index] == 2) {
      return;
    }
    state[index] = 1;
    const TreeNode& node = nodes[index];
    if (node.IsLeaf()) {
      if (node.flags != kTreeNodeLeafFlag || !std::isfinite(node.leaf_value)) {
        throw TrainingError("模型树叶值不是有限数");
      }
    } else {
      if (node.flags != 0 || node.feature_index >= feature_count ||
          !std::isfinite(node.gain) || node.gain <= 0.0 ||
          node.left_child == node.right_child) {
        throw TrainingError("模型树分支字段不合法");
      }
      self(self, node.left_child);
      self(self, node.right_child);
    }
    state[index] = 2;
  };
  visit(visit, 0);
  if (std::find(state.begin(), state.end(), std::uint8_t{0}) != state.end()) {
    throw TrainingError("模型树包含不可达节点");
  }
}

}  // namespace mpsboost::tree_internal
