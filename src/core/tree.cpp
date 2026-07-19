// MPSBoost 深度受限单树的唯一设备无关实现。
//
// 本文件负责按层生长、稳定 split 选择、样本分区、扁平节点组装和预测遍历。Histogram
// 计算通过最小后端接口注入，因此 CPU 与 MPS 不会形成两套树控制流或参数语义。

#include "mpsboost/tree.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "mpsboost/backend.hpp"
#include "mpsboost/binned_dataset.hpp"

namespace mpsboost {
namespace {

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

void ValidateParameters(const TreeTrainingParameters& parameters) {
  if (parameters.min_samples_leaf == 0) {
    throw TrainingError("min_samples_leaf 必须至少为 1");
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

std::uint32_t StableRandomThreshold(std::uint32_t seed,
                                    std::uint32_t feature,
                                    std::uint32_t node_index,
                                    std::uint32_t depth,
                                    std::uint32_t candidate_count) {
  if (candidate_count == 0) {
    throw TrainingError("random threshold candidate count must be positive");
  }
  std::uint64_t value = seed;
  value ^= (static_cast<std::uint64_t>(feature) + 0x9E3779B97F4A7C15ULL);
  value *= 0xBF58476D1CE4E5B9ULL;
  value ^= (static_cast<std::uint64_t>(node_index) << 32U) | depth;
  value *= 0x94D049BB133111EBULL;
  value ^= value >> 31U;
  return static_cast<std::uint32_t>(value % candidate_count);
}

NodeStatistics SumRows(const std::vector<std::uint64_t>& rows,
                       const std::vector<GradientPair>& gradients) {
  NodeStatistics result;
  for (const std::uint64_t row : rows) {
    if (row >= gradients.size()) {
      throw TrainingError("节点统计行索引越界");
    }
    const GradientPair& pair = gradients[static_cast<std::size_t>(row)];
    if (!std::isfinite(pair.gradient) || !std::isfinite(pair.hessian) ||
        pair.hessian < 0.0) {
      throw TrainingError("Gradient/Hessian 必须有限且 Hessian 非负");
    }
    ++result.count;
    result.gradient_sum += pair.gradient;
    result.hessian_sum += pair.hessian;
    if (!std::isfinite(result.gradient_sum) ||
        !std::isfinite(result.hessian_sum)) {
      throw TrainingError("节点 FP64 统计发生溢出");
    }
  }
  return result;
}

bool IsBetterSplit(const SplitCandidate& candidate,
                   const SplitCandidate& incumbent) {
  if (!incumbent.valid) {
    return true;
  }
  // 不使用 epsilon 合并“接近”增益，因为 epsilon 会让结果随数据尺度变化。FP64 值
  // 严格较大时获胜；完全相等才依次比较 feature 与 threshold，冻结确定性顺序。
  if (candidate.gain != incumbent.gain) {
    return candidate.gain > incumbent.gain;
  }
  if (candidate.feature != incumbent.feature) {
    return candidate.feature < incumbent.feature;
  }
  return candidate.threshold_bin < incumbent.threshold_bin;
}

SplitCandidate FindBestSplit(const NodeHistograms& histograms,
                             const BinnedDataset& dataset,
                             const NodeStatistics& parent,
                             std::uint32_t node_index,
                             std::uint32_t depth,
                             const TreeTrainingParameters& parameters) {
  if (histograms.size() != dataset.features()) {
    throw TrainingError("Histogram 特征数量与数据集不一致");
  }
  SplitCandidate best;
  for (std::uint32_t feature = 0; feature < dataset.features(); ++feature) {
    const FeatureHistogram& bins = histograms[feature];
    if (bins.size() != dataset.feature_metadata()[feature].bin_count) {
      throw TrainingError("Histogram bin 数量与特征元数据不一致");
    }
    std::uint64_t histogram_count = 0;
    for (const HistogramBin& bin : bins) {
      if (bin.count > std::numeric_limits<std::uint64_t>::max() -
                          histogram_count) {
        throw TrainingError("Histogram 样本计数溢出");
      }
      histogram_count += bin.count;
      if (!std::isfinite(bin.gradient_sum) ||
          !std::isfinite(bin.hessian_sum) || bin.hessian_sum < 0.0) {
        throw TrainingError("Histogram G/H 必须有限且 Hessian 非负");
      }
    }
    if (histogram_count != parent.count) {
      throw TrainingError("Histogram 样本计数与节点统计不一致");
    }
    if (bins.size() < 2) {
      continue;
    }
    NodeStatistics left;
    for (std::uint32_t threshold = 0; threshold + 1 < bins.size(); ++threshold) {
      const HistogramBin& bin = bins[threshold];
      left.count += bin.count;
      left.gradient_sum += bin.gradient_sum;
      left.hessian_sum += bin.hessian_sum;
      if (!std::isfinite(left.gradient_sum) ||
          !std::isfinite(left.hessian_sum)) {
        throw TrainingError("Histogram 前缀累计发生浮点溢出");
      }
      const NodeStatistics right{
          parent.count - left.count,
          parent.gradient_sum - left.gradient_sum,
          parent.hessian_sum - left.hessian_sum,
      };
      if (left.count < parameters.min_samples_leaf ||
          right.count < parameters.min_samples_leaf ||
          left.hessian_sum < parameters.min_child_weight ||
          right.hessian_sum < parameters.min_child_weight ||
          left.hessian_sum <= 0.0 || right.hessian_sum <= 0.0) {
        continue;
      }
      if (parameters.split_strategy ==
              TreeTrainingParameters::SplitStrategy::kRandomThreshold &&
          threshold != StableRandomThreshold(parameters.random_seed, feature,
                                             node_index, depth,
                                             static_cast<std::uint32_t>(bins.size() - 1))) {
        continue;
      }
      const double gain = SplitGain(
          left.gradient_sum, left.hessian_sum, right.gradient_sum,
          right.hessian_sum, parameters.reg_lambda, parameters.gamma);
      // 零或负增益不能改善目标，因此节点保持叶子。这个严格条件同时避免在全常量
      // 统计上创建无意义结构，不能改成 >= 或带任意容差的比较。
      if (gain <= 0.0) {
        continue;
      }
      const SplitCandidate candidate{
          true,
          feature,
          threshold,
          gain,
          left,
          right,
      };
      if (IsBetterSplit(candidate, best)) {
        best = candidate;
      }
    }
  }
  return best;
}

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
      feature_histogram.push_back(HistogramBin{
          parent_bin.count - child_bin.count,
          parent_bin.gradient_sum - child_bin.gradient_sum,
          parent_bin.hessian_sum - child_bin.hessian_sum});
    }
    result.push_back(std::move(feature_histogram));
  }
  return result;
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
    return layer_builder->BuildLayerHistograms(dataset, rows_to_build, gradients);
  }
  std::vector<NodeHistograms> result;
  result.reserve(rows_to_build.size());
  for (const std::vector<std::uint64_t>& rows : rows_to_build) {
    result.push_back(histogram_builder.BuildHistograms(dataset, rows, gradients));
  }
  return result;
}

}  // namespace

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
        // 分箱采用 lower_bound 语义，因此等于原始边界的值位于较小 bin；预测与训练
        // 统一使用 bin <= threshold 进入左子树，避免边界样本在两条路径间漂移。
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
  RegressionTree tree;
  tree.feature_count_ = feature_count;
  tree.nodes_ = std::move(nodes);
  return tree;
}

}  // namespace mpsboost
