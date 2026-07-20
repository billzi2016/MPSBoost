// MPSBoost 单树领域模型与设备无关训练契约。
//
// 职责：定义紧凑扁平节点、唯一按层生长入口和确定性预测。实现只消费分箱数据、
// gradient/Hessian 与 histogram 抽象，不依赖 Python、Metal、缓存或文件系统。
#pragma once

#include <cstdint>
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

// 扁平树节点。分支使用 feature_index/threshold_bin/children/gain；叶节点只使用
// leaf_value。flags 明确区分叶子，禁止用 NaN、负索引等特殊值编码节点类型。
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

// 单树训练参数。min_child_weight 约束子节点 Hessian 和；平方误差下 Hessian 恒为 1，
// 但保留独立字段才能让后续目标函数不改变树生长契约。
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
  double gamma{0.0};
  double min_gain_to_split{0.0};
  SplitStrategy split_strategy{SplitStrategy::kBestGain};
  GrowthStrategy growth_strategy{GrowthStrategy::kLevelWise};
  std::uint32_t random_seed{0};
};

class RegressionTree final {
 public:
  std::uint32_t feature_count() const noexcept { return feature_count_; }
  const std::vector<TreeNode>& nodes() const noexcept { return nodes_; }

  // 对已量化数据执行确定性逐行预测。特征数量或内部索引不一致时明确失败；本函数
  // 不修改树或数据集，因此同一模型可被多个只读调用方安全共享。
  std::vector<double> Predict(const BinnedDataset& dataset) const;

  // 从模型文件字段恢复扁平树。该入口完整验证根节点、索引、环、可达性和叶值，只有
  // 通过验证的树才能进入 RegressionModel，loader 不能直接写私有节点。
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

// 使用唯一按层策略训练一棵深度受限回归树。训练核心负责控制流和稳定选 split，
// HistogramBuilder 只负责统计计算；异常时局部结果销毁，不返回部分模型。
RegressionTree TrainSingleRegressionTree(
    const BinnedDataset& dataset,
    const std::vector<GradientPair>& gradients,
    const TreeTrainingParameters& parameters,
    const HistogramBuilder& histogram_builder);

}  // namespace mpsboost
