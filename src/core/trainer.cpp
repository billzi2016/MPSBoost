// MPSBoost 多轮 GBDT 训练状态机。
//
// 本文件是 boosting 顺序、base score、学习率缩放和模型预测的唯一实现。后端仅提供
// gradient/histogram 计算；本文件不判断设备、不访问文件系统，也不保留训练数据。

#include "mpsboost/trainer.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "mpsboost/backend.hpp"
#include "mpsboost/objective.hpp"
#include "trainer_internal.hpp"

namespace mpsboost {
namespace trainer_internal {

void ValidateTrainingParameters(const TrainingParameters& parameters) {
  if (parameters.n_estimators == 0) {
    throw TrainingError("n_estimators 必须至少为 1");
  }
  if (!std::isfinite(parameters.learning_rate) ||
      parameters.learning_rate <= 0.0 || parameters.learning_rate > 1.0) {
    throw TrainingError("learning_rate 必须位于 (0, 1]");
  }
  if (parameters.max_bins < 2 || parameters.max_bins > 65536) {
    throw TrainingError("max_bins 必须位于 [2, 65536]");
  }
}

double ValidateWeightsAndTotal(const std::vector<double>& labels,
                               const std::vector<double>& sample_weights) {
  if (labels.empty()) {
    throw TrainingError("labels must be non-empty");
  }
  if (labels.size() != sample_weights.size()) {
    throw TrainingError("labels and sample weights must have the same length");
  }
  double total_weight = 0.0;
  for (const double weight : sample_weights) {
    if (!std::isfinite(weight) || weight < 0.0) {
      throw TrainingError("sample weights must be finite non-negative values");
    }
    total_weight += weight;
    if (!std::isfinite(total_weight)) {
      throw TrainingError("sample weight sum overflowed");
    }
  }
  if (total_weight <= 0.0) {
    throw TrainingError("sample weights must contain positive total weight");
  }
  return total_weight;
}

double WeightedMeanLabel(const std::vector<double>& labels,
                         const std::vector<double>& sample_weights) {
  if (labels.empty()) {
    throw TrainingError("标签不能为空");
  }
  double sum = 0.0;
  const double total_weight = ValidateWeightsAndTotal(labels, sample_weights);
  for (std::size_t index = 0; index < labels.size(); ++index) {
    const double label = labels[index];
    if (!std::isfinite(label)) {
      throw TrainingError("标签必须是有限值");
    }
    sum += label * sample_weights[index];
    if (!std::isfinite(sum)) {
      throw TrainingError("标签累计发生浮点溢出");
    }
  }
  return sum / total_weight;
}

double WeightedBinaryLogitBaseScore(const std::vector<double>& labels,
                                    const std::vector<double>& sample_weights) {
  if (labels.empty()) {
    throw TrainingError("labels must be non-empty");
  }
  const double total_weight = ValidateWeightsAndTotal(labels, sample_weights);
  double positive_weight = 0.0;
  for (std::size_t index = 0; index < labels.size(); ++index) {
    const double label = labels[index];
    if (label == 0.0) {
      continue;
    }
    if (label == 1.0) {
      positive_weight += sample_weights[index];
      continue;
    }
    throw TrainingError("binary-logistic labels must be 0 or 1");
  }
  const double epsilon = 1e-15;
  double probability = positive_weight / total_weight;
  probability = std::min(1.0 - epsilon, std::max(epsilon, probability));
  return std::log(probability / (1.0 - probability));
}

double InitialBaseScore(const std::vector<double>& labels,
                        const std::vector<double>& sample_weights,
                        TrainingParameters::Objective objective) {
  switch (objective) {
    case TrainingParameters::Objective::kSquaredError:
      return WeightedMeanLabel(labels, sample_weights);
    case TrainingParameters::Objective::kBinaryLogistic:
      return WeightedBinaryLogitBaseScore(labels, sample_weights);
  }
  throw TrainingError("unknown training objective");
}

std::vector<GradientPair> ComputeObjectiveGradients(
    const std::vector<double>& labels,
    const std::vector<double>& predictions,
    const GradientComputer& gradient_computer,
    TrainingParameters::Objective objective) {
  switch (objective) {
    case TrainingParameters::Objective::kSquaredError:
      return gradient_computer.ComputeSquaredError(labels, predictions);
    case TrainingParameters::Objective::kBinaryLogistic:
      return ComputeBinaryLogisticGradients(labels, predictions);
  }
  throw TrainingError("unknown training objective");
}

std::vector<GradientPair> ApplySampleWeights(
    std::vector<GradientPair> gradients,
    const std::vector<double>& sample_weights) {
  if (gradients.size() != sample_weights.size()) {
    throw TrainingError("gradient and sample weight lengths do not match");
  }
  for (std::size_t index = 0; index < gradients.size(); ++index) {
    gradients[index].gradient *= sample_weights[index];
    gradients[index].hessian *= sample_weights[index];
    if (!std::isfinite(gradients[index].gradient) ||
        !std::isfinite(gradients[index].hessian) ||
        gradients[index].hessian < 0.0) {
      throw TrainingError("weighted gradient/Hessian overflowed");
    }
  }
  return gradients;
}

}  // namespace trainer_internal

using trainer_internal::ApplySampleWeights;
using trainer_internal::ComputeObjectiveGradients;
using trainer_internal::InitialBaseScore;
using trainer_internal::ValidateTrainingParameters;
using trainer_internal::ValidateWeightsAndTotal;

RegressionModel TrainRegressionModel(
    const BinnedDataset& dataset,
    const std::vector<double>& labels,
    const std::vector<double>& sample_weights,
    const TrainingParameters& parameters,
    const GradientComputer& gradient_computer,
    const HistogramBuilder& histogram_builder) {
  ValidateTrainingParameters(parameters);
  if (dataset.rows() != labels.size() || dataset.max_bins() != parameters.max_bins) {
    throw TrainingError("训练数据、标签或 max_bins 契约不一致");
  }
  ValidateWeightsAndTotal(labels, sample_weights);

  RegressionModel model;
  model.schema_ = dataset.schema();
  model.base_score_ = InitialBaseScore(labels, sample_weights, parameters.objective);
  model.learning_rate_ = parameters.learning_rate;
  model.objective_ = parameters.objective;
  model.trees_.reserve(parameters.n_estimators);
  std::vector<double> predictions(labels.size(), model.base_score_);

  for (std::uint32_t round = 0; round < parameters.n_estimators; ++round) {
    const std::vector<GradientPair> gradients = ApplySampleWeights(
        ComputeObjectiveGradients(labels, predictions, gradient_computer,
                                  parameters.objective),
        sample_weights);
    RegressionTree tree = TrainSingleRegressionTree(
        dataset, gradients, parameters.tree, histogram_builder);
    const std::vector<double> update = tree.Predict(dataset);
    for (std::size_t row = 0; row < predictions.size(); ++row) {
      predictions[row] += parameters.learning_rate * update[row];
      if (!std::isfinite(predictions[row])) {
        throw TrainingError("Boosting prediction update 发生浮点溢出");
      }
    }
    model.trees_.push_back(std::move(tree));
  }
  return model;
}

std::vector<double> RegressionModel::Predict(const BinnedDataset& dataset) const {
  if (dataset.features() != schema_.features() ||
      dataset.max_bins() != schema_.max_bins() ||
      dataset.boundaries() != schema_.boundaries()) {
    throw TrainingError("预测数据未使用模型冻结的分箱 schema");
  }
  std::vector<double> result(static_cast<std::size_t>(dataset.rows()), base_score_);
  for (const RegressionTree& tree : trees_) {
    const std::vector<double> update = tree.Predict(dataset);
    for (std::size_t row = 0; row < result.size(); ++row) {
      result[row] += learning_rate_ * update[row];
    }
  }
  return result;
}

RegressionModel RegressionModel::Restore(QuantizationSchema schema,
                                         double base_score,
                                         double learning_rate,
                                         TrainingParameters::Objective objective,
                                         std::vector<RegressionTree> trees) {
  if (!std::isfinite(base_score) || !std::isfinite(learning_rate) ||
      learning_rate <= 0.0 || learning_rate > 1.0 || trees.empty()) {
    throw TrainingError("模型 base score、learning rate 或树数量不合法");
  }
  for (const RegressionTree& tree : trees) {
    if (tree.feature_count() != schema.features()) {
      throw TrainingError("模型树特征数与分箱 schema 不一致");
    }
    for (const TreeNode& node : tree.nodes()) {
      if (!node.IsLeaf() &&
          node.threshold_bin >=
              schema.feature_metadata()[node.feature_index].bin_count - 1) {
        throw TrainingError("模型树切分阈值超出特征分箱范围");
      }
      if (node.IsLeaf() && !node.default_left) {
        throw TrainingError("model leaf nodes must keep default_left true");
      }
    }
  }
  RegressionModel model;
  model.schema_ = std::move(schema);
  model.base_score_ = base_score;
  model.learning_rate_ = learning_rate;
  model.objective_ = objective;
  model.trees_ = std::move(trees);
  return model;
}

}  // namespace mpsboost
