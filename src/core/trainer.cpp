// MPSBoost multi-round GBDT training state machine.
//
// This file is the single implementation of boosting order, base score, learning-rate
// scaling, and model prediction. Backends provide only gradient/histogram computation;
// this file does not select devices, access the file system, or retain training data.

#include "mpsboost/trainer.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <utility>

#include "mpsboost/backend.hpp"
#include "mpsboost/objective.hpp"
#include "trainer_internal.hpp"

namespace mpsboost {
namespace trainer_internal {

void ValidateTrainingParameters(const TrainingParameters& parameters) {
  if (parameters.n_estimators == 0) {
    throw TrainingError("n_estimators must be at least 1");
  }
  if (!std::isfinite(parameters.learning_rate) ||
      parameters.learning_rate <= 0.0 || parameters.learning_rate > 1.0) {
    throw TrainingError("learning_rate must be in (0, 1]");
  }
  if (parameters.max_bins < 2 || parameters.max_bins > 65536) {
    throw TrainingError("max_bins must be in [2, 65536]");
  }
  if (!std::isfinite(parameters.objective_alpha) ||
      parameters.objective_alpha <= 0.0 || parameters.objective_alpha >= 1.0) {
    throw TrainingError("quantile alpha must be in (0, 1)");
  }
  if (!std::isfinite(parameters.tweedie_variance_power) ||
      parameters.tweedie_variance_power <= 1.0 ||
      parameters.tweedie_variance_power >= 2.0) {
    throw TrainingError("tweedie variance power must be in (1, 2)");
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
    throw TrainingError("Labels must not be empty");
  }
  double sum = 0.0;
  const double total_weight = ValidateWeightsAndTotal(labels, sample_weights);
  for (std::size_t index = 0; index < labels.size(); ++index) {
    const double label = labels[index];
    if (!std::isfinite(label)) {
      throw TrainingError("Labels must be finite");
    }
    sum += label * sample_weights[index];
    if (!std::isfinite(sum)) {
      throw TrainingError("Label accumulation overflowed");
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

double WeightedQuantileBaseScore(const std::vector<double>& labels,
                                 const std::vector<double>& sample_weights,
                                 double alpha) {
  const double total_weight = ValidateWeightsAndTotal(labels, sample_weights);
  std::vector<std::pair<double, double>> ordered;
  ordered.reserve(labels.size());
  for (std::size_t index = 0; index < labels.size(); ++index) {
    const double label = labels[index];
    if (!std::isfinite(label)) {
      throw TrainingError("label must be finite");
    }
    ordered.push_back({label, sample_weights[index]});
  }
  std::sort(ordered.begin(), ordered.end(),
            [](const auto& left, const auto& right) {
              return left.first < right.first;
            });
  const double threshold = alpha * total_weight;
  double cumulative = 0.0;
  for (const auto& item : ordered) {
    cumulative += item.second;
    if (cumulative >= threshold) {
      return item.first;
    }
  }
  return ordered.back().first;
}

double WeightedLogMeanBaseScore(const std::vector<double>& labels,
                                const std::vector<double>& sample_weights,
                                const char* objective) {
  const double total_weight = ValidateWeightsAndTotal(labels, sample_weights);
  double weighted_sum = 0.0;
  for (std::size_t index = 0; index < labels.size(); ++index) {
    const double label = labels[index];
    if (!std::isfinite(label) || label < 0.0) {
      throw TrainingError(std::string(objective) + " labels must be non-negative");
    }
    weighted_sum += label * sample_weights[index];
    if (!std::isfinite(weighted_sum)) {
      throw TrainingError(std::string(objective) + " label sum overflowed");
    }
  }
  constexpr double kEpsilon = 1e-15;
  return std::log(std::max(kEpsilon, weighted_sum / total_weight));
}

double InitialBaseScore(const std::vector<double>& labels,
                        const std::vector<double>& sample_weights,
                        const TrainingParameters& parameters) {
  switch (parameters.objective) {
    case TrainingParameters::Objective::kSquaredError:
      return WeightedMeanLabel(labels, sample_weights);
    case TrainingParameters::Objective::kBinaryLogistic:
      return WeightedBinaryLogitBaseScore(labels, sample_weights);
    case TrainingParameters::Objective::kQuantile:
      return WeightedQuantileBaseScore(labels, sample_weights,
                                       parameters.objective_alpha);
    case TrainingParameters::Objective::kPoisson:
      return WeightedLogMeanBaseScore(labels, sample_weights, "poisson");
    case TrainingParameters::Objective::kTweedie:
      return WeightedLogMeanBaseScore(labels, sample_weights, "tweedie");
  }
  throw TrainingError("unknown training objective");
}

std::vector<GradientPair> ComputeObjectiveGradients(
    const std::vector<double>& labels,
    const std::vector<double>& predictions,
    const GradientComputer& gradient_computer,
    const TrainingParameters& parameters) {
  switch (parameters.objective) {
    case TrainingParameters::Objective::kSquaredError:
      return gradient_computer.ComputeSquaredError(labels, predictions);
    case TrainingParameters::Objective::kBinaryLogistic:
      return ComputeBinaryLogisticGradients(labels, predictions);
    case TrainingParameters::Objective::kQuantile:
      return ComputeQuantileGradients(labels, predictions,
                                      parameters.objective_alpha);
    case TrainingParameters::Objective::kPoisson:
      return ComputePoissonGradients(labels, predictions);
    case TrainingParameters::Objective::kTweedie:
      return ComputeTweedieGradients(labels, predictions,
                                     parameters.tweedie_variance_power);
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
    throw TrainingError("Training data, labels, or max_bins contract does not match");
  }
  ValidateWeightsAndTotal(labels, sample_weights);

  RegressionModel model;
  model.schema_ = dataset.schema();
  model.base_score_ = InitialBaseScore(labels, sample_weights, parameters);
  model.learning_rate_ = parameters.learning_rate;
  model.objective_ = parameters.objective;
  model.objective_alpha_ = parameters.objective_alpha;
  model.tweedie_variance_power_ = parameters.tweedie_variance_power;
  model.trees_.reserve(parameters.n_estimators);
  std::vector<double> predictions(labels.size(), model.base_score_);

  for (std::uint32_t round = 0; round < parameters.n_estimators; ++round) {
    const std::vector<GradientPair> gradients = ApplySampleWeights(
        ComputeObjectiveGradients(labels, predictions, gradient_computer,
                                  parameters),
        sample_weights);
    RegressionTree tree = TrainSingleRegressionTree(
        dataset, gradients, parameters.tree, histogram_builder);
    const std::vector<double> update = tree.Predict(dataset);
    for (std::size_t row = 0; row < predictions.size(); ++row) {
      predictions[row] += parameters.learning_rate * update[row];
      if (!std::isfinite(predictions[row])) {
        throw TrainingError("Boosting prediction update overflowed");
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
    throw TrainingError("Prediction data does not use the model's frozen binned schema");
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
                                         double objective_alpha,
                                         double tweedie_variance_power,
                                         std::vector<RegressionTree> trees) {
  if (!std::isfinite(base_score) || !std::isfinite(learning_rate) ||
      learning_rate <= 0.0 || learning_rate > 1.0 || trees.empty()) {
    throw TrainingError("Model base score, learning rate, or tree count is invalid");
  }
  if (!std::isfinite(objective_alpha) || objective_alpha <= 0.0 ||
      objective_alpha >= 1.0) {
    throw TrainingError("model quantile alpha is invalid");
  }
  if (!std::isfinite(tweedie_variance_power) ||
      tweedie_variance_power <= 1.0 || tweedie_variance_power >= 2.0) {
    throw TrainingError("model tweedie variance power is invalid");
  }
  for (const RegressionTree& tree : trees) {
    if (tree.feature_count() != schema.features()) {
      throw TrainingError("Model tree feature count does not match the binned schema");
    }
    for (const TreeNode& node : tree.nodes()) {
      if (!node.IsLeaf() &&
          node.threshold_bin >=
              schema.feature_metadata()[node.feature_index].bin_count - 1) {
        throw TrainingError("Model tree split threshold exceeds the feature bin range");
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
  model.objective_alpha_ = objective_alpha;
  model.tweedie_variance_power_ = tweedie_variance_power;
  model.trees_ = std::move(trees);
  return model;
}

}  // namespace mpsboost
