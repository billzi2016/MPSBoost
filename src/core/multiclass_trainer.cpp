// Native CPU-oracle multiclass softmax boosting.
//
// This unit owns only the multiclass training loop and prediction aggregation.
// Shared validation, sample-weight application, and tree growth stay in the
// regression trainer and tree modules so objective semantics do not fork.

#include "mpsboost/trainer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "mpsboost/backend.hpp"
#include "mpsboost/objective.hpp"
#include "trainer_internal.hpp"

namespace mpsboost {
namespace {

void ValidateMulticlassLabels(const std::vector<double>& labels,
                              std::uint32_t class_count) {
  if (class_count < 2) {
    throw TrainingError("softmax class_count must be at least two");
  }
  for (const double label : labels) {
    if (!std::isfinite(label)) {
      throw TrainingError("softmax labels must be finite");
    }
    const double rounded = std::floor(label);
    if (rounded != label || label < 0.0 ||
        label >= static_cast<double>(class_count)) {
      throw TrainingError("softmax labels must be encoded class ids");
    }
  }
}

std::vector<double> WeightedClassBaseScores(
    const std::vector<double>& labels,
    const std::vector<double>& sample_weights,
    std::uint32_t class_count) {
  const double total_weight =
      trainer_internal::ValidateWeightsAndTotal(labels, sample_weights);
  std::vector<double> class_weights(class_count, 0.0);
  for (std::size_t row = 0; row < labels.size(); ++row) {
    const auto class_index = static_cast<std::uint32_t>(labels[row]);
    class_weights[class_index] += sample_weights[row];
    if (!std::isfinite(class_weights[class_index])) {
      throw TrainingError("softmax class prior overflowed");
    }
  }

  constexpr double kEpsilon = 1e-15;
  std::vector<double> base_scores;
  base_scores.reserve(class_count);
  for (double class_weight : class_weights) {
    const double probability =
        std::max(kEpsilon, class_weight / total_weight);
    const double score = std::log(probability);
    if (!std::isfinite(score)) {
      throw TrainingError("softmax base score is not finite");
    }
    base_scores.push_back(score);
  }
  return base_scores;
}

std::vector<double> InitialMargins(std::uint64_t rows,
                                   const std::vector<double>& base_scores) {
  std::vector<double> margins(
      static_cast<std::size_t>(rows) * base_scores.size(), 0.0);
  for (std::uint64_t row = 0; row < rows; ++row) {
    for (std::size_t class_index = 0; class_index < base_scores.size();
         ++class_index) {
      margins[static_cast<std::size_t>(row) * base_scores.size() + class_index] =
          base_scores[class_index];
    }
  }
  return margins;
}

void AddClassUpdate(std::vector<double>* margins,
                    const std::vector<double>& update,
                    std::uint32_t class_count,
                    std::uint32_t target_class,
                    double learning_rate) {
  if (update.size() * static_cast<std::size_t>(class_count) !=
      margins->size()) {
    throw TrainingError("softmax update shape does not match margins");
  }
  for (std::size_t row = 0; row < update.size(); ++row) {
    double& margin =
        (*margins)[row * static_cast<std::size_t>(class_count) + target_class];
    margin += learning_rate * update[row];
    if (!std::isfinite(margin)) {
      throw TrainingError("softmax margin update overflowed");
    }
  }
}

}  // namespace

MulticlassModel TrainMulticlassSoftmaxModel(
    const BinnedDataset& dataset,
    const std::vector<double>& labels,
    const std::vector<double>& sample_weights,
    const TrainingParameters& parameters,
    std::uint32_t class_count,
    const HistogramBuilder& histogram_builder) {
  trainer_internal::ValidateTrainingParameters(parameters);
  if (dataset.rows() != labels.size() || dataset.max_bins() != parameters.max_bins) {
    throw TrainingError("training data, labels, or max_bins contract mismatch");
  }
  trainer_internal::ValidateWeightsAndTotal(labels, sample_weights);
  ValidateMulticlassLabels(labels, class_count);

  MulticlassModel model;
  model.schema_ = dataset.schema();
  model.class_count_ = class_count;
  model.learning_rate_ = parameters.learning_rate;
  model.base_scores_ =
      WeightedClassBaseScores(labels, sample_weights, class_count);
  model.class_labels_.reserve(class_count);
  for (std::uint32_t class_index = 0; class_index < class_count; ++class_index) {
    model.class_labels_.push_back(static_cast<double>(class_index));
  }
  model.trees_.reserve(static_cast<std::size_t>(parameters.n_estimators) *
                       class_count);

  std::vector<double> margins = InitialMargins(dataset.rows(), model.base_scores_);
  for (std::uint32_t round = 0; round < parameters.n_estimators; ++round) {
    for (std::uint32_t class_index = 0; class_index < class_count;
         ++class_index) {
      const std::vector<GradientPair> gradients =
          trainer_internal::ApplySampleWeights(
              ComputeMulticlassSoftmaxGradients(labels, margins, class_count,
                                                class_index),
              sample_weights);
      RegressionTree tree = TrainSingleRegressionTree(
          dataset, gradients, parameters.tree, histogram_builder);
      const std::vector<double> update = tree.Predict(dataset);
      AddClassUpdate(&margins, update, class_count, class_index,
                     parameters.learning_rate);
      model.trees_.push_back(std::move(tree));
    }
  }
  return model;
}

MulticlassModel MulticlassModel::Restore(QuantizationSchema schema,
                                         std::uint32_t class_count,
                                         double learning_rate,
                                         std::vector<double> base_scores,
                                         std::vector<double> class_labels,
                                         std::vector<RegressionTree> trees) {
  if (class_count < 2 || base_scores.size() != class_count ||
      class_labels.size() != class_count || trees.empty() ||
      trees.size() % class_count != 0 || !std::isfinite(learning_rate) ||
      learning_rate <= 0.0 || learning_rate > 1.0) {
    throw TrainingError("softmax model metadata is invalid");
  }
  for (std::uint32_t class_index = 0; class_index < class_count; ++class_index) {
    if (!std::isfinite(base_scores[class_index]) ||
        !std::isfinite(class_labels[class_index])) {
      throw TrainingError("softmax model class metadata is not finite");
    }
    if (class_index > 0 &&
        !(class_labels[class_index - 1] < class_labels[class_index])) {
      throw TrainingError("softmax model class labels must be strictly increasing");
    }
  }
  for (const RegressionTree& tree : trees) {
    if (tree.feature_count() != schema.features()) {
      throw TrainingError("softmax model tree feature count does not match schema");
    }
    for (const TreeNode& node : tree.nodes()) {
      if (!node.IsLeaf() &&
          node.threshold_bin >=
              schema.feature_metadata()[node.feature_index].bin_count - 1) {
        throw TrainingError("softmax model split threshold is outside schema");
      }
      if (node.IsLeaf() && !node.default_left) {
        throw TrainingError("softmax model leaf nodes must keep default_left true");
      }
    }
  }
  MulticlassModel model;
  model.schema_ = std::move(schema);
  model.class_count_ = class_count;
  model.learning_rate_ = learning_rate;
  model.base_scores_ = std::move(base_scores);
  model.class_labels_ = std::move(class_labels);
  model.trees_ = std::move(trees);
  return model;
}

void MulticlassModel::SetClassLabels(std::vector<double> class_labels) {
  if (class_labels.size() != class_count_) {
    throw TrainingError("softmax class label count does not match class_count");
  }
  for (std::uint32_t class_index = 0; class_index < class_count_;
       ++class_index) {
    if (!std::isfinite(class_labels[class_index])) {
      throw TrainingError("softmax class labels must be finite");
    }
    if (class_index > 0 &&
        !(class_labels[class_index - 1] < class_labels[class_index])) {
      throw TrainingError("softmax class labels must be strictly increasing");
    }
  }
  class_labels_ = std::move(class_labels);
}

std::vector<double> MulticlassModel::PredictMargins(
    const BinnedDataset& dataset) const {
  if (class_count_ < 2 || base_scores_.size() != class_count_ ||
      trees_.empty() || trees_.size() % class_count_ != 0) {
    throw TrainingError("softmax model metadata is inconsistent");
  }
  if (dataset.features() != schema_.features() ||
      dataset.max_bins() != schema_.max_bins() ||
      dataset.boundaries() != schema_.boundaries()) {
    throw TrainingError("prediction data does not use the frozen model binning schema");
  }

  std::vector<double> margins = InitialMargins(dataset.rows(), base_scores_);
  for (std::size_t tree_index = 0; tree_index < trees_.size(); ++tree_index) {
    const std::uint32_t class_index =
        static_cast<std::uint32_t>(tree_index % class_count_);
    const std::vector<double> update = trees_[tree_index].Predict(dataset);
    AddClassUpdate(&margins, update, class_count_, class_index, learning_rate_);
  }
  return margins;
}

}  // namespace mpsboost
