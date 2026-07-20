// MPSBoost multi-round regression training and stable model contract.
//
// Responsibility: defines the device-independent boosting state machine,
// predictive models, and model-I/O entries. The training core depends only on
// GradientComputer/HistogramBuilder, not Python, Metal objects, caches, or files.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mpsboost/binned_dataset.hpp"
#include "mpsboost/tree.hpp"

namespace mpsboost {

class GradientComputer;
class HistogramBuilder;

struct TrainingParameters final {
  enum class Objective : std::uint32_t {
    kSquaredError = 0,
    kBinaryLogistic = 1,
    kQuantile = 2,
    kPoisson = 3,
    kTweedie = 4,
  };

  std::uint32_t n_estimators{100};
  double learning_rate{0.1};
  std::uint32_t max_bins{256};
  Objective objective{Objective::kSquaredError};
  TreeTrainingParameters tree;
  double objective_alpha{0.5};
  double tweedie_variance_power{1.5};
};

class RegressionModel final {
 public:
  std::uint32_t feature_count() const noexcept { return schema_.features(); }
  std::uint32_t tree_count() const noexcept {
    return static_cast<std::uint32_t>(trees_.size());
  }
  double base_score() const noexcept { return base_score_; }
  double learning_rate() const noexcept { return learning_rate_; }
  TrainingParameters::Objective objective() const noexcept { return objective_; }
  double objective_alpha() const noexcept { return objective_alpha_; }
  double tweedie_variance_power() const noexcept {
    return tweedie_variance_power_;
  }
  const QuantizationSchema& schema() const noexcept { return schema_; }
  const std::vector<RegressionTree>& trees() const noexcept { return trees_; }

  // Predict data transformed with this model's schema. The function does not mutate
  // model or input and supports concurrent read-only calls. Schema mismatches fail
  // explicitly to prevent use of data with refitted boundaries.
  std::vector<double> Predict(const BinnedDataset& dataset) const;

  // Sole construction entry for model loaders. All fields are fully validated before
  // acceptance, and failures never produce a partially constructed model.
  static RegressionModel Restore(QuantizationSchema schema,
                                 double base_score,
                                 double learning_rate,
                                 TrainingParameters::Objective objective,
                                 double objective_alpha,
                                 double tweedie_variance_power,
                                 std::vector<RegressionTree> trees);

 private:
  friend RegressionModel TrainRegressionModel(
      const BinnedDataset&,
      const std::vector<double>&,
      const std::vector<double>&,
      const TrainingParameters&,
      const GradientComputer&,
      const HistogramBuilder&);

  QuantizationSchema schema_;
  double base_score_{0.0};
  double learning_rate_{0.1};
  TrainingParameters::Objective objective_{TrainingParameters::Objective::kSquaredError};
  double objective_alpha_{0.5};
  double tweedie_variance_power_{1.5};
  std::vector<RegressionTree> trees_;
};

class MulticlassModel final {
 public:
  std::uint32_t feature_count() const noexcept { return schema_.features(); }
  std::uint32_t class_count() const noexcept { return class_count_; }
  std::uint32_t tree_count() const noexcept {
    return static_cast<std::uint32_t>(trees_.size());
  }
  double learning_rate() const noexcept { return learning_rate_; }
  const QuantizationSchema& schema() const noexcept { return schema_; }
  const std::vector<RegressionTree>& trees() const noexcept { return trees_; }
  const std::vector<double>& base_scores() const noexcept { return base_scores_; }
  const std::vector<double>& class_labels() const noexcept { return class_labels_; }

  // Return row-major raw class margins with shape rows × class_count.
  std::vector<double> PredictMargins(const BinnedDataset& dataset) const;

  // Attach the user-visible numeric class mapping owned by the model file.
  // Labels must be finite, strictly increasing, and match class_count.
  void SetClassLabels(std::vector<double> class_labels);

  // Restore a validated native softmax model from versioned model bytes. Trees
  // must be stored in round-major class order.
  static MulticlassModel Restore(QuantizationSchema schema,
                                 std::uint32_t class_count,
                                 double learning_rate,
                                 std::vector<double> base_scores,
                                 std::vector<double> class_labels,
                                 std::vector<RegressionTree> trees);

 private:
  friend MulticlassModel TrainMulticlassSoftmaxModel(
      const BinnedDataset&,
      const std::vector<double>&,
      const std::vector<double>&,
      const TrainingParameters&,
      std::uint32_t,
      const HistogramBuilder&);

  QuantizationSchema schema_;
  std::uint32_t class_count_{0};
  double learning_rate_{0.1};
  std::vector<double> base_scores_;
  std::vector<double> class_labels_;
  std::vector<RegressionTree> trees_;
};

RegressionModel TrainRegressionModel(
    const BinnedDataset& dataset,
    const std::vector<double>& labels,
    const std::vector<double>& sample_weights,
    const TrainingParameters& parameters,
    const GradientComputer& gradient_computer,
    const HistogramBuilder& histogram_builder);

MulticlassModel TrainMulticlassSoftmaxModel(
    const BinnedDataset& dataset,
    const std::vector<double>& labels,
    const std::vector<double>& sample_weights,
    const TrainingParameters& parameters,
    std::uint32_t class_count,
    const HistogramBuilder& histogram_builder);

// Model-format entries use a versioned binary container, integrity validation, and
// same-directory atomic replacement. Files contain no training data, paths, device
// identifiers, or caches; load failures do not affect a caller's existing model.
std::vector<std::uint8_t> SerializeModel(const RegressionModel& model);
RegressionModel DeserializeModel(const std::vector<std::uint8_t>& bytes);
std::vector<std::uint8_t> SerializeModel(const MulticlassModel& model);
MulticlassModel DeserializeMulticlassModel(const std::vector<std::uint8_t>& bytes);
void SaveModelFile(const RegressionModel& model, const std::string& path);
RegressionModel LoadModelFile(const std::string& path);
void SaveModelFile(const MulticlassModel& model, const std::string& path);
MulticlassModel LoadMulticlassModelFile(const std::string& path);

}  // namespace mpsboost
