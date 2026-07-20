// MPSBoost objective math implementation.
//
// This file is the single authority for objective statistics, node scores, leaf weights, and
// split gains. Other modules must call these functions instead of copying approximate formulas.

#include "mpsboost/objective.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

namespace mpsboost {
namespace {

void RequireFinite(double value, const char* field) {
  if (!std::isfinite(value)) {
    throw TrainingError(std::string(field) + " must be finite");
  }
}

double ValidatedDenominator(double hessian_sum, double reg_lambda) {
  RequireFinite(hessian_sum, "Hessian sum");
  RequireFinite(reg_lambda, "reg_lambda");
  if (hessian_sum < 0.0) {
    throw TrainingError("Hessian sum cannot be negative");
  }
  if (reg_lambda < 0.0) {
    throw TrainingError("reg_lambda cannot be negative");
  }
  const double denominator = hessian_sum + reg_lambda;
  if (!std::isfinite(denominator) || denominator <= 0.0) {
    throw TrainingError("Hessian sum plus reg_lambda must be finite and positive");
  }
  return denominator;
}

void RequireSameNonEmptyLength(const std::vector<double>& labels,
                               const std::vector<double>& predictions) {
  if (labels.empty()) {
    throw TrainingError("Labels must not be empty");
  }
  if (labels.size() != predictions.size()) {
    throw TrainingError("Label and prediction lengths do not match");
  }
}

double RequireBinaryLabel(double label) {
  RequireFinite(label, "label");
  if (label == 0.0 || label == 1.0) {
    return label;
  }
  throw TrainingError("Binary logistic labels must be 0 or 1");
}

void RequireNonNegativeLabel(double label, const char* objective) {
  RequireFinite(label, "label");
  if (label < 0.0) {
    throw TrainingError(std::string(objective) + " labels must be non-negative");
  }
}

double SafeExp(double value, const char* field) {
  RequireFinite(value, field);
  if (value > 709.0) {
    throw TrainingError(std::string(field) + " exponential overflowed");
  }
  const double result = std::exp(value);
  if (!std::isfinite(result)) {
    throw TrainingError(std::string(field) + " exponential overflowed");
  }
  return result;
}

double SoftThresholdGradient(double gradient_sum, double reg_alpha) {
  RequireFinite(gradient_sum, "Gradient sum");
  RequireFinite(reg_alpha, "reg_alpha");
  if (reg_alpha < 0.0) {
    throw TrainingError("reg_alpha cannot be negative");
  }
  const double magnitude = std::fabs(gradient_sum) - reg_alpha;
  if (magnitude <= 0.0) {
    return 0.0;
  }
  return std::copysign(magnitude, gradient_sum);
}

double ClipLeafWeight(double value, double max_delta_step) {
  RequireFinite(max_delta_step, "max_delta_step");
  if (max_delta_step < 0.0) {
    throw TrainingError("max_delta_step cannot be negative");
  }
  if (max_delta_step == 0.0) {
    return value;
  }
  return std::max(-max_delta_step, std::min(max_delta_step, value));
}

}  // namespace

std::vector<GradientPair> ComputeSquaredErrorGradients(
    const std::vector<double>& labels,
    const std::vector<double>& predictions) {
  RequireSameNonEmptyLength(labels, predictions);

  std::vector<GradientPair> result;
  result.reserve(labels.size());
  for (std::size_t index = 0; index < labels.size(); ++index) {
    RequireFinite(labels[index], "label");
    RequireFinite(predictions[index], "prediction");
    const double gradient = predictions[index] - labels[index];
    if (!std::isfinite(gradient)) {
      throw TrainingError("Squared-error gradient overflowed");
    }
    result.push_back(GradientPair{gradient, 1.0});
  }
  return result;
}

double LogisticProbability(double logit) {
  RequireFinite(logit, "logit");
  if (logit >= 0.0) {
    const double exp_negative = std::exp(-logit);
    return 1.0 / (1.0 + exp_negative);
  }
  const double exp_positive = std::exp(logit);
  return exp_positive / (1.0 + exp_positive);
}

std::vector<double> SoftmaxProbabilities(const std::vector<double>& margins) {
  if (margins.size() < 2) {
    throw TrainingError("softmax requires at least two class margins");
  }
  double maximum = -std::numeric_limits<double>::infinity();
  for (const double margin : margins) {
    RequireFinite(margin, "softmax margin");
    maximum = std::max(maximum, margin);
  }
  std::vector<double> probabilities(margins.size());
  double normalizer = 0.0;
  for (std::size_t index = 0; index < margins.size(); ++index) {
    probabilities[index] = std::exp(margins[index] - maximum);
    normalizer += probabilities[index];
  }
  if (!std::isfinite(normalizer) || normalizer <= 0.0) {
    throw TrainingError("softmax normalizer must be finite and positive");
  }
  for (double& probability : probabilities) {
    probability /= normalizer;
  }
  return probabilities;
}

std::vector<GradientPair> ComputeBinaryLogisticGradients(
    const std::vector<double>& labels,
    const std::vector<double>& logits) {
  RequireSameNonEmptyLength(labels, logits);

  std::vector<GradientPair> result;
  result.reserve(labels.size());
  for (std::size_t index = 0; index < labels.size(); ++index) {
    const double label = RequireBinaryLabel(labels[index]);
    const double probability = LogisticProbability(logits[index]);
    const double gradient = probability - label;
    const double hessian = probability * (1.0 - probability);
    if (!std::isfinite(gradient) || !std::isfinite(hessian) || hessian < 0.0) {
      throw TrainingError("Binary logistic gradient/Hessian overflowed");
    }
    result.push_back(GradientPair{gradient, hessian});
  }
  return result;
}

std::vector<GradientPair> ComputeQuantileGradients(
    const std::vector<double>& labels,
    const std::vector<double>& predictions,
    double alpha) {
  RequireSameNonEmptyLength(labels, predictions);
  RequireFinite(alpha, "quantile alpha");
  if (alpha <= 0.0 || alpha >= 1.0) {
    throw TrainingError("quantile alpha must be in (0, 1)");
  }
  std::vector<GradientPair> result;
  result.reserve(labels.size());
  for (std::size_t index = 0; index < labels.size(); ++index) {
    RequireFinite(labels[index], "label");
    RequireFinite(predictions[index], "prediction");
    const double gradient = predictions[index] < labels[index] ? -alpha : 1.0 - alpha;
    result.push_back(GradientPair{gradient, 1.0});
  }
  return result;
}

std::vector<GradientPair> ComputePoissonGradients(
    const std::vector<double>& labels,
    const std::vector<double>& log_means) {
  RequireSameNonEmptyLength(labels, log_means);
  std::vector<GradientPair> result;
  result.reserve(labels.size());
  for (std::size_t index = 0; index < labels.size(); ++index) {
    RequireNonNegativeLabel(labels[index], "poisson");
    const double mean = SafeExp(log_means[index], "poisson log mean");
    const double gradient = mean - labels[index];
    const double hessian = mean;
    if (!std::isfinite(gradient) || !std::isfinite(hessian) || hessian <= 0.0) {
      throw TrainingError("poisson gradient/Hessian overflowed");
    }
    result.push_back(GradientPair{gradient, hessian});
  }
  return result;
}

std::vector<GradientPair> ComputeTweedieGradients(
    const std::vector<double>& labels,
    const std::vector<double>& log_means,
    double variance_power) {
  RequireSameNonEmptyLength(labels, log_means);
  RequireFinite(variance_power, "tweedie variance power");
  if (variance_power <= 1.0 || variance_power >= 2.0) {
    throw TrainingError("tweedie variance power must be in (1, 2)");
  }
  std::vector<GradientPair> result;
  result.reserve(labels.size());
  for (std::size_t index = 0; index < labels.size(); ++index) {
    RequireNonNegativeLabel(labels[index], "tweedie");
    const double first =
        SafeExp((2.0 - variance_power) * log_means[index], "tweedie first term");
    const double second =
        SafeExp((1.0 - variance_power) * log_means[index], "tweedie second term");
    const double gradient = first - labels[index] * second;
    const double hessian =
        (2.0 - variance_power) * first -
        (1.0 - variance_power) * labels[index] * second;
    if (!std::isfinite(gradient) || !std::isfinite(hessian) || hessian <= 0.0) {
      throw TrainingError("tweedie gradient/Hessian overflowed");
    }
    result.push_back(GradientPair{gradient, hessian});
  }
  return result;
}

std::vector<GradientPair> ComputeMulticlassSoftmaxGradients(
    const std::vector<double>& labels,
    const std::vector<double>& margins,
    std::uint32_t class_count,
    std::uint32_t target_class) {
  if (class_count < 2) {
    throw TrainingError("softmax class_count must be at least two");
  }
  if (target_class >= class_count) {
    throw TrainingError("softmax target_class out of range");
  }
  if (labels.empty()) {
    throw TrainingError("labels must be non-empty");
  }
  if (margins.size() != labels.size() * static_cast<std::size_t>(class_count)) {
    throw TrainingError("softmax margins must have rows * class_count values");
  }
  std::vector<GradientPair> result;
  result.reserve(labels.size());
  std::vector<double> row_margins(class_count);
  for (std::size_t row = 0; row < labels.size(); ++row) {
    const double label = labels[row];
    RequireFinite(label, "softmax label");
    const double rounded = std::floor(label);
    if (rounded != label || label < 0.0 ||
        label >= static_cast<double>(class_count)) {
      throw TrainingError("softmax labels must be encoded class ids");
    }
    for (std::uint32_t class_index = 0; class_index < class_count;
         ++class_index) {
      row_margins[class_index] =
          margins[row * static_cast<std::size_t>(class_count) + class_index];
    }
    const std::vector<double> probabilities = SoftmaxProbabilities(row_margins);
    const double indicator =
        static_cast<std::uint32_t>(label) == target_class ? 1.0 : 0.0;
    const double probability = probabilities[target_class];
    const double gradient = probability - indicator;
    const double hessian = probability * (1.0 - probability);
    if (!std::isfinite(gradient) || !std::isfinite(hessian) || hessian < 0.0) {
      throw TrainingError("softmax gradient/Hessian overflowed");
    }
    result.push_back(GradientPair{gradient, hessian});
  }
  return result;
}

double NodeScore(double gradient_sum,
                 double hessian_sum,
                 double reg_lambda,
                 double reg_alpha) {
  const double regularized_gradient =
      SoftThresholdGradient(gradient_sum, reg_alpha);
  const double denominator = ValidatedDenominator(hessian_sum, reg_lambda);
  const double score = regularized_gradient * regularized_gradient / denominator;
  if (!std::isfinite(score)) {
    throw TrainingError("Node score overflowed");
  }
  return score;
}

double LeafWeight(double gradient_sum,
                  double hessian_sum,
                  double reg_lambda,
                  double reg_alpha,
                  double max_delta_step) {
  const double regularized_gradient =
      SoftThresholdGradient(gradient_sum, reg_alpha);
  const double weight =
      ClipLeafWeight(-regularized_gradient /
                         ValidatedDenominator(hessian_sum, reg_lambda),
                     max_delta_step);
  if (!std::isfinite(weight)) {
    throw TrainingError("Leaf value overflowed");
  }
  return weight;
}

double SplitGain(double left_gradient,
                 double left_hessian,
                 double right_gradient,
                 double right_hessian,
                 double reg_lambda,
                 double reg_alpha,
                 double gamma) {
  RequireFinite(gamma, "gamma");
  if (gamma < 0.0) {
    throw TrainingError("gamma cannot be negative");
  }
  // lambda can make a zero-Hessian denominator nonzero numerically, but an empty
  // child is not a valid split. Reject it here so backends and tests cannot bypass
  // tree control flow and obtain an apparently valid gain.
  RequireFinite(left_hessian, "left child Hessian sum");
  RequireFinite(right_hessian, "right child Hessian sum");
  if (left_hessian <= 0.0 || right_hessian <= 0.0) {
    throw TrainingError("Split child Hessian sums must be strictly positive");
  }
  const double parent_gradient = left_gradient + right_gradient;
  const double parent_hessian = left_hessian + right_hessian;
  if (!std::isfinite(parent_gradient) || !std::isfinite(parent_hessian)) {
    throw TrainingError("Parent node statistics overflowed");
  }
  const double gain = 0.5 *
                          (NodeScore(left_gradient, left_hessian, reg_lambda,
                                     reg_alpha) +
                           NodeScore(right_gradient, right_hessian, reg_lambda,
                                     reg_alpha) -
                           NodeScore(parent_gradient, parent_hessian, reg_lambda,
                                     reg_alpha)) -
                      gamma;
  if (!std::isfinite(gain)) {
    throw TrainingError("Split gain overflowed");
  }
  return gain;
}

}  // namespace mpsboost
