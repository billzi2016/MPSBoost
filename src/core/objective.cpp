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
    throw TrainingError(std::string(field) + " 必须是有限值");
  }
}

double ValidatedDenominator(double hessian_sum, double reg_lambda) {
  RequireFinite(hessian_sum, "Hessian 和");
  RequireFinite(reg_lambda, "reg_lambda");
  if (hessian_sum < 0.0) {
    throw TrainingError("Hessian 和不能为负数");
  }
  if (reg_lambda < 0.0) {
    throw TrainingError("reg_lambda 不能为负数");
  }
  const double denominator = hessian_sum + reg_lambda;
  if (!std::isfinite(denominator) || denominator <= 0.0) {
    throw TrainingError("Hessian 和与 reg_lambda 之和必须是有限正数");
  }
  return denominator;
}

void RequireSameNonEmptyLength(const std::vector<double>& labels,
                               const std::vector<double>& predictions) {
  if (labels.empty()) {
    throw TrainingError("标签不能为空");
  }
  if (labels.size() != predictions.size()) {
    throw TrainingError("标签与预测长度不一致");
  }
}

double RequireBinaryLabel(double label) {
  RequireFinite(label, "标签");
  if (label == 0.0 || label == 1.0) {
    return label;
  }
  throw TrainingError("二分类 logistic 标签必须是 0 或 1");
}

double SoftThresholdGradient(double gradient_sum, double reg_alpha) {
  RequireFinite(gradient_sum, "Gradient 和");
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
    RequireFinite(labels[index], "标签");
    RequireFinite(predictions[index], "预测");
    const double gradient = predictions[index] - labels[index];
    if (!std::isfinite(gradient)) {
      throw TrainingError("平方误差 gradient 发生浮点溢出");
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
      throw TrainingError("二分类 logistic gradient/Hessian 发生浮点溢出");
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
    throw TrainingError("节点分数发生浮点溢出");
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
    throw TrainingError("叶值发生浮点溢出");
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
    throw TrainingError("gamma 不能为负数");
  }
  // lambda 可以让零 Hessian 的分母在数值上非零，但空子节点并不是合法 split。
  // 在唯一数学入口拒绝它，避免后端或测试绕过树控制流后得到看似有效的增益。
  RequireFinite(left_hessian, "左子节点 Hessian 和");
  RequireFinite(right_hessian, "右子节点 Hessian 和");
  if (left_hessian <= 0.0 || right_hessian <= 0.0) {
    throw TrainingError("切分左右子节点 Hessian 和必须严格为正");
  }
  const double parent_gradient = left_gradient + right_gradient;
  const double parent_hessian = left_hessian + right_hessian;
  if (!std::isfinite(parent_gradient) || !std::isfinite(parent_hessian)) {
    throw TrainingError("父节点统计发生浮点溢出");
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
    throw TrainingError("切分增益发生浮点溢出");
  }
  return gain;
}

}  // namespace mpsboost
