// MPSBoost objective math implementation.
//
// This file is the single authority for objective statistics, node scores, leaf weights, and
// split gains. Other modules must call these functions instead of copying approximate formulas.

#include "mpsboost/objective.hpp"

#include <cmath>
#include <cstddef>
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

double NodeScore(double gradient_sum, double hessian_sum, double reg_lambda) {
  RequireFinite(gradient_sum, "Gradient 和");
  const double denominator = ValidatedDenominator(hessian_sum, reg_lambda);
  const double score = gradient_sum * gradient_sum / denominator;
  if (!std::isfinite(score)) {
    throw TrainingError("节点分数发生浮点溢出");
  }
  return score;
}

double LeafWeight(double gradient_sum, double hessian_sum, double reg_lambda) {
  RequireFinite(gradient_sum, "Gradient 和");
  const double weight = -gradient_sum /
                        ValidatedDenominator(hessian_sum, reg_lambda);
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
                          (NodeScore(left_gradient, left_hessian, reg_lambda) +
                           NodeScore(right_gradient, right_hessian, reg_lambda) -
                           NodeScore(parent_gradient, parent_hessian, reg_lambda)) -
                      gamma;
  if (!std::isfinite(gain)) {
    throw TrainingError("切分增益发生浮点溢出");
  }
  return gain;
}

}  // namespace mpsboost
