// MPSBoost objective functions and training math contracts.
//
// This header is the only domain contract for first/second-order statistics, node scores, leaf
// weights, and split gains. CPU oracle code, training core code, and future Metal kernels must
// match these semantics exactly; Python tests may call the functions but must not reimplement
// generic formulas as production behavior.
#pragma once

#include <stdexcept>
#include <vector>

namespace mpsboost {

// Unified exception for invalid training inputs, parameters, or intermediate statistics. Once
// raised, callers must not export partial trees or continue as if the model were trustworthy.
class TrainingError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Per-sample second-order statistics. CPU code accumulates these values in double precision as
// the oracle; device implementations may use float internally but must be tested against this
// layer-by-layer semantic contract.
struct GradientPair final {
  double gradient{0.0};
  double hessian{0.0};
};

// Compute squared-error statistics: g = prediction - label, h = 1. Inputs must be non-empty,
// equal-length, and finite. The return value owns independent memory.
std::vector<GradientPair> ComputeSquaredErrorGradients(
    const std::vector<double>& labels,
    const std::vector<double>& predictions);

// Compute binary-logistic statistics for labels in {0, 1}: g = sigmoid(logit) - label and
// h = sigmoid(logit) * (1 - sigmoid(logit)). Inputs must be non-empty, equal-length, finite, and
// binary-labeled. Logits are raw margins, not probabilities.
std::vector<GradientPair> ComputeBinaryLogisticGradients(
    const std::vector<double>& labels,
    const std::vector<double>& logits);

// Convert one raw binary-logistic margin to probability with overflow-stable sigmoid semantics.
double LogisticProbability(double logit);

// 计算节点分数 G²/(H+lambda)。H 和 lambda 必须非负，分母必须严格为正。
double NodeScore(double gradient_sum, double hessian_sum, double reg_lambda);

// 计算叶值 -G/(H+lambda)。验证规则与 NodeScore 完全一致。
double LeafWeight(double gradient_sum, double hessian_sum, double reg_lambda);

// 计算 0.5*(score_left+score_right-score_parent)-gamma。左右统计必须代表非空、
// 正 Hessian 候选；gamma 必须非负。函数不决定是否切分，调用方只接受严格正增益。
double SplitGain(double left_gradient,
                 double left_hessian,
                 double right_gradient,
                 double right_hessian,
                 double reg_lambda,
                 double gamma);

}  // namespace mpsboost
