// MPSBoost objective functions and training math contracts.
//
// This header is the only domain contract for first/second-order statistics, node scores, leaf
// weights, and split gains. CPU oracle code, training core code, and future Metal kernels must
// match these semantics exactly; Python tests may call the functions but must not reimplement
// generic formulas as production behavior.
#pragma once

#include <cstdint>
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

// Compute pinball-loss statistics with constant Hessian. ``alpha`` is the target quantile in
// (0, 1). This is a controlled diagonal approximation so the shared Newton tree path can train
// quantile models without introducing a second tree optimizer.
std::vector<GradientPair> ComputeQuantileGradients(
    const std::vector<double>& labels,
    const std::vector<double>& predictions,
    double alpha);

// Compute Poisson log-link statistics for non-negative labels: raw predictions are log means.
std::vector<GradientPair> ComputePoissonGradients(
    const std::vector<double>& labels,
    const std::vector<double>& log_means);

// Compute Tweedie log-link statistics for variance power in (1, 2). Labels must be non-negative.
std::vector<GradientPair> ComputeTweedieGradients(
    const std::vector<double>& labels,
    const std::vector<double>& log_means,
    double variance_power);

// Convert one raw binary-logistic margin to probability with overflow-stable sigmoid semantics.
double LogisticProbability(double logit);

// Convert one row of raw multiclass margins to overflow-stable softmax probabilities.
std::vector<double> SoftmaxProbabilities(const std::vector<double>& margins);

// Compute diagonal softmax statistics for one class. Labels must be integer class ids encoded in
// [0, class_count). Margins are row-major with shape rows × class_count.
std::vector<GradientPair> ComputeMulticlassSoftmaxGradients(
    const std::vector<double>& labels,
    const std::vector<double>& margins,
    std::uint32_t class_count,
    std::uint32_t target_class);

// Compute the node score G^2/(H+lambda). H and lambda must be non-negative,
// and the denominator must be strictly positive.
double NodeScore(double gradient_sum,
                 double hessian_sum,
                 double reg_lambda,
                 double reg_alpha = 0.0);

// Compute the leaf value -G/(H+lambda). Validation rules match NodeScore exactly.
double LeafWeight(double gradient_sum,
                  double hessian_sum,
                  double reg_lambda,
                  double reg_alpha = 0.0,
                  double max_delta_step = 0.0);

// Compute 0.5*(score_left+score_right-score_parent)-gamma. Left and right
// statistics must represent non-empty candidates with positive Hessians; gamma
// must be non-negative. This function does not decide whether to split; callers
// accept only strictly positive gain.
double SplitGain(double left_gradient,
                 double left_hessian,
                 double right_gradient,
                 double right_hessian,
                 double reg_lambda,
                 double reg_alpha,
                 double gamma);

}  // namespace mpsboost
