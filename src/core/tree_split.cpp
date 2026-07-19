// Deterministic split evaluation for native regression trees.
//
// This unit owns node-statistic accumulation and split ranking. The training
// orchestrator therefore consumes one stable policy on both CPU and MPS.

#include "tree_internal.hpp"

#include <cmath>
#include <limits>

namespace mpsboost::tree_internal {
namespace {

std::uint32_t StableRandomThreshold(std::uint32_t seed,
                                    std::uint32_t feature,
                                    std::uint32_t node_index,
                                    std::uint32_t depth,
                                    std::uint32_t candidate_count) {
  if (candidate_count == 0) {
    throw TrainingError("random threshold candidate count must be positive");
  }
  std::uint64_t value = seed;
  value ^= (static_cast<std::uint64_t>(feature) + 0x9E3779B97F4A7C15ULL);
  value *= 0xBF58476D1CE4E5B9ULL;
  value ^= (static_cast<std::uint64_t>(node_index) << 32U) | depth;
  value *= 0x94D049BB133111EBULL;
  value ^= value >> 31U;
  return static_cast<std::uint32_t>(value % candidate_count);
}

bool IsBetterSplit(const SplitCandidate& candidate,
                   const SplitCandidate& incumbent) {
  if (!incumbent.valid) {
    return true;
  }
  // Do not merge nearly equal gains with an epsilon because that would make
  // ordering depend on the data scale. Exact ties use stable field ordering.
  if (candidate.gain != incumbent.gain) {
    return candidate.gain > incumbent.gain;
  }
  if (candidate.feature != incumbent.feature) {
    return candidate.feature < incumbent.feature;
  }
  return candidate.threshold_bin < incumbent.threshold_bin;
}

}  // namespace

NodeStatistics SumRows(const std::vector<std::uint64_t>& rows,
                       const std::vector<GradientPair>& gradients) {
  NodeStatistics result;
  for (const std::uint64_t row : rows) {
    if (row >= gradients.size()) {
      throw TrainingError("节点统计行索引越界");
    }
    const GradientPair& pair = gradients[static_cast<std::size_t>(row)];
    if (!std::isfinite(pair.gradient) || !std::isfinite(pair.hessian) ||
        pair.hessian < 0.0) {
      throw TrainingError("Gradient/Hessian 必须有限且 Hessian 非负");
    }
    ++result.count;
    result.gradient_sum += pair.gradient;
    result.hessian_sum += pair.hessian;
    if (!std::isfinite(result.gradient_sum) ||
        !std::isfinite(result.hessian_sum)) {
      throw TrainingError("节点 FP64 统计发生溢出");
    }
  }
  return result;
}

SplitCandidate FindBestSplit(const NodeHistograms& histograms,
                             const BinnedDataset& dataset,
                             const NodeStatistics& parent,
                             std::uint32_t node_index,
                             std::uint32_t depth,
                             const TreeTrainingParameters& parameters) {
  if (histograms.size() != dataset.features()) {
    throw TrainingError("Histogram 特征数量与数据集不一致");
  }
  SplitCandidate best;
  for (std::uint32_t feature = 0; feature < dataset.features(); ++feature) {
    const FeatureHistogram& bins = histograms[feature];
    if (bins.size() != dataset.feature_metadata()[feature].bin_count) {
      throw TrainingError("Histogram bin 数量与特征元数据不一致");
    }
    std::uint64_t histogram_count = 0;
    for (const HistogramBin& bin : bins) {
      if (bin.count >
          std::numeric_limits<std::uint64_t>::max() - histogram_count) {
        throw TrainingError("Histogram 样本计数溢出");
      }
      histogram_count += bin.count;
      if (!std::isfinite(bin.gradient_sum) ||
          !std::isfinite(bin.hessian_sum) || bin.hessian_sum < 0.0) {
        throw TrainingError("Histogram G/H 必须有限且 Hessian 非负");
      }
    }
    if (histogram_count != parent.count) {
      throw TrainingError("Histogram 样本计数与节点统计不一致");
    }
    if (bins.size() < 2) {
      continue;
    }
    NodeStatistics left;
    for (std::uint32_t threshold = 0; threshold + 1 < bins.size();
         ++threshold) {
      const HistogramBin& bin = bins[threshold];
      left.count += bin.count;
      left.gradient_sum += bin.gradient_sum;
      left.hessian_sum += bin.hessian_sum;
      if (!std::isfinite(left.gradient_sum) ||
          !std::isfinite(left.hessian_sum)) {
        throw TrainingError("Histogram 前缀累计发生浮点溢出");
      }
      const NodeStatistics right{
          parent.count - left.count,
          parent.gradient_sum - left.gradient_sum,
          parent.hessian_sum - left.hessian_sum,
      };
      if (left.count < parameters.min_samples_leaf ||
          right.count < parameters.min_samples_leaf ||
          left.hessian_sum < parameters.min_child_weight ||
          right.hessian_sum < parameters.min_child_weight ||
          left.hessian_sum <= 0.0 || right.hessian_sum <= 0.0) {
        continue;
      }
      if (parameters.split_strategy ==
              TreeTrainingParameters::SplitStrategy::kRandomThreshold &&
          threshold != StableRandomThreshold(
                           parameters.random_seed, feature, node_index, depth,
                           static_cast<std::uint32_t>(bins.size() - 1))) {
        continue;
      }
      const double gain = SplitGain(
          left.gradient_sum, left.hessian_sum, right.gradient_sum,
          right.hessian_sum, parameters.reg_lambda, parameters.gamma);
      // A non-positive gain cannot improve the objective and must not create a
      // meaningless branch. The strict comparison is part of tree semantics.
      if (gain <= 0.0) {
        continue;
      }
      const SplitCandidate candidate{
          true, feature, threshold, gain, left, right,
      };
      if (IsBetterSplit(candidate, best)) {
        best = candidate;
      }
    }
  }
  return best;
}

}  // namespace mpsboost::tree_internal
