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

NodeStatistics SubtractStatistics(const NodeStatistics& left,
                                  const NodeStatistics& right) {
  return NodeStatistics{
      left.count - right.count,
      left.gradient_sum - right.gradient_sum,
      left.hessian_sum - right.hessian_sum,
  };
}

NodeStatistics AddStatistics(const NodeStatistics& left,
                             const NodeStatistics& right) {
  return NodeStatistics{
      left.count + right.count,
      left.gradient_sum + right.gradient_sum,
      left.hessian_sum + right.hessian_sum,
  };
}

void ConsiderMissingDirection(const NodeStatistics& left,
                              const NodeStatistics& right,
                              const NodeStatistics& missing,
                              double parent_lower_bound,
                              double parent_upper_bound,
                              bool default_left,
                              std::uint32_t feature,
                              std::uint32_t threshold,
                              const TreeTrainingParameters& parameters,
                              SplitCandidate* best) {
  const NodeStatistics candidate_left =
      default_left ? AddStatistics(left, missing) : left;
  const NodeStatistics candidate_right =
      default_left ? right : AddStatistics(right, missing);
  if (candidate_left.count < parameters.min_samples_leaf ||
      candidate_right.count < parameters.min_samples_leaf ||
      candidate_left.hessian_sum < parameters.min_child_weight ||
      candidate_right.hessian_sum < parameters.min_child_weight ||
      candidate_left.hessian_sum <= 0.0 || candidate_right.hessian_sum <= 0.0) {
    return;
  }
  const double gain = SplitGain(
      candidate_left.gradient_sum, candidate_left.hessian_sum,
      candidate_right.gradient_sum, candidate_right.hessian_sum,
      parameters.reg_lambda, parameters.reg_alpha, parameters.gamma);
  if (gain <= 0.0 || gain < parameters.min_gain_to_split) {
    return;
  }
  const MonotonicChildBounds bounds = MonotonicBoundsForSplit(
      candidate_left, candidate_right, parent_lower_bound, parent_upper_bound,
      feature, parameters);
  if (!bounds.valid) {
    return;
  }
  const SplitCandidate candidate{
      true,
      feature,
      threshold,
      gain,
      default_left,
      candidate_left,
      candidate_right,
      bounds.left_lower_bound,
      bounds.left_upper_bound,
      bounds.right_lower_bound,
      bounds.right_upper_bound,
  };
  if (IsBetterSplit(candidate, *best)) {
    *best = candidate;
  }
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
                             double lower_bound,
                             double upper_bound,
                             const std::vector<std::uint32_t>& path_features,
                             std::uint32_t node_index,
                             std::uint32_t depth,
                             const TreeTrainingParameters& parameters,
                             const std::vector<FeatureMissingStatistics>& missing) {
  if (histograms.size() != dataset.features()) {
    throw TrainingError("Histogram 特征数量与数据集不一致");
  }
  if (missing.size() != dataset.features()) {
    throw TrainingError("missing statistics feature count mismatch");
  }
  SplitCandidate best;
  for (std::uint32_t feature = 0; feature < dataset.features(); ++feature) {
    if (!InteractionAllowsFeature(path_features, feature, parameters)) {
      continue;
    }
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
    const NodeStatistics missing_stats = missing[feature].missing;
    if (missing_stats.count > parent.count ||
        missing_stats.hessian_sum > parent.hessian_sum) {
      throw TrainingError("missing statistics exceed parent statistics");
    }
    const NodeStatistics non_missing_parent =
        SubtractStatistics(parent, missing_stats);
    NodeStatistics left;
    for (std::uint32_t threshold = 0; threshold + 1 < bins.size();
         ++threshold) {
      const HistogramBin& bin = bins[threshold];
      const std::uint64_t bin_count =
          threshold == 0 ? bin.count - missing_stats.count : bin.count;
      const double bin_gradient =
          threshold == 0 ? bin.gradient_sum - missing_stats.gradient_sum
                         : bin.gradient_sum;
      const double bin_hessian =
          threshold == 0 ? bin.hessian_sum - missing_stats.hessian_sum
                         : bin.hessian_sum;
      left.count += bin_count;
      left.gradient_sum += bin_gradient;
      left.hessian_sum += bin_hessian;
      if (!std::isfinite(left.gradient_sum) ||
          !std::isfinite(left.hessian_sum)) {
        throw TrainingError("Histogram 前缀累计发生浮点溢出");
      }
      const NodeStatistics right = SubtractStatistics(non_missing_parent, left);
      if (left.count > non_missing_parent.count) {
        throw TrainingError("non-missing histogram prefix exceeded parent");
      }
      if (left.count == 0 && right.count == 0) {
        continue;
      }
      if (parameters.split_strategy ==
              TreeTrainingParameters::SplitStrategy::kRandomThreshold &&
          threshold != StableRandomThreshold(
                           parameters.random_seed, feature, node_index, depth,
                           static_cast<std::uint32_t>(bins.size() - 1))) {
        continue;
      }
      ConsiderMissingDirection(left, right, missing_stats, lower_bound,
                               upper_bound, true, feature, threshold,
                               parameters, &best);
      ConsiderMissingDirection(left, right, missing_stats, lower_bound,
                               upper_bound, false, feature, threshold,
                               parameters, &best);
    }
  }
  return best;
}

PreparedSplit PrepareSplitRows(const BinnedDataset& dataset,
                               const ActiveNode& active,
                               const NodeHistograms& histograms,
                               const std::vector<GradientPair>& gradients,
                               const TreeTrainingParameters& parameters) {
  if (histograms.size() != dataset.features()) {
    throw TrainingError("Histogram 特征数量与数据集不一致");
  }
  std::vector<FeatureMissingStatistics> missing(dataset.features());
  for (std::uint32_t feature = 0; feature < dataset.features(); ++feature) {
    for (const std::uint64_t row : active.rows) {
      if (!dataset.IsMissing(row, feature)) {
        continue;
      }
      if (row >= gradients.size()) {
        throw TrainingError("missing statistics row index out of range");
      }
      const GradientPair& pair = gradients[static_cast<std::size_t>(row)];
      FeatureMissingStatistics& stats = missing[feature];
      ++stats.missing.count;
      stats.missing.gradient_sum += pair.gradient;
      stats.missing.hessian_sum += pair.hessian;
    }
  }
  const SplitCandidate split =
      FindBestSplit(histograms, dataset, active.statistics, active.lower_bound,
                    active.upper_bound, active.path_features, active.node_index,
                    active.depth, parameters, missing);
  if (!split.valid) {
    return {};
  }

  PreparedSplit prepared;
  prepared.valid = true;
  prepared.split = split;
  prepared.left_rows.reserve(static_cast<std::size_t>(split.left.count));
  prepared.right_rows.reserve(static_cast<std::size_t>(split.right.count));
  for (const std::uint64_t row : active.rows) {
    // Binning uses lower-bound semantics, so values equal to a boundary stay in
    // the lower bin. Training and prediction must route bin <= threshold left.
    const bool goes_left =
        dataset.IsMissing(row, split.feature)
            ? split.default_left
            : dataset.GetBin(row, split.feature) <= split.threshold_bin;
    if (goes_left) {
      prepared.left_rows.push_back(row);
    } else {
      prepared.right_rows.push_back(row);
    }
  }
  if (prepared.left_rows.size() != split.left.count ||
      prepared.right_rows.size() != split.right.count) {
    throw TrainingError("样本分区数量与 histogram 统计不一致");
  }
  return prepared;
}

}  // namespace mpsboost::tree_internal
