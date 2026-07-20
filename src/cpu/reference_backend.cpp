// MPSBoost CPU histogram oracle。
//
// Responsibility: accumulates a selected node's count/G/H in fixed row order and
// FP64 as the correctness baseline for Metal kernels. This file neither selects
// splits nor grows trees, and it is not a silent fallback for device="mps".

#include <cmath>
#include <limits>

#include "mpsboost/backend.hpp"

namespace mpsboost {

std::vector<GradientPair> CpuReferenceBackend::ComputeSquaredError(
    const std::vector<double>& labels,
    const std::vector<double>& predictions) const {
  // The CPU oracle delegates to the single objective implementation; do not copy
  // formulas or parameter validation into the backend.
  return ComputeSquaredErrorGradients(labels, predictions);
}

NodeHistograms CpuReferenceBackend::BuildHistograms(
    const BinnedDataset& dataset,
    const std::vector<std::uint64_t>& rows,
    const std::vector<GradientPair>& gradients) const {
  if (rows.empty()) {
    throw TrainingError("Node row set must not be empty");
  }
  if (dataset.rows() != static_cast<std::uint64_t>(gradients.size())) {
    throw TrainingError("Gradient count does not match binned dataset row count");
  }

  NodeHistograms histograms;
  histograms.reserve(dataset.features());
  for (const FeatureBinMetadata& metadata : dataset.feature_metadata()) {
    histograms.emplace_back(metadata.bin_count);
  }

  // Rows are the outer loop so FP64 accumulation order within every bin is defined
  // by input row index rather than thread scheduling. GPU comparison may use a
  // frozen tolerance, but the CPU result itself must be exactly deterministic.
  for (const std::uint64_t row : rows) {
    if (row >= dataset.rows()) {
      throw TrainingError("Histogram row index is out of bounds");
    }
    const GradientPair& pair = gradients[static_cast<std::size_t>(row)];
    if (!std::isfinite(pair.gradient) || !std::isfinite(pair.hessian) ||
        pair.hessian < 0.0) {
      throw TrainingError("Gradient/Hessian must be finite and Hessian non-negative");
    }
    for (std::uint32_t feature = 0; feature < dataset.features(); ++feature) {
      const std::uint32_t bin = dataset.GetBin(row, feature);
      FeatureHistogram& feature_histogram = histograms[feature];
      if (bin >= feature_histogram.size()) {
        throw TrainingError("Binned value exceeds the feature histogram range");
      }
      HistogramBin& target = feature_histogram[bin];
      if (target.count == std::numeric_limits<std::uint64_t>::max()) {
        throw TrainingError("Histogram sample count overflowed");
      }
      ++target.count;
      target.gradient_sum += pair.gradient;
      target.hessian_sum += pair.hessian;
      if (!std::isfinite(target.gradient_sum) ||
          !std::isfinite(target.hessian_sum)) {
        throw TrainingError("Histogram FP64 accumulation overflowed");
      }
    }
  }
  return histograms;
}

}  // namespace mpsboost
