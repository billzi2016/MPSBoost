// MPSBoost CPU histogram oracle。
//
// 职责：以固定行顺序和 FP64 累计指定节点的 count/G/H，作为后续 Metal kernel 的
// 正确性基线。本文件不选择 split、不生长树，也不充当 device="mps" 的静默回退。

#include <cmath>
#include <limits>

#include "mpsboost/backend.hpp"

namespace mpsboost {

std::vector<GradientPair> CpuReferenceBackend::ComputeSquaredError(
    const std::vector<double>& labels,
    const std::vector<double>& predictions) const {
  // CPU oracle 直接委托唯一目标函数实现，不能在后端中复制公式或参数校验。
  return ComputeSquaredErrorGradients(labels, predictions);
}

NodeHistograms CpuReferenceBackend::BuildHistograms(
    const BinnedDataset& dataset,
    const std::vector<std::uint64_t>& rows,
    const std::vector<GradientPair>& gradients) const {
  if (rows.empty()) {
    throw TrainingError("节点行集合不能为空");
  }
  if (dataset.rows() != static_cast<std::uint64_t>(gradients.size())) {
    throw TrainingError("Gradient 数量与分箱数据行数不一致");
  }

  NodeHistograms histograms;
  histograms.reserve(dataset.features());
  for (const FeatureBinMetadata& metadata : dataset.feature_metadata()) {
    histograms.emplace_back(metadata.bin_count);
  }

  // 行是最外层循环，确保每个 bin 内的 FP64 累计顺序由输入 row index 明确定义，
  // 不依赖线程调度。GPU 对照允许冻结容差，但 CPU 自身必须逐次完全确定。
  for (const std::uint64_t row : rows) {
    if (row >= dataset.rows()) {
      throw TrainingError("Histogram 行索引越界");
    }
    const GradientPair& pair = gradients[static_cast<std::size_t>(row)];
    if (!std::isfinite(pair.gradient) || !std::isfinite(pair.hessian) ||
        pair.hessian < 0.0) {
      throw TrainingError("Gradient/Hessian 必须有限且 Hessian 非负");
    }
    for (std::uint32_t feature = 0; feature < dataset.features(); ++feature) {
      const std::uint32_t bin = dataset.GetBin(row, feature);
      FeatureHistogram& feature_histogram = histograms[feature];
      if (bin >= feature_histogram.size()) {
        throw TrainingError("分箱值超出特征 histogram 范围");
      }
      HistogramBin& target = feature_histogram[bin];
      if (target.count == std::numeric_limits<std::uint64_t>::max()) {
        throw TrainingError("Histogram 样本计数溢出");
      }
      ++target.count;
      target.gradient_sum += pair.gradient;
      target.hessian_sum += pair.hessian;
      if (!std::isfinite(target.gradient_sum) ||
          !std::isfinite(target.hessian_sum)) {
        throw TrainingError("Histogram FP64 累计发生溢出");
      }
    }
  }
  return histograms;
}

}  // namespace mpsboost
