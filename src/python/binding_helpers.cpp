// Shared conversion helpers for MPSBoost native Python bindings.
//
// The functions in this file are intentionally policy-free. They translate stable native PODs into
// Python diagnostic values and normalize Python buffers into native matrix views.

#include "binding_helpers.hpp"

#include <pybind11/stl.h>

#include <limits>
#include <utility>

#include "mpsboost/objective.hpp"

namespace mpsboost::python_binding {

DenseMatrixView MakeDenseView(const py::buffer& matrix) {
  const py::buffer_info info = matrix.request();
  if (info.ndim != 2) {
    throw py::value_error("输入矩阵必须是二维 buffer");
  }
  if (info.shape[0] <= 0 || info.shape[1] <= 0) {
    throw py::value_error("输入矩阵必须至少包含一行和一个特征");
  }
  if (info.shape[1] > std::numeric_limits<std::uint32_t>::max()) {
    throw py::value_error("输入特征数量超出 uint32 范围");
  }
  if (info.strides[0] <= 0 || info.strides[1] <= 0) {
    throw py::value_error("当前版本不支持零或负 stride");
  }

  ScalarType scalar_type;
  if (info.format == py::format_descriptor<float>::format() &&
      info.itemsize == static_cast<py::ssize_t>(sizeof(float))) {
    scalar_type = ScalarType::kFloat32;
  } else if (info.format == py::format_descriptor<double>::format() &&
             info.itemsize == static_cast<py::ssize_t>(sizeof(double))) {
    scalar_type = ScalarType::kFloat64;
  } else {
    throw py::type_error("输入 dtype 必须是原生 float32 或 float64");
  }

  const auto features = static_cast<std::uint64_t>(info.shape[1]);
  const auto item_size = static_cast<std::uint64_t>(info.itemsize);
  const bool contiguous =
      static_cast<std::uint64_t>(info.strides[1]) == item_size &&
      features <= std::numeric_limits<std::uint64_t>::max() / item_size &&
      static_cast<std::uint64_t>(info.strides[0]) == features * item_size;

  return DenseMatrixView{info.ptr,
                         static_cast<std::uint64_t>(info.shape[0]),
                         static_cast<std::uint32_t>(info.shape[1]),
                         static_cast<std::uint64_t>(info.strides[0]),
                         static_cast<std::uint64_t>(info.strides[1]),
                         scalar_type,
                         contiguous};
}

py::list BoundariesByFeature(const BinnedDataset& dataset) {
  py::list result;
  const auto& boundaries = dataset.boundaries();
  for (const auto& metadata : dataset.feature_metadata()) {
    py::list feature;
    for (std::uint32_t index = 0; index < metadata.boundary_count; ++index) {
      feature.append(boundaries[metadata.boundary_offset + index]);
    }
    result.append(std::move(feature));
  }
  return result;
}

py::list BinsByFeature(const BinnedDataset& dataset) {
  py::list result;
  for (std::uint32_t feature_index = 0; feature_index < dataset.features();
       ++feature_index) {
    py::list feature;
    for (std::uint64_t row = 0; row < dataset.rows(); ++row) {
      feature.append(dataset.GetBin(row, feature_index));
    }
    result.append(std::move(feature));
  }
  return result;
}

py::list MissingByFeature(const BinnedDataset& dataset) {
  py::list result;
  for (std::uint32_t feature_index = 0; feature_index < dataset.features();
       ++feature_index) {
    py::list feature;
    for (std::uint64_t row = 0; row < dataset.rows(); ++row) {
      feature.append(dataset.IsMissing(row, feature_index));
    }
    result.append(std::move(feature));
  }
  return result;
}

py::list TreeNodes(const RegressionTree& tree) {
  py::list result;
  for (const TreeNode& node : tree.nodes()) {
    py::dict item;
    item["is_leaf"] = node.IsLeaf();
    item["feature_index"] = node.feature_index;
    item["threshold_bin"] = node.threshold_bin;
    item["left_child"] = node.left_child;
    item["right_child"] = node.right_child;
    item["leaf_value"] = node.leaf_value;
    item["gain"] = node.gain;
    item["default_left"] = node.default_left;
    result.append(std::move(item));
  }
  return result;
}

py::list ModelTrees(const RegressionModel& model) {
  py::list result;
  for (const RegressionTree& tree : model.trees()) {
    result.append(tree);
  }
  return result;
}

py::list HistogramsToPython(const NodeHistograms& histograms) {
  py::list result;
  for (const FeatureHistogram& feature : histograms) {
    py::list bins;
    for (const HistogramBin& bin : feature) {
      py::dict item;
      item["count"] = bin.count;
      item["gradient_sum"] = bin.gradient_sum;
      item["hessian_sum"] = bin.hessian_sum;
      bins.append(std::move(item));
    }
    result.append(std::move(bins));
  }
  return result;
}

py::list GradientsToPython(const std::vector<GradientPair>& gradients) {
  py::list result;
  for (const GradientPair& pair : gradients) {
    result.append(py::make_tuple(pair.gradient, pair.hessian));
  }
  return result;
}

py::list SplitCandidatesToPython(
    const std::vector<SplitScanCandidate>& candidates) {
  py::list result;
  for (const SplitScanCandidate& candidate : candidates) {
    py::dict item;
    item["valid"] = candidate.valid;
    item["feature"] = candidate.feature;
    item["threshold_bin"] = candidate.threshold_bin;
    item["left_count"] = candidate.left_count;
    item["right_count"] = candidate.right_count;
    item["left_gradient_sum"] = candidate.left_gradient_sum;
    item["left_hessian_sum"] = candidate.left_hessian_sum;
    item["right_gradient_sum"] = candidate.right_gradient_sum;
    item["right_hessian_sum"] = candidate.right_hessian_sum;
    item["gain"] = candidate.gain;
    result.append(std::move(item));
  }
  return result;
}

py::object RunMpsHistogramForTest(const MpsBackend& backend,
                                  const BinnedDataset& dataset,
                                  const std::vector<double>& labels,
                                  const std::vector<double>& predictions,
                                  const std::vector<std::uint64_t>& rows,
                                  bool baseline) {
  const std::vector<GradientPair> gradients =
      ComputeSquaredErrorGradients(labels, predictions);
  if (baseline) {
    return HistogramsToPython(
        backend.BuildBaselineHistogramsForTest(dataset, rows, gradients));
  }
  py::dict result;
  result["histograms"] =
      HistogramsToPython(backend.BuildHistograms(dataset, rows, gradients));
  const BackendTiming timing = backend.last_timing();
  result["encode_seconds"] = timing.histogram_encode_seconds;
  result["command_seconds"] = timing.histogram_command_seconds;
  return result;
}

}  // namespace mpsboost::python_binding
