// MPSBoost Python 原生绑定。
//
// 职责：把稳定 C++ POD 和异常转换为轻量 Python 对象。这里不得复制设备实现、参数
// 校验或训练算法。所有公开用户体验由 Python 层组织，以下下划线入口属于内部接口。

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <limits>

#include "mpsboost/backend.hpp"
#include "mpsboost/binned_dataset.hpp"
#include "mpsboost/objective.hpp"
#include "mpsboost/trainer.hpp"
#include "mpsboost/tree.hpp"
#include "mpsboost/version.hpp"

namespace py = pybind11;

namespace {

// 把任意支持 Python Buffer Protocol 的二维 float32/float64 对象转换为同步借用视图。
// 这里不复制输入，也不保存指针；QuantizeDense 返回前，Python buffer 始终保持存活。
mpsboost::DenseMatrixView MakeDenseView(const py::buffer& matrix) {
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

  mpsboost::ScalarType scalar_type;
  if (info.format == py::format_descriptor<float>::format() &&
      info.itemsize == static_cast<py::ssize_t>(sizeof(float))) {
    scalar_type = mpsboost::ScalarType::kFloat32;
  } else if (info.format == py::format_descriptor<double>::format() &&
             info.itemsize == static_cast<py::ssize_t>(sizeof(double))) {
    scalar_type = mpsboost::ScalarType::kFloat64;
  } else {
    throw py::type_error("输入 dtype 必须是原生 float32 或 float64");
  }

  const auto features = static_cast<std::uint64_t>(info.shape[1]);
  const auto item_size = static_cast<std::uint64_t>(info.itemsize);
  const bool contiguous =
      static_cast<std::uint64_t>(info.strides[1]) == item_size &&
      features <= std::numeric_limits<std::uint64_t>::max() / item_size &&
      static_cast<std::uint64_t>(info.strides[0]) == features * item_size;

  return mpsboost::DenseMatrixView{
      info.ptr,
      static_cast<std::uint64_t>(info.shape[0]),
      static_cast<std::uint32_t>(info.shape[1]),
      static_cast<std::uint64_t>(info.strides[0]),
      static_cast<std::uint64_t>(info.strides[1]),
      scalar_type,
      contiguous,
  };
}

py::list BoundariesByFeature(const mpsboost::BinnedDataset& dataset) {
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

py::list BinsByFeature(const mpsboost::BinnedDataset& dataset) {
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

// 把扁平节点转换为只读 Python 字典，供 CPU oracle 手算测试逐字段断言。该函数不
// 暴露可写节点，避免测试绕过 C++ 结构验证制造生产路径不存在的模型。
py::list TreeNodes(const mpsboost::RegressionTree& tree) {
  py::list result;
  for (const mpsboost::TreeNode& node : tree.nodes()) {
    py::dict item;
    item["is_leaf"] = node.IsLeaf();
    item["feature_index"] = node.feature_index;
    item["threshold_bin"] = node.threshold_bin;
    item["left_child"] = node.left_child;
    item["right_child"] = node.right_child;
    item["leaf_value"] = node.leaf_value;
    item["gain"] = node.gain;
    result.append(std::move(item));
  }
  return result;
}

// 将 CPU oracle histogram 转换为不可变语义的嵌套 Python 值，供测试逐 bin 对照。
// 转换只发生在内部测试入口，不进入训练热路径，也不向 Python 复制计算公式。
py::list HistogramsToPython(const mpsboost::NodeHistograms& histograms) {
  py::list result;
  for (const mpsboost::FeatureHistogram& feature : histograms) {
    py::list bins;
    for (const mpsboost::HistogramBin& bin : feature) {
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

py::list GradientsToPython(const std::vector<mpsboost::GradientPair>& gradients) {
  py::list result;
  for (const mpsboost::GradientPair& pair : gradients) {
    result.append(py::make_tuple(pair.gradient, pair.hessian));
  }
  return result;
}

py::list SplitCandidatesToPython(
    const std::vector<mpsboost::SplitScanCandidate>& candidates) {
  py::list result;
  for (const mpsboost::SplitScanCandidate& candidate : candidates) {
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

py::object RunMpsHistogramForTest(
    const mpsboost::MpsBackend& backend,
    const mpsboost::BinnedDataset& dataset,
    const std::vector<double>& labels,
    const std::vector<double>& predictions,
    const std::vector<std::uint64_t>& rows,
    bool baseline) {
  const std::vector<mpsboost::GradientPair> gradients =
      mpsboost::ComputeSquaredErrorGradients(labels, predictions);
  if (baseline) {
    return HistogramsToPython(
        backend.BuildBaselineHistogramsForTest(dataset, rows, gradients));
  }
  py::dict result;
  result["histograms"] =
      HistogramsToPython(backend.BuildHistograms(dataset, rows, gradients));
  const mpsboost::BackendTiming timing = backend.last_timing();
  result["encode_seconds"] = timing.histogram_encode_seconds;
  result["command_seconds"] = timing.histogram_command_seconds;
  return result;
}

}  // namespace

PYBIND11_MODULE(_native, module) {
  module.doc() = "MPSBoost 原生 MPS/Metal 后端";
  module.attr("__version__") = MPSBOOST_VERSION;

  // 统一注册后端异常，使 Python 可以明确区分设备执行失败与普通参数错误。
  py::register_exception<mpsboost::BackendError>(module, "BackendError");
  py::register_exception<mpsboost::DataError>(module, "DataError", PyExc_ValueError);
  py::register_exception<mpsboost::TrainingError>(module, "TrainingError",
                                                  PyExc_ValueError);

  py::class_<mpsboost::BinnedDataset>(module, "_BinnedDataset")
      .def_property_readonly("rows", &mpsboost::BinnedDataset::rows)
      .def_property_readonly("features", &mpsboost::BinnedDataset::features)
      .def_property_readonly("max_bins", &mpsboost::BinnedDataset::max_bins)
      .def_property_readonly(
          "bin_width",
          [](const mpsboost::BinnedDataset& dataset) {
            return dataset.storage() == mpsboost::BinStorage::kUInt8 ? 8 : 16;
          })
      .def_property_readonly("source_contiguous",
                             &mpsboost::BinnedDataset::source_contiguous)
      .def_property_readonly("source_was_copied",
                             &mpsboost::BinnedDataset::source_was_copied)
      .def_property_readonly("boundaries", &BoundariesByFeature)
      .def_property_readonly("bins", &BinsByFeature)
      .def("serialize",
           [](const mpsboost::BinnedDataset& dataset) {
             const std::vector<std::uint8_t> bytes = dataset.Serialize();
             return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
           });

  py::class_<mpsboost::RegressionTree>(module, "_RegressionTree")
      .def_property_readonly("feature_count",
                             &mpsboost::RegressionTree::feature_count)
      .def_property_readonly("nodes", &TreeNodes)
      .def("predict", &mpsboost::RegressionTree::Predict, py::arg("dataset"),
           "对内部量化数据执行唯一 C++ 扁平树预测。");

  py::class_<mpsboost::TrainingParameters>(module, "_TrainingParameters")
      .def(py::init([](std::uint32_t n_estimators, double learning_rate,
                       std::uint32_t max_bins, std::uint32_t max_depth,
                       std::uint64_t min_samples_leaf,
                       double min_child_weight, double reg_lambda,
                       double gamma) {
             return mpsboost::TrainingParameters{
                 n_estimators,
                 learning_rate,
                 max_bins,
                 mpsboost::TreeTrainingParameters{
                     max_depth, min_samples_leaf, min_child_weight,
                     reg_lambda, gamma}};
           }),
           py::arg("n_estimators"), py::arg("learning_rate"),
           py::arg("max_bins"), py::arg("max_depth"),
           py::arg("min_samples_leaf"), py::arg("min_child_weight"),
           py::arg("reg_lambda"), py::arg("gamma") = 0.0,
           "创建已命名字段的内部训练参数值对象。");

  py::class_<mpsboost::MpsBackend>(module, "_MpsBackend")
      .def(py::init<std::string>(), py::arg("metallib_path"),
           "创建可复用 device/library/pipeline 的真实 MPS 测试会话。")
      .def("gradients", [](const mpsboost::MpsBackend& backend, const std::vector<double>& labels, const std::vector<double>& predictions) { return GradientsToPython(
                                                                                                                                                 backend.ComputeSquaredError(labels, predictions)); }, py::arg("labels"), py::arg("predictions"))
      .def("histograms", [](const mpsboost::MpsBackend& backend, const mpsboost::BinnedDataset& dataset, const std::vector<double>& labels, const std::vector<double>& predictions, const std::vector<std::uint64_t>& rows) { return RunMpsHistogramForTest(
                                                                                                                                                                                                                                  backend, dataset, labels, predictions, rows, false); }, py::arg("dataset"), py::arg("labels"), py::arg("predictions"), py::arg("rows"))
      .def("baseline_histograms", [](const mpsboost::MpsBackend& backend, const mpsboost::BinnedDataset& dataset, const std::vector<double>& labels, const std::vector<double>& predictions, const std::vector<std::uint64_t>& rows) { return RunMpsHistogramForTest(
                                                                                                                                                                                                                                           backend, dataset, labels, predictions, rows, true); }, py::arg("dataset"), py::arg("labels"), py::arg("predictions"), py::arg("rows"))
      .def("split_candidates", [](const mpsboost::MpsBackend& backend, const mpsboost::BinnedDataset& dataset, const std::vector<double>& labels, const std::vector<double>& predictions, const std::vector<std::uint64_t>& rows, std::uint64_t min_samples_leaf, double min_child_weight, double reg_lambda, double gamma) {
        const std::vector<mpsboost::GradientPair> gradients =
            mpsboost::ComputeSquaredErrorGradients(labels, predictions);
        return SplitCandidatesToPython(backend.ScanSplitsForTest(
            dataset, rows, gradients, min_samples_leaf, min_child_weight,
            reg_lambda, gamma)); }, py::arg("dataset"), py::arg("labels"), py::arg("predictions"), py::arg("rows"), py::arg("min_samples_leaf") = 1, py::arg("min_child_weight") = 0.0, py::arg("reg_lambda") = 1.0, py::arg("gamma") = 0.0)
      .def("partition_rows", [](const mpsboost::MpsBackend& backend, const mpsboost::BinnedDataset& dataset, const std::vector<std::uint64_t>& rows, std::uint32_t feature, std::uint32_t threshold_bin) {
        const auto parts =
            backend.PartitionRowsForTest(dataset, rows, feature, threshold_bin);
        return py::make_tuple(parts.first, parts.second); }, py::arg("dataset"), py::arg("rows"), py::arg("feature"), py::arg("threshold_bin"))
      .def_property_readonly("last_timing", [](const mpsboost::MpsBackend& backend) {
        const mpsboost::BackendTiming timing = backend.last_timing();
        py::dict result;
        result["gradient_seconds"] = timing.gradient_seconds;
        result["histogram_encode_seconds"] = timing.histogram_encode_seconds;
        result["histogram_command_seconds"] = timing.histogram_command_seconds;
        result["hot_path_encode_seconds"] = timing.hot_path_encode_seconds;
        result["hot_path_command_seconds"] = timing.hot_path_command_seconds;
        result["pooled_buffer_reuse_count"] = timing.pooled_buffer_reuse_count;
        result["pooled_buffer_allocation_count"] =
            timing.pooled_buffer_allocation_count;
        return result; });

  py::class_<mpsboost::RegressionModel>(module, "_RegressionModel")
      .def_property_readonly("feature_count",
                             &mpsboost::RegressionModel::feature_count)
      .def_property_readonly("tree_count",
                             &mpsboost::RegressionModel::tree_count)
      .def("predict", [](const mpsboost::RegressionModel& model, const py::buffer& matrix) {
             const mpsboost::DenseMatrixView view = MakeDenseView(matrix);
             py::gil_scoped_release release;
             const mpsboost::BinnedDataset dataset =
                 mpsboost::TransformDense(view, model.schema());
             return model.Predict(dataset); }, py::arg("matrix"), "使用模型冻结的分箱边界执行批量预测。")
      .def("save", &mpsboost::SaveModelFile, py::arg("path"), "使用版本化格式原子保存模型。");

  module.def("_load_regression_model", &mpsboost::LoadModelFile,
             py::arg("path"), "加载并完整验证版本化回归模型。");

  module.def(
      "_train_regressor_cpu",
      [](const py::buffer& matrix, const std::vector<double>& labels,
         const mpsboost::TrainingParameters& parameters) {
        const mpsboost::DenseMatrixView view = MakeDenseView(matrix);
        py::gil_scoped_release release;
        const mpsboost::BinnedDataset dataset =
            mpsboost::QuantizeDense(view, parameters.max_bins);
        const mpsboost::CpuReferenceBackend backend;
        return mpsboost::TrainRegressionModel(
            dataset, labels, parameters, backend, backend);
      },
      py::arg("matrix"), py::arg("labels"), py::arg("parameters"),
      "使用唯一 CPU oracle 训练多轮回归模型，仅供显式 CPU 模式和正确性对照。");

  module.def(
      "_train_regressor_mps",
      [](const py::buffer& matrix, const std::vector<double>& labels,
         const mpsboost::TrainingParameters& parameters,
         const std::string& metallib_path) {
        const mpsboost::DenseMatrixView view = MakeDenseView(matrix);
        py::gil_scoped_release release;
        const mpsboost::BinnedDataset dataset =
            mpsboost::QuantizeDense(view, parameters.max_bins);
        const mpsboost::MpsBackend backend(metallib_path);
        return mpsboost::TrainRegressionModel(
            dataset, labels, parameters, backend, backend);
      },
      py::arg("matrix"), py::arg("labels"), py::arg("parameters"),
      py::arg("metallib_path"),
      "在真实 MPS gradient/histogram 后端上训练多轮回归模型。");

  module.def(
      "_squared_error_gradients",
      [](const std::vector<double>& labels,
         const std::vector<double>& predictions) {
        py::list result;
        for (const mpsboost::GradientPair& pair :
             mpsboost::ComputeSquaredErrorGradients(labels, predictions)) {
          result.append(py::make_tuple(pair.gradient, pair.hessian));
        }
        return result;
      },
      py::arg("labels"), py::arg("predictions"),
      "计算平方误差 FP64 gradient/Hessian，仅供领域语义测试和后端对照。");

  module.def(
      "_binary_logistic_gradients",
      [](const std::vector<double>& labels, const std::vector<double>& logits) {
        py::list result;
        for (const mpsboost::GradientPair& pair :
             mpsboost::ComputeBinaryLogisticGradients(labels, logits)) {
          result.append(py::make_tuple(pair.gradient, pair.hessian));
        }
        return result;
      },
      py::arg("labels"), py::arg("logits"),
      "Compute binary-logistic FP64 gradient/Hessian for semantic tests.");

  module.def("_logistic_probability", &mpsboost::LogisticProbability,
             py::arg("logit"),
             "Convert a raw binary-logistic margin to probability.");

  module.def("_node_score", &mpsboost::NodeScore, py::arg("gradient_sum"),
             py::arg("hessian_sum"), py::arg("reg_lambda"),
             "调用唯一 C++ 节点分数公式。");
  module.def("_leaf_weight", &mpsboost::LeafWeight,
             py::arg("gradient_sum"), py::arg("hessian_sum"),
             py::arg("reg_lambda"), "调用唯一 C++ 叶值公式。");
  module.def("_split_gain", &mpsboost::SplitGain,
             py::arg("left_gradient"), py::arg("left_hessian"),
             py::arg("right_gradient"), py::arg("right_hessian"),
             py::arg("reg_lambda"), py::arg("gamma"),
             "调用唯一 C++ 切分增益公式。");

  module.def(
      "_cpu_histograms",
      [](const mpsboost::BinnedDataset& dataset,
         const std::vector<double>& labels,
         const std::vector<double>& predictions,
         const std::vector<std::uint64_t>& rows) {
        const std::vector<mpsboost::GradientPair> gradients =
            mpsboost::ComputeSquaredErrorGradients(labels, predictions);
        const mpsboost::CpuReferenceBackend backend;
        return HistogramsToPython(
            backend.BuildHistograms(dataset, rows, gradients));
      },
      py::arg("dataset"), py::arg("labels"), py::arg("predictions"),
      py::arg("rows"),
      "构建指定行集合的 CPU FP64 histogram，仅供逐 bin 正确性对照。");

  module.def(
      "_train_single_tree_cpu",
      [](const mpsboost::BinnedDataset& dataset,
         const std::vector<double>& labels,
         const std::vector<double>& predictions,
         std::uint32_t max_depth,
         std::uint64_t min_samples_leaf,
         double min_child_weight,
         double reg_lambda,
         double gamma) {
        const std::vector<mpsboost::GradientPair> gradients =
            mpsboost::ComputeSquaredErrorGradients(labels, predictions);
        const mpsboost::TreeTrainingParameters parameters{
            max_depth, min_samples_leaf, min_child_weight, reg_lambda, gamma};
        const mpsboost::CpuReferenceBackend backend;
        return mpsboost::TrainSingleRegressionTree(dataset, gradients, parameters,
                                                   backend);
      },
      py::arg("dataset"), py::arg("labels"), py::arg("predictions"),
      py::arg("max_depth"), py::arg("min_samples_leaf") = 1,
      py::arg("min_child_weight") = 0.0, py::arg("reg_lambda") = 1.0,
      py::arg("gamma") = 0.0,
      "使用 CPU FP64 histogram oracle 训练一棵真实深度受限回归树。");

  module.def(
      "_quantize_dense",
      [](const py::buffer& matrix, std::uint32_t max_bins) {
        return mpsboost::QuantizeDense(MakeDenseView(matrix), max_bins);
      },
      py::arg("matrix"), py::arg("max_bins") = 256,
      "同步量化二维 float buffer，返回拥有自身内存的内部数据集。");

  module.def(
      "_validate_dense_view",
      [](std::uint64_t rows, std::uint32_t features,
         std::uint64_t row_stride_bytes, std::uint64_t column_stride_bytes,
         std::uint32_t scalar_bytes, std::uint32_t max_bins) {
        // 静态哨兵只用于满足非空指针不变量；ValidateDenseView 不会读取它。该内部入口
        // 可以验证无法实际分配的超大元数据，而不是通过制造危险 buffer 测试溢出。
        static const double sentinel = 0.0;
        mpsboost::ScalarType scalar_type;
        if (scalar_bytes == sizeof(float)) {
          scalar_type = mpsboost::ScalarType::kFloat32;
        } else if (scalar_bytes == sizeof(double)) {
          scalar_type = mpsboost::ScalarType::kFloat64;
        } else {
          throw py::value_error("scalar_bytes 必须是 4 或 8");
        }
        mpsboost::ValidateDenseView(
            mpsboost::DenseMatrixView{&sentinel, rows, features, row_stride_bytes,
                                      column_stride_bytes, scalar_type, false},
            max_bins);
      },
      py::arg("rows"), py::arg("features"), py::arg("row_stride_bytes"),
      py::arg("column_stride_bytes"), py::arg("scalar_bytes"),
      py::arg("max_bins"),
      "验证不读取内存的二维 view 元数据，仅用于安全边界测试。");

  module.def(
      "_deserialize_binned",
      [](const py::bytes& serialized) {
        const std::string bytes = serialized;
        return mpsboost::BinnedDataset::Deserialize(
            std::vector<std::uint8_t>(bytes.begin(), bytes.end()));
      },
      py::arg("serialized"), "反序列化并完整验证内部量化数据集。");

  module.def(
      "_backend_info",
      []() {
        const mpsboost::BackendInfo info = mpsboost::QueryBackendInfo();
        py::dict result;
        result["available"] = info.available;
        result["device_name"] = info.available ? py::cast(info.device_name) : py::none();
        result["recommended_max_working_set_size"] =
            info.recommended_max_working_set_size;
        result["has_unified_memory"] = info.has_unified_memory;
        return result;
      },
      "返回不包含敏感标识的真实 Metal 设备能力摘要。");

  module.def(
      "_vector_add",
      &mpsboost::RunVectorAdd,
      py::arg("left"),
      py::arg("right"),
      py::arg("metallib_path"),
      "在真实 GPU 上执行 smoke 向量加法，仅用于后端集成验证。");
}
