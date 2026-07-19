// MPSBoost Python 原生绑定。
//
// 职责：把稳定 C++ POD 和异常转换为轻量 Python 对象。这里不得复制设备实现、参数
// 校验或训练算法。所有公开用户体验由 Python 层组织，以下下划线入口属于内部接口。

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <limits>

#include "mpsboost/backend.hpp"
#include "mpsboost/binned_dataset.hpp"
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

}  // namespace

PYBIND11_MODULE(_native, module) {
  module.doc() = "MPSBoost 原生 MPS/Metal 后端";
  module.attr("__version__") = MPSBOOST_VERSION;

  // 统一注册后端异常，使 Python 可以明确区分设备执行失败与普通参数错误。
  py::register_exception<mpsboost::BackendError>(module, "BackendError");
  py::register_exception<mpsboost::DataError>(module, "DataError", PyExc_ValueError);

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
