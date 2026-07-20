// Dataset utility pybind registrations for tests and internal Python adapters.
//
// These bindings expose quantization, dense-view validation, and serialized binned-dataset loading
// without mixing them into model or backend registrations.

#include "binding_registrations.hpp"

#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "binding_helpers.hpp"
#include "mpsboost/binned_dataset.hpp"

namespace mpsboost::python_binding {

void RegisterDatasetToolBindings(py::module_& module) {
  module.def("_quantize_dense", [](const py::buffer& matrix,
                                    std::uint32_t max_bins) {
    return QuantizeDense(MakeDenseView(matrix), max_bins);
  }, py::arg("matrix"), py::arg("max_bins") = 256,
     "Synchronously quantize a dense float buffer into an owned binned dataset.");

  module.def("_validate_dense_view", [](std::uint64_t rows,
                                         std::uint32_t features,
                                         std::uint64_t row_stride_bytes,
                                         std::uint64_t column_stride_bytes,
                                         std::uint32_t scalar_bytes,
                                         std::uint32_t max_bins) {
    static const double sentinel = 0.0;
    ScalarType scalar_type;
    if (scalar_bytes == sizeof(float)) {
      scalar_type = ScalarType::kFloat32;
    } else if (scalar_bytes == sizeof(double)) {
      scalar_type = ScalarType::kFloat64;
    } else {
      throw py::value_error("scalar_bytes must be 4 or 8");
    }
    ValidateDenseView(DenseMatrixView{&sentinel, rows, features,
                                      row_stride_bytes, column_stride_bytes,
                                      scalar_type, false},
                      max_bins);
  }, py::arg("rows"), py::arg("features"), py::arg("row_stride_bytes"),
     py::arg("column_stride_bytes"), py::arg("scalar_bytes"),
     py::arg("max_bins"), "Validate dense-view metadata without reading memory.");

  module.def("_deserialize_binned", [](const py::bytes& serialized) {
    const std::string bytes = serialized;
    return BinnedDataset::Deserialize(
        std::vector<std::uint8_t>(bytes.begin(), bytes.end()));
  }, py::arg("serialized"), "Deserialize and validate an internal binned dataset.");
}

}  // namespace mpsboost::python_binding
