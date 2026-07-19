// MPS backend diagnostic pybind registrations.
//
// These bindings expose real backend test hooks. They do not provide public estimator behavior and
// must never mock Metal device availability.

#include "binding_registrations.hpp"

#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "binding_helpers.hpp"
#include "mpsboost/backend.hpp"

namespace mpsboost::python_binding {

void RegisterBackendBindings(py::module_& module) {
  py::class_<MpsBackend>(module, "_MpsBackend")
      .def(py::init<std::string>(), py::arg("metallib_path"),
           "Create a reusable real MPS test session.")
      .def("gradients", [](const MpsBackend& backend,
                           const std::vector<double>& labels,
                           const std::vector<double>& predictions) {
        return GradientsToPython(backend.ComputeSquaredError(labels, predictions));
      }, py::arg("labels"), py::arg("predictions"))
      .def("histograms", [](const MpsBackend& backend,
                             const BinnedDataset& dataset,
                             const std::vector<double>& labels,
                             const std::vector<double>& predictions,
                             const std::vector<std::uint64_t>& rows) {
        return RunMpsHistogramForTest(backend, dataset, labels, predictions, rows,
                                      false);
      }, py::arg("dataset"), py::arg("labels"), py::arg("predictions"),
         py::arg("rows"))
      .def("baseline_histograms", [](const MpsBackend& backend,
                                      const BinnedDataset& dataset,
                                      const std::vector<double>& labels,
                                      const std::vector<double>& predictions,
                                      const std::vector<std::uint64_t>& rows) {
        return RunMpsHistogramForTest(backend, dataset, labels, predictions, rows,
                                      true);
      }, py::arg("dataset"), py::arg("labels"), py::arg("predictions"),
         py::arg("rows"))
      .def("split_candidates", [](const MpsBackend& backend,
                                   const BinnedDataset& dataset,
                                   const std::vector<double>& labels,
                                   const std::vector<double>& predictions,
                                   const std::vector<std::uint64_t>& rows,
                                   std::uint64_t min_samples_leaf,
                                   double min_child_weight, double reg_lambda,
                                   double gamma) {
        const std::vector<GradientPair> gradients =
            ComputeSquaredErrorGradients(labels, predictions);
        return SplitCandidatesToPython(backend.ScanSplitsForTest(
            dataset, rows, gradients, min_samples_leaf, min_child_weight,
            reg_lambda, gamma));
      }, py::arg("dataset"), py::arg("labels"), py::arg("predictions"),
         py::arg("rows"), py::arg("min_samples_leaf") = 1,
         py::arg("min_child_weight") = 0.0, py::arg("reg_lambda") = 1.0,
         py::arg("gamma") = 0.0)
      .def("partition_rows", [](const MpsBackend& backend,
                                 const BinnedDataset& dataset,
                                 const std::vector<std::uint64_t>& rows,
                                 std::uint32_t feature,
                                 std::uint32_t threshold_bin) {
        const auto parts =
            backend.PartitionRowsForTest(dataset, rows, feature, threshold_bin);
        return py::make_tuple(parts.first, parts.second);
      }, py::arg("dataset"), py::arg("rows"), py::arg("feature"),
         py::arg("threshold_bin"))
      .def_property_readonly("last_timing", [](const MpsBackend& backend) {
        const BackendTiming timing = backend.last_timing();
        py::dict result;
        result["gradient_seconds"] = timing.gradient_seconds;
        result["histogram_encode_seconds"] = timing.histogram_encode_seconds;
        result["histogram_command_seconds"] = timing.histogram_command_seconds;
        result["hot_path_encode_seconds"] = timing.hot_path_encode_seconds;
        result["hot_path_command_seconds"] = timing.hot_path_command_seconds;
        result["pooled_buffer_reuse_count"] = timing.pooled_buffer_reuse_count;
        result["pooled_buffer_allocation_count"] =
            timing.pooled_buffer_allocation_count;
        return result;
      });

  module.def("_backend_info", []() {
    const BackendInfo info = QueryBackendInfo();
    py::dict result;
    result["available"] = info.available;
    result["device_name"] = info.available ? py::cast(info.device_name) : py::none();
    result["recommended_max_working_set_size"] =
        info.recommended_max_working_set_size;
    result["has_unified_memory"] = info.has_unified_memory;
    return result;
  }, "Return a real Metal backend capability summary without sensitive identifiers.");

  module.def("_vector_add", &RunVectorAdd, py::arg("left"), py::arg("right"),
             py::arg("metallib_path"),
             "Run a real GPU vector-add smoke kernel for backend validation.");
}

}  // namespace mpsboost::python_binding
