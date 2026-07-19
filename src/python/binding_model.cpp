// Regression model and full-model training pybind registrations.
//
// This file owns model prediction, model persistence, and full boosting training entry points for
// both CPU oracle and MPS backends.

#include "binding_registrations.hpp"

#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "binding_helpers.hpp"
#include "mpsboost/backend.hpp"
#include "mpsboost/binned_dataset.hpp"
#include "mpsboost/trainer.hpp"

namespace mpsboost::python_binding {

namespace {

RegressionModel TrainRegressorCpu(const py::buffer& matrix,
                                  const std::vector<double>& labels,
                                  const std::vector<double>& sample_weights,
                                  const TrainingParameters& parameters) {
  const DenseMatrixView view = MakeDenseView(matrix);
  py::gil_scoped_release release;
  const BinnedDataset dataset = QuantizeDense(view, parameters.max_bins);
  const CpuReferenceBackend backend;
  return TrainRegressionModel(dataset, labels, sample_weights, parameters,
                              backend, backend);
}

RegressionModel TrainRegressorMps(const py::buffer& matrix,
                                  const std::vector<double>& labels,
                                  const std::vector<double>& sample_weights,
                                  const TrainingParameters& parameters,
                                  const std::string& metallib_path) {
  const DenseMatrixView view = MakeDenseView(matrix);
  py::gil_scoped_release release;
  const BinnedDataset dataset = QuantizeDense(view, parameters.max_bins);
  const MpsBackend backend(metallib_path);
  return TrainRegressionModel(dataset, labels, sample_weights, parameters,
                              backend, backend);
}

}  // namespace

void RegisterModelBindings(py::module_& module) {
  py::class_<RegressionModel>(module, "_RegressionModel")
      .def_property_readonly("feature_count", &RegressionModel::feature_count)
      .def_property_readonly("tree_count", &RegressionModel::tree_count)
      .def_property_readonly("trees", &ModelTrees)
      .def_property_readonly("objective", [](const RegressionModel& model) {
        return model.objective() == TrainingParameters::Objective::kBinaryLogistic
                   ? "binary_logistic"
                   : "squared_error";
      })
      .def("predict", [](const RegressionModel& model, const py::buffer& matrix) {
        const DenseMatrixView view = MakeDenseView(matrix);
        py::gil_scoped_release release;
        const BinnedDataset dataset = TransformDense(view, model.schema());
        return model.Predict(dataset);
      }, py::arg("matrix"), "Predict using the model's frozen binning schema.")
      .def("save", &SaveModelFile, py::arg("path"),
           "Atomically save a versioned model file.");

  module.def("_load_regression_model", &LoadModelFile, py::arg("path"),
             "Load and fully validate a versioned regression model.");
  module.def("_train_regressor_cpu", &TrainRegressorCpu, py::arg("matrix"),
             py::arg("labels"), py::arg("sample_weights"),
             py::arg("parameters"),
             "Train a full boosting model with the CPU oracle backend.");
  module.def("_train_regressor_mps", &TrainRegressorMps, py::arg("matrix"),
             py::arg("labels"), py::arg("sample_weights"),
             py::arg("parameters"), py::arg("metallib_path"),
             "Train a full boosting model with the real MPS backend.");
}

}  // namespace mpsboost::python_binding
