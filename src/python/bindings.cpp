// Minimal pybind module entry point for MPSBoost.
//
// Registration details are split by binding family so this file only owns module metadata,
// exception registration, and registration sequencing.

#include <pybind11/pybind11.h>

#include "binding_registrations.hpp"
#include "mpsboost/backend.hpp"
#include "mpsboost/objective.hpp"
#include "mpsboost/version.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_native, module) {
  module.doc() = "MPSBoost native MPS/Metal backend";
  module.attr("__version__") = MPSBOOST_VERSION;

  py::register_exception<mpsboost::BackendError>(module, "BackendError");
  py::register_exception<mpsboost::DataError>(module, "DataError", PyExc_ValueError);
  py::register_exception<mpsboost::TrainingError>(module, "TrainingError",
                                                  PyExc_ValueError);

  mpsboost::python_binding::RegisterDatasetBindings(module);
  mpsboost::python_binding::RegisterBackendBindings(module);
  mpsboost::python_binding::RegisterModelBindings(module);
  mpsboost::python_binding::RegisterObjectiveBindings(module);
  mpsboost::python_binding::RegisterDatasetToolBindings(module);
}
