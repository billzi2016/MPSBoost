// Registration function declarations for the MPSBoost pybind module.
//
// Each implementation file owns one binding family so bindings.cpp can remain a minimal module
// entry point and future changes do not accumulate in one giant file.
#pragma once

#include <pybind11/pybind11.h>

namespace mpsboost::python_binding {

namespace py = pybind11;

void RegisterDatasetBindings(py::module_& module);
void RegisterBackendBindings(py::module_& module);
void RegisterModelBindings(py::module_& module);
void RegisterObjectiveBindings(py::module_& module);
void RegisterDatasetToolBindings(py::module_& module);

}  // namespace mpsboost::python_binding
