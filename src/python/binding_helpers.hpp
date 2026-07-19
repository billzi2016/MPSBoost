// Shared pybind helper declarations for MPSBoost native bindings.
//
// These helpers convert native diagnostic values to Python objects. They do not register module
// symbols and do not own training, device, or serialization policy.
#pragma once

#include <pybind11/pybind11.h>

#include <vector>

#include "mpsboost/backend.hpp"
#include "mpsboost/binned_dataset.hpp"
#include "mpsboost/tree.hpp"
#include "mpsboost/trainer.hpp"

namespace mpsboost::python_binding {

namespace py = pybind11;

DenseMatrixView MakeDenseView(const py::buffer& matrix);
py::list BoundariesByFeature(const BinnedDataset& dataset);
py::list BinsByFeature(const BinnedDataset& dataset);
py::list TreeNodes(const RegressionTree& tree);
py::list ModelTrees(const RegressionModel& model);
py::list HistogramsToPython(const NodeHistograms& histograms);
py::list GradientsToPython(const std::vector<GradientPair>& gradients);
py::list SplitCandidatesToPython(
    const std::vector<SplitScanCandidate>& candidates);
py::object RunMpsHistogramForTest(const MpsBackend& backend,
                                  const BinnedDataset& dataset,
                                  const std::vector<double>& labels,
                                  const std::vector<double>& predictions,
                                  const std::vector<std::uint64_t>& rows,
                                  bool baseline);

}  // namespace mpsboost::python_binding
