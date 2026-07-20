// Dataset, tree, and training-parameter pybind registrations.
//
// This file owns structural native value bindings only. Training entry points and backend test
// helpers live in separate registration files.

#include "binding_registrations.hpp"

#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "binding_helpers.hpp"
#include "mpsboost/binned_dataset.hpp"
#include "mpsboost/trainer.hpp"
#include "mpsboost/tree.hpp"

namespace mpsboost::python_binding {

namespace {

TrainingParameters MakeTrainingParameters(std::uint32_t n_estimators,
                                          double learning_rate,
                                          std::uint32_t max_bins,
                                          std::uint32_t max_depth,
                                          std::uint64_t min_samples_leaf,
                                          double min_child_weight,
                                          double reg_lambda,
                                          double gamma,
                                          double min_gain_to_split,
                                          const std::string& objective,
                                          const std::string& split_strategy,
                                          const std::string& growth_strategy,
                                          std::uint32_t max_leaves,
                                          std::uint32_t max_active_leaves,
                                          std::uint32_t random_seed) {
  TrainingParameters::Objective objective_kind =
      TrainingParameters::Objective::kSquaredError;
  if (objective == "squared_error") {
    objective_kind = TrainingParameters::Objective::kSquaredError;
  } else if (objective == "binary_logistic") {
    objective_kind = TrainingParameters::Objective::kBinaryLogistic;
  } else {
    throw std::invalid_argument("unknown training objective");
  }

  TreeTrainingParameters::SplitStrategy split_kind =
      TreeTrainingParameters::SplitStrategy::kBestGain;
  if (split_strategy == "best_gain") {
    split_kind = TreeTrainingParameters::SplitStrategy::kBestGain;
  } else if (split_strategy == "random_threshold") {
    split_kind = TreeTrainingParameters::SplitStrategy::kRandomThreshold;
  } else {
    throw std::invalid_argument("unknown split strategy");
  }

  TreeTrainingParameters::GrowthStrategy growth_kind =
      TreeTrainingParameters::GrowthStrategy::kLevelWise;
  if (growth_strategy == "level_wise") {
    growth_kind = TreeTrainingParameters::GrowthStrategy::kLevelWise;
  } else if (growth_strategy == "leaf_wise") {
    growth_kind = TreeTrainingParameters::GrowthStrategy::kLeafWise;
  } else {
    throw std::invalid_argument("unknown growth strategy");
  }

  return TrainingParameters{n_estimators,
                            learning_rate,
                            max_bins,
                            objective_kind,
                            TreeTrainingParameters{max_depth,
                                                   max_leaves,
                                                   max_active_leaves,
                                                   min_samples_leaf,
                                                   min_child_weight,
                                                   reg_lambda,
                                                   gamma,
                                                   min_gain_to_split,
                                                   split_kind,
                                                   growth_kind,
                                                   random_seed}};
}

}  // namespace

void RegisterDatasetBindings(py::module_& module) {
  py::class_<BinnedDataset>(module, "_BinnedDataset")
      .def_property_readonly("rows", &BinnedDataset::rows)
      .def_property_readonly("features", &BinnedDataset::features)
      .def_property_readonly("max_bins", &BinnedDataset::max_bins)
      .def_property_readonly("bin_width", [](const BinnedDataset& dataset) {
        return dataset.storage() == BinStorage::kUInt8 ? 8 : 16;
      })
      .def_property_readonly("source_contiguous",
                             &BinnedDataset::source_contiguous)
      .def_property_readonly("source_was_copied",
                             &BinnedDataset::source_was_copied)
      .def_property_readonly("boundaries", &BoundariesByFeature)
      .def_property_readonly("bins", &BinsByFeature)
      .def_property_readonly("missing", &MissingByFeature)
      .def("serialize", [](const BinnedDataset& dataset) {
        const std::vector<std::uint8_t> bytes = dataset.Serialize();
        return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      });

  py::class_<RegressionTree>(module, "_RegressionTree")
      .def_property_readonly("feature_count", &RegressionTree::feature_count)
      .def_property_readonly("nodes", &TreeNodes)
      .def("predict", &RegressionTree::Predict, py::arg("dataset"),
           "Run deterministic flat-tree prediction on an internal binned dataset.");

  py::class_<TrainingParameters>(module, "_TrainingParameters")
      .def(py::init(&MakeTrainingParameters), py::arg("n_estimators"),
           py::arg("learning_rate"), py::arg("max_bins"),
           py::arg("max_depth"), py::arg("min_samples_leaf"),
           py::arg("min_child_weight"), py::arg("reg_lambda"),
           py::arg("gamma") = 0.0, py::arg("min_gain_to_split") = 0.0,
           py::arg("objective") = "squared_error",
           py::arg("split_strategy") = "best_gain",
           py::arg("growth_strategy") = "level_wise",
           py::arg("max_leaves") = 0,
           py::arg("max_active_leaves") = 0,
           py::arg("random_seed") = 0,
           "Create the internal named training-parameter value object.");
}

}  // namespace mpsboost::python_binding
