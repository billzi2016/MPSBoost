// Objective and single-tree oracle pybind registrations.
//
// These bindings expose the native mathematical contract for tests. They call the production C++
// functions directly and must not duplicate formulas in binding code.

#include "binding_registrations.hpp"

#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "binding_helpers.hpp"
#include "mpsboost/backend.hpp"
#include "mpsboost/objective.hpp"
#include "mpsboost/tree.hpp"

namespace mpsboost::python_binding {

namespace {

RegressionTree TrainSingleTreeCpu(const BinnedDataset& dataset,
                                  const std::vector<double>& labels,
                                  const std::vector<double>& predictions,
                                  std::uint32_t max_depth,
                                  std::uint64_t min_samples_leaf,
                                  double min_child_weight,
                                  double reg_lambda,
                                  double gamma,
                                  double min_gain_to_split,
                                  const std::string& split_strategy,
                                  const std::string& growth_strategy,
                                  std::uint32_t max_leaves,
                                  std::uint32_t max_active_leaves,
                                  std::uint32_t random_seed,
                                  const std::vector<std::int8_t>&
                                      monotonic_constraints,
                                  const std::vector<std::vector<std::uint32_t>>&
                                      interaction_constraints) {
  const std::vector<GradientPair> gradients =
      ComputeSquaredErrorGradients(labels, predictions);
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
  const TreeTrainingParameters parameters{max_depth, max_leaves,
                                          max_active_leaves, min_samples_leaf,
                                          min_child_weight, reg_lambda, gamma,
                                          min_gain_to_split, split_kind,
                                          growth_kind, random_seed,
                                          monotonic_constraints,
                                          interaction_constraints};
  const CpuReferenceBackend backend;
  return TrainSingleRegressionTree(dataset, gradients, parameters, backend);
}

}  // namespace

void RegisterObjectiveBindings(py::module_& module) {
  module.def("_squared_error_gradients", [](const std::vector<double>& labels,
                                             const std::vector<double>& predictions) {
    return GradientsToPython(ComputeSquaredErrorGradients(labels, predictions));
  }, py::arg("labels"), py::arg("predictions"),
     "Compute squared-error FP64 gradient/Hessian pairs for tests.");

  module.def("_binary_logistic_gradients", [](const std::vector<double>& labels,
                                               const std::vector<double>& logits) {
    return GradientsToPython(ComputeBinaryLogisticGradients(labels, logits));
  }, py::arg("labels"), py::arg("logits"),
     "Compute binary-logistic FP64 gradient/Hessian pairs for tests.");

  module.def("_logistic_probability", &LogisticProbability, py::arg("logit"),
             "Convert a raw binary-logistic margin to probability.");
  module.def("_node_score", &NodeScore, py::arg("gradient_sum"),
             py::arg("hessian_sum"), py::arg("reg_lambda"),
             "Call the single native node-score formula.");
  module.def("_leaf_weight", &LeafWeight, py::arg("gradient_sum"),
             py::arg("hessian_sum"), py::arg("reg_lambda"),
             "Call the single native leaf-weight formula.");
  module.def("_split_gain", &SplitGain, py::arg("left_gradient"),
             py::arg("left_hessian"), py::arg("right_gradient"),
             py::arg("right_hessian"), py::arg("reg_lambda"),
             py::arg("gamma"), "Call the single native split-gain formula.");

  module.def("_cpu_histograms", [](const BinnedDataset& dataset,
                                    const std::vector<double>& labels,
                                    const std::vector<double>& predictions,
                                    const std::vector<std::uint64_t>& rows) {
    const std::vector<GradientPair> gradients =
        ComputeSquaredErrorGradients(labels, predictions);
    const CpuReferenceBackend backend;
    return HistogramsToPython(backend.BuildHistograms(dataset, rows, gradients));
  }, py::arg("dataset"), py::arg("labels"), py::arg("predictions"),
     py::arg("rows"), "Build CPU FP64 histograms for exact test comparisons.");

  module.def("_train_single_tree_cpu", &TrainSingleTreeCpu, py::arg("dataset"),
             py::arg("labels"), py::arg("predictions"), py::arg("max_depth"),
             py::arg("min_samples_leaf") = 1,
             py::arg("min_child_weight") = 0.0,
             py::arg("reg_lambda") = 1.0, py::arg("gamma") = 0.0,
             py::arg("min_gain_to_split") = 0.0,
             py::arg("split_strategy") = "best_gain",
             py::arg("growth_strategy") = "level_wise",
             py::arg("max_leaves") = 0,
             py::arg("max_active_leaves") = 0,
             py::arg("random_seed") = 0,
             py::arg("monotonic_constraints") = std::vector<std::int8_t>{},
             py::arg("interaction_constraints") =
                 std::vector<std::vector<std::uint32_t>>{},
             "Train one real CPU-oracle depth-limited regression tree.");
}

}  // namespace mpsboost::python_binding
