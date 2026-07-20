// Internal trainer helpers shared by regression and multiclass boosting units.
//
// Intent: keep validation and sample-weight semantics in one native place while
// allowing trainer.cpp and multiclass_trainer.cpp to stay below maintenance
// length limits.
#pragma once

#include <vector>

#include "mpsboost/objective.hpp"
#include "mpsboost/trainer.hpp"

namespace mpsboost::trainer_internal {

void ValidateTrainingParameters(const TrainingParameters& parameters);

double ValidateWeightsAndTotal(const std::vector<double>& labels,
                               const std::vector<double>& sample_weights);

std::vector<GradientPair> ApplySampleWeights(
    std::vector<GradientPair> gradients,
    const std::vector<double>& sample_weights);

}  // namespace mpsboost::trainer_internal
