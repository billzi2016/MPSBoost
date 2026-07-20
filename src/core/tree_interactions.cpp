// Interaction-constraint checks for native regression trees.
//
// Interaction constraints are path constraints: every feature used along a
// root-to-leaf path must fit inside at least one declared group. This unit owns
// that policy so split scanning and growth loops only ask whether a feature is
// legal for the current path.

#include "tree_internal.hpp"

#include <algorithm>

namespace mpsboost::tree_internal {
namespace {

bool GroupContains(const std::vector<std::uint32_t>& group,
                   std::uint32_t feature) {
  return std::find(group.begin(), group.end(), feature) != group.end();
}

bool GroupContainsAll(const std::vector<std::uint32_t>& group,
                      const std::vector<std::uint32_t>& features) {
  for (const std::uint32_t feature : features) {
    if (!GroupContains(group, feature)) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool InteractionAllowsFeature(const std::vector<std::uint32_t>& path_features,
                              std::uint32_t feature,
                              const TreeTrainingParameters& parameters) {
  if (parameters.interaction_constraints.empty()) {
    return true;
  }
  for (const std::vector<std::uint32_t>& group :
       parameters.interaction_constraints) {
    if (GroupContains(group, feature) && GroupContainsAll(group, path_features)) {
      return true;
    }
  }
  return false;
}

std::vector<std::uint32_t> ExtendInteractionPath(
    const std::vector<std::uint32_t>& path_features,
    std::uint32_t feature) {
  std::vector<std::uint32_t> result = path_features;
  if (std::find(result.begin(), result.end(), feature) == result.end()) {
    result.push_back(feature);
  }
  return result;
}

void ValidateInteractionConstraints(const TreeTrainingParameters& parameters,
                                    std::uint32_t feature_count) {
  for (const std::vector<std::uint32_t>& group :
       parameters.interaction_constraints) {
    if (group.empty()) {
      throw TrainingError("interaction constraint groups must be non-empty");
    }
    for (const std::uint32_t feature : group) {
      if (feature >= feature_count) {
        throw TrainingError("interaction constraint feature index out of range");
      }
    }
  }
}

}  // namespace mpsboost::tree_internal
