// Monotonic split-bound propagation for native regression trees.
//
// This unit keeps constraint-specific leaf-bound math out of the generic split
// scanner. The split scanner still owns gain ranking and missing-value routing.

#include "tree_internal.hpp"

#include <algorithm>

namespace mpsboost::tree_internal {

MonotonicChildBounds MonotonicBoundsForSplit(
    const NodeStatistics& left,
    const NodeStatistics& right,
    double parent_lower_bound,
    double parent_upper_bound,
    std::uint32_t feature,
    const TreeTrainingParameters& parameters) {
  MonotonicChildBounds bounds;
  bounds.left_lower_bound = parent_lower_bound;
  bounds.left_upper_bound = parent_upper_bound;
  bounds.right_lower_bound = parent_lower_bound;
  bounds.right_upper_bound = parent_upper_bound;
  const std::int8_t constraint =
      parameters.monotonic_constraints.empty()
          ? 0
          : parameters.monotonic_constraints[feature];
  if (constraint == 0) {
    return bounds;
  }

  const double left_value =
      std::clamp(LeafWeight(left.gradient_sum, left.hessian_sum,
                            parameters.reg_lambda, parameters.reg_alpha,
                            parameters.max_delta_step),
                 parent_lower_bound, parent_upper_bound);
  const double right_value =
      std::clamp(LeafWeight(right.gradient_sum, right.hessian_sum,
                            parameters.reg_lambda, parameters.reg_alpha,
                            parameters.max_delta_step),
                 parent_lower_bound, parent_upper_bound);
  if (constraint > 0) {
    if (left_value > right_value) {
      bounds.valid = false;
      return bounds;
    }
    bounds.left_upper_bound = std::min(bounds.left_upper_bound, right_value);
    bounds.right_lower_bound = std::max(bounds.right_lower_bound, left_value);
    return bounds;
  }
  if (left_value < right_value) {
    bounds.valid = false;
    return bounds;
  }
  bounds.left_lower_bound = std::max(bounds.left_lower_bound, right_value);
  bounds.right_upper_bound = std::min(bounds.right_upper_bound, left_value);
  return bounds;
}

}  // namespace mpsboost::tree_internal
