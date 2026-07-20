// MPSBoost GPU split-scan kernel.
//
// Responsibility: scans candidate thresholds over completed feature histograms and
// emits the best candidate for each feature. The C++ training core still confirms the
// final split under its frozen FP64 rules; this kernel only removes repeated prefix
// scans from the hot path and owns neither tree-growth control flow nor defaults.

#include <metal_stdlib>

using namespace metal;

struct HistogramValue {
  uint count;
  float gradient;
  float hessian;
  uint reserved;
};

struct SplitCandidateValue {
  uint valid;
  uint feature;
  uint threshold_bin;
  uint left_count;
  uint right_count;
  float left_gradient;
  float left_hessian;
  float right_gradient;
  float right_hessian;
  float gain;
};

inline float NodeScore(float gradient, float hessian, float reg_lambda) {
  return (gradient * gradient) / (hessian + reg_lambda);
}

kernel void split_scan_features(
    device const HistogramValue* histogram [[buffer(0)]],
    device const uint* feature_offsets [[buffer(1)]],
    device const uint* feature_bin_counts [[buffer(2)]],
    device SplitCandidateValue* output [[buffer(3)]],
    constant uint& feature_count [[buffer(4)]],
    constant uint& min_samples_leaf [[buffer(5)]],
    constant float& min_child_weight [[buffer(6)]],
    constant float& reg_lambda [[buffer(7)]],
    constant float& gamma [[buffer(8)]],
    uint feature [[thread_position_in_grid]]) {
  if (feature >= feature_count) {
    return;
  }

  const uint offset = feature_offsets[feature];
  const uint bin_count = feature_bin_counts[feature];
  SplitCandidateValue best{0, feature, 0, 0, 0, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  if (bin_count < 2) {
    output[feature] = best;
    return;
  }

  uint parent_count = 0;
  float parent_gradient = 0.0F;
  float parent_hessian = 0.0F;
  for (uint bin = 0; bin < bin_count; ++bin) {
    const HistogramValue value = histogram[offset + bin];
    parent_count += value.count;
    parent_gradient += value.gradient;
    parent_hessian += value.hessian;
  }

  uint left_count = 0;
  float left_gradient = 0.0F;
  float left_hessian = 0.0F;
  const float parent_score = NodeScore(parent_gradient, parent_hessian, reg_lambda);
  for (uint threshold = 0; threshold + 1 < bin_count; ++threshold) {
    const HistogramValue value = histogram[offset + threshold];
    left_count += value.count;
    left_gradient += value.gradient;
    left_hessian += value.hessian;

    const uint right_count = parent_count - left_count;
    const float right_gradient = parent_gradient - left_gradient;
    const float right_hessian = parent_hessian - left_hessian;
    if (left_count < min_samples_leaf || right_count < min_samples_leaf ||
        left_hessian < min_child_weight || right_hessian < min_child_weight ||
        left_hessian <= 0.0F || right_hessian <= 0.0F) {
      continue;
    }

    const float gain = 0.5F * (NodeScore(left_gradient, left_hessian, reg_lambda) +
                               NodeScore(right_gradient, right_hessian, reg_lambda) -
                               parent_score) -
                       gamma;
    if (gain <= 0.0F) {
      continue;
    }
    if (best.valid == 0 || gain > best.gain ||
        (gain == best.gain && threshold < best.threshold_bin)) {
      best = SplitCandidateValue{1,
                                 feature,
                                 threshold,
                                 left_count,
                                 right_count,
                                 left_gradient,
                                 left_hessian,
                                 right_gradient,
                                 right_hessian,
                                 gain};
    }
  }
  output[feature] = best;
}
