// MPSBoost minimal stable compute-backend contract.
//
// This file declares only device-independent POD data and backend entries, exposing
// no Objective-C/Metal types. The training core depends on these abstractions rather
// than concrete devices, preserving dependency inversion and an independent CPU oracle.
#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mpsboost/binned_dataset.hpp"
#include "mpsboost/objective.hpp"

namespace mpsboost {

// Unified exception for backend execution failures. Device layers retain stage
// context rather than exposing Python users to uncontextualized native error codes.
class BackendError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Device capability summary safe to return to users. It intentionally excludes
// paths, usernames, and persistent identifiers usable for telemetry, providing
// only information needed to decide availability and estimate memory.
struct BackendInfo final {
  bool available{false};
  std::string device_name;
  std::uint64_t recommended_max_working_set_size{0};
  bool has_unified_memory{false};
};

// Deterministic second-order statistics for one feature bin. count uses unsigned
// 64-bit counting and G/H use FP64 so the CPU oracle provides a high-precision
// reference for device kernels independent of accumulation order.
struct HistogramBin final {
  std::uint64_t count{0};
  double gradient_sum{0.0};
  double hessian_sum{0.0};
};

using FeatureHistogram = std::vector<HistogramBin>;
using NodeHistograms = std::vector<FeatureHistogram>;

// Minimal interface for device objective execution. The training state machine
// requests only mathematical results and does not know whether gradients come from
// the CPU oracle or a Metal kernel, keeping device selection in application wiring.
class GradientComputer {
 public:
  virtual ~GradientComputer() = default;
  virtual std::vector<GradientPair> ComputeSquaredError(
      const std::vector<double>& labels,
      const std::vector<double>& predictions) const = 0;
};

// The training core currently depends only on histogram capability, avoiding an
// oversized backend interface for future methods. The S4 MPS implementation and
// CPU oracle must implement the same minimal contract; later capabilities extend it
// through interface segregation.
class HistogramBuilder {
 public:
  virtual ~HistogramBuilder() = default;

  // Build histograms for all features over a selected node's rows. Implementations
  // must not reorder rows or retain borrowed pointers; result shape must exactly
  // match dataset features and their bin_count values.
  virtual NodeHistograms BuildHistograms(
      const BinnedDataset& dataset,
      const std::vector<std::uint64_t>& rows,
      const std::vector<GradientPair>& gradients) const = 0;
};

// Sole CPU histogram oracle. It implements computation only, not tree growth,
// split selection, or parameter semantics, preventing separate CPU and MPS control flows.
class CpuReferenceBackend final : public GradientComputer,
                                  public HistogramBuilder {
 public:
  std::vector<GradientPair> ComputeSquaredError(
      const std::vector<double>& labels,
      const std::vector<double>& predictions) const override;

  NodeHistograms BuildHistograms(
      const BinnedDataset& dataset,
      const std::vector<std::uint64_t>& rows,
      const std::vector<GradientPair>& gradients) const override;
};

// Non-sensitive timing for the latest synchronized device work. It is for
// diagnostics and benchmarks only and never affects training results, cache keys,
// or scheduling decisions.
struct BackendTiming final {
  double gradient_seconds{0.0};
  double histogram_encode_seconds{0.0};
  double histogram_command_seconds{0.0};
  double hot_path_encode_seconds{0.0};
  double hot_path_command_seconds{0.0};
  std::uint64_t pooled_buffer_reuse_count{0};
  std::uint64_t pooled_buffer_allocation_count{0};
};

// GPU split-scan candidate for one feature. This structure carries only device scan
// results; the training core still validates the final split under unique FP64 rules,
// preventing parallel accumulation order from changing model semantics.
struct SplitScanCandidate final {
  bool valid{false};
  std::uint32_t feature{0};
  std::uint32_t threshold_bin{0};
  std::uint64_t left_count{0};
  std::uint64_t right_count{0};
  double left_gradient_sum{0.0};
  double left_hessian_sum{0.0};
  double right_gradient_sum{0.0};
  double right_hessian_sum{0.0};
  double gain{0.0};
};

// Optional layer-wise histogram capability. The training core detects this narrow
// interface with dynamic_cast; unsupported backends use the existing per-node
// HistogramBuilder, so no second training semantic path exists.
class LayerHistogramBuilder {
 public:
  virtual ~LayerHistogramBuilder() = default;

  // Build histograms for multiple active nodes in one layer. Each input node returns
  // independent NodeHistograms in node_rows order; implementations may batch GPU
  // encoding but must not change row-set contents.
  virtual std::vector<NodeHistograms> BuildLayerHistograms(
      const BinnedDataset& dataset,
      const std::vector<std::vector<std::uint64_t>>& node_rows,
      const std::vector<GradientPair>& gradients) const = 0;
};

// Real MPS compute backend. Objective-C/Metal objects remain hidden in Impl so this
// stable C++ header exposes no platform types. Objects are non-copyable but can reuse
// pipeline and command queue within one training session.
class MpsBackend final : public GradientComputer,
                         public HistogramBuilder,
                         public LayerHistogramBuilder {
 public:
  explicit MpsBackend(std::string metallib_path);
  ~MpsBackend() override;
  MpsBackend(MpsBackend&&) noexcept;
  MpsBackend& operator=(MpsBackend&&) noexcept;
  MpsBackend(const MpsBackend&) = delete;
  MpsBackend& operator=(const MpsBackend&) = delete;

  std::vector<GradientPair> ComputeSquaredError(
      const std::vector<double>& labels,
      const std::vector<double>& predictions) const override;
  NodeHistograms BuildHistograms(
      const BinnedDataset& dataset,
      const std::vector<std::uint64_t>& rows,
      const std::vector<GradientPair>& gradients) const override;
  std::vector<NodeHistograms> BuildLayerHistograms(
      const BinnedDataset& dataset,
      const std::vector<std::vector<std::uint64_t>>& node_rows,
      const std::vector<GradientPair>& gradients) const override;
  BackendTiming last_timing() const noexcept;

  // Reuse the same context/pipeline/command implementation to validate the wheel's
  // minimal GPU pipeline. Only internal test bindings call this method; it is not a
  // public Python numerical API.
  std::vector<float> RunVectorAddForTest(const std::vector<float>& left,
                                         const std::vector<float>& right) const;
  NodeHistograms BuildBaselineHistogramsForTest(
      const BinnedDataset& dataset,
      const std::vector<std::uint64_t>& rows,
      const std::vector<GradientPair>& gradients) const;
  std::vector<SplitScanCandidate> ScanSplitsForTest(
      const BinnedDataset& dataset,
      const std::vector<std::uint64_t>& rows,
      const std::vector<GradientPair>& gradients,
      std::uint64_t min_samples_leaf,
      double min_child_weight,
      double reg_lambda,
      double gamma) const;
  std::pair<std::vector<std::uint64_t>, std::vector<std::uint64_t>>
  PartitionRowsForTest(const BinnedDataset& dataset,
                       const std::vector<std::uint64_t>& rows,
                       std::uint32_t feature,
                       std::uint32_t threshold_bin) const;

 private:
  NodeHistograms BuildHistogramsInternal(
      const BinnedDataset& dataset,
      const std::vector<std::uint64_t>& rows,
      const std::vector<GradientPair>& gradients,
      bool baseline) const;
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Query the default Metal device. No device is a normal unavailable state and does
// not throw; only runtime initialization failure raises BackendError, keeping
// is_available() distinct from real faults.
BackendInfo QueryBackendInfo();

// Add vectors element by element on a real GPU to validate the full shader,
// pipeline, buffer, command, and synchronization path. This entry is only for
// backend integration tests, not a public numerical API.
std::vector<float> RunVectorAdd(const std::vector<float>& left,
                                const std::vector<float>& right,
                                const std::string& metallib_path);

}  // namespace mpsboost
