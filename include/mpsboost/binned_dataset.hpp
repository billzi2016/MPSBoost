// MPSBoost quantized-data domain model.
//
// Responsibility: defines device-independent two-dimensional numeric views,
// deterministic binning results, compact storage, and stable serialization APIs.
// This file does not depend on Python, Metal, or the file system; CPU oracle and
// MPS backends must share the same result.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <variant>
#include <vector>

namespace mpsboost {

// Unified exception for input or serialized data that violates domain contracts.
class DataError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

enum class ScalarType : std::uint8_t { kFloat32 = 1,
                                       kFloat64 = 2 };
enum class BinStorage : std::uint8_t { kUInt8 = 1,
                                       kUInt16 = 2 };

// Read-only view borrowing an external two-dimensional buffer. Quantization reads
// the pointer only during the synchronous call and never retains it.
struct DenseMatrixView final {
  const void* data{nullptr};
  std::uint64_t rows{0};
  std::uint32_t features{0};
  std::uint64_t row_stride_bytes{0};
  std::uint64_t column_stride_bytes{0};
  ScalarType scalar_type{ScalarType::kFloat32};
  bool source_contiguous{false};
};

// A feature's range in the global boundary array and its actual bin count.
struct FeatureBinMetadata final {
  std::uint64_t boundary_offset{0};
  std::uint32_t boundary_count{0};
  std::uint32_t bin_count{0};
  std::uint64_t missing_count{0};
};

class BinnedDataset;

// Binning rules frozen during training. The schema stores only boundaries and
// layout, never training samples, so it can safely enter model files and predict
// new data. Prediction must never refit quantile boundaries.
class QuantizationSchema final {
 public:
  std::uint32_t features() const noexcept { return features_; }
  std::uint32_t max_bins() const noexcept { return max_bins_; }
  BinStorage storage() const noexcept { return storage_; }
  const std::vector<float>& boundaries() const noexcept { return boundaries_; }
  const std::vector<FeatureBinMetadata>& feature_metadata() const noexcept {
    return feature_metadata_;
  }

 private:
  friend class BinnedDataset;
  friend BinnedDataset QuantizeDense(const DenseMatrixView&, std::uint32_t);
  friend BinnedDataset TransformDense(const DenseMatrixView&,
                                      const QuantizationSchema&);
  friend QuantizationSchema RestoreQuantizationSchema(
      std::uint32_t,
      std::uint32_t,
      std::vector<float>,
      std::vector<FeatureBinMetadata>);

  std::uint32_t features_{0};
  std::uint32_t max_bins_{0};
  BinStorage storage_{BinStorage::kUInt8};
  std::vector<float> boundaries_;
  std::vector<FeatureBinMetadata> feature_metadata_;
};

class BinnedDataset final {
 public:
  std::uint64_t rows() const noexcept { return rows_; }
  std::uint32_t features() const noexcept { return schema_.features(); }
  std::uint32_t max_bins() const noexcept { return schema_.max_bins(); }
  BinStorage storage() const noexcept { return schema_.storage(); }
  bool source_contiguous() const noexcept { return source_contiguous_; }

  // Quantization reads the strided view directly without a full normalized input
  // copy; this value is for diagnostics and tests.
  bool source_was_copied() const noexcept { return false; }

  const std::vector<float>& boundaries() const noexcept {
    return schema_.boundaries();
  }
  const std::vector<FeatureBinMetadata>& feature_metadata() const noexcept {
    return schema_.feature_metadata();
  }
  const QuantizationSchema& schema() const noexcept { return schema_; }

  // Return the read-only address and element count of the compact feature-major
  // bin buffer. The address is valid only while this object is alive and unmoved;
  // MPS must copy synchronously or finish its command before returning.
  const void* bin_data() const noexcept;
  std::uint64_t bin_value_count() const noexcept;

  // Read one bin from feature-major compact storage. Invalid indices fail
  // explicitly and must not produce out-of-bounds reads.
  std::uint32_t GetBin(std::uint64_t row, std::uint32_t feature) const;
  bool IsMissing(std::uint64_t row, std::uint32_t feature) const;

  // Produce deterministic versioned bytes for round-trip tests and future cache reuse.
  std::vector<std::uint8_t> Serialize() const;
  static BinnedDataset Deserialize(const std::vector<std::uint8_t>& bytes);

 private:
  friend BinnedDataset QuantizeDense(const DenseMatrixView&, std::uint32_t);
  friend BinnedDataset TransformDense(const DenseMatrixView&,
                                      const QuantizationSchema&);

  std::uint64_t rows_{0};
  bool source_contiguous_{false};
  QuantizationSchema schema_;
  std::variant<std::vector<std::uint8_t>, std::vector<std::uint16_t>> bins_;
  std::vector<std::uint8_t> missing_;
};

// Quantize a finite float32/float64 matrix into an owning feature-major dataset.
BinnedDataset QuantizeDense(const DenseMatrixView& view, std::uint32_t max_bins);

// Transform a new matrix with the training schema. This entry applies only existing
// lower_bound boundaries, reads no labels, and never re-estimates quantiles, keeping
// routing semantics identical for fit, predict, and loaded models.
BinnedDataset TransformDense(const DenseMatrixView& view,
                             const QuantizationSchema& schema);

// Restore a schema from validated model fields. This sole entry checks every offset,
// boundary monotonicity, and bin count; model loaders must not bypass validation.
QuantizationSchema RestoreQuantizationSchema(
    std::uint32_t features,
    std::uint32_t max_bins,
    std::vector<float> boundaries,
    std::vector<FeatureBinMetadata> metadata);

// Validate only view metadata and accessible range without reading input or
// allocating binned memory. Quantization and boundary tests share this one
// validation path to prevent false coverage from simplified test rules.
void ValidateDenseView(const DenseMatrixView& view, std::uint32_t max_bins);

}  // namespace mpsboost
