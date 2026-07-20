// Deterministic dense-matrix quantization for MPSBoost.
//
// This file owns dense-view validation, quantile boundary construction, schema restoration,
// transform-time bin mapping, and read-only binned dataset accessors. Binary serialization lives in
// binned_dataset_serialization.cpp.

#include "mpsboost/binned_dataset.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "binned_dataset_internal.hpp"

namespace mpsboost {
namespace {

void ValidateViewImpl(const DenseMatrixView& view, std::uint32_t max_bins) {
  if (view.data == nullptr) {
    throw DataError("输入矩阵 data 不能为空");
  }
  if (view.rows == 0 || view.features == 0) {
    throw DataError("输入矩阵必须至少包含一行和一个特征");
  }
  if (max_bins < 2 || max_bins > 65536) {
    throw DataError("max_bins 必须位于 [2, 65536]");
  }

  const std::uint64_t item_size = internal::ScalarByteSize(view.scalar_type);
  if (view.row_stride_bytes < item_size || view.column_stride_bytes < item_size) {
    throw DataError("输入矩阵 stride 小于元素字节数");
  }

  const std::uint64_t last_row = internal::CheckedMultiply(
      view.rows - 1, view.row_stride_bytes, "输入矩阵 row stride");
  const std::uint64_t last_column = internal::CheckedMultiply(
      static_cast<std::uint64_t>(view.features - 1), view.column_stride_bytes,
      "输入矩阵 column stride");
  const std::uint64_t last_offset = internal::CheckedAdd(
      internal::CheckedAdd(last_row, last_column, "输入矩阵最大 offset"),
      item_size, "输入矩阵末端 offset");
  internal::CheckedSize(last_offset, "输入矩阵最大 offset");

  const std::uint64_t value_count = internal::CheckedMultiply(
      view.rows, static_cast<std::uint64_t>(view.features), "输入矩阵元素数量");
  internal::CheckedSize(value_count, "分箱元素数量");
}

float ReadFiniteOrMissingFloat(const DenseMatrixView& view,
                               std::uint64_t row,
                               std::uint32_t feature) {
  const std::uint64_t offset =
      row * view.row_stride_bytes +
      static_cast<std::uint64_t>(feature) * view.column_stride_bytes;
  const auto* address = static_cast<const std::uint8_t*>(view.data) + offset;
  if (view.scalar_type == ScalarType::kFloat32) {
    float value = 0.0F;
    std::memcpy(&value, address, sizeof(value));
    if (std::isnan(value)) {
      return value;
    }
    if (!std::isfinite(value)) {
      throw DataError("输入矩阵包含 Inf");
    }
    return value;
  }
  double source = 0.0;
  std::memcpy(&source, address, sizeof(source));
  if (std::isnan(source)) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  if (!std::isfinite(source)) {
    throw DataError("输入矩阵包含 Inf");
  }
  constexpr double kFloatMax = static_cast<double>(std::numeric_limits<float>::max());
  if (source > kFloatMax || source < -kFloatMax) {
    throw DataError("float64 输入值超出有限 float32 表示范围");
  }
  return static_cast<float>(source);
}

std::uint64_t QuantileRank(std::uint64_t part,
                           std::uint64_t total,
                           std::uint64_t divisor) {
  const std::uint64_t quotient = total / divisor;
  const std::uint64_t remainder = total % divisor;
  const std::uint64_t whole = part * quotient;
  const std::uint64_t fraction_numerator = part * remainder;
  const std::uint64_t fraction =
      fraction_numerator / divisor + (fraction_numerator % divisor != 0 ? 1 : 0);
  return whole + fraction - 1;
}

std::vector<float> BuildBoundaries(std::vector<float> values,
                                   std::uint32_t max_bins) {
  values.erase(std::remove_if(values.begin(), values.end(),
                              [](float value) { return std::isnan(value); }),
               values.end());
  if (values.empty()) {
    return {};
  }
  std::sort(values.begin(), values.end());
  const std::uint64_t desired_bins =
      std::min<std::uint64_t>(max_bins, values.size());
  std::vector<float> boundaries;
  boundaries.reserve(internal::CheckedSize(desired_bins - 1, "边界数量"));
  const float maximum = values.back();
  for (std::uint64_t part = 1; part < desired_bins; ++part) {
    const std::uint64_t rank = QuantileRank(part, values.size(), desired_bins);
    const float candidate = values[internal::CheckedSize(rank, "分位 rank")];
    if (candidate < maximum &&
        (boundaries.empty() || candidate > boundaries.back())) {
      boundaries.push_back(candidate);
    }
  }
  return boundaries;
}

}  // namespace

void ValidateDenseView(const DenseMatrixView& view, std::uint32_t max_bins) {
  ValidateViewImpl(view, max_bins);
}

BinnedDataset QuantizeDense(const DenseMatrixView& view, std::uint32_t max_bins) {
  ValidateDenseView(view, max_bins);
  BinnedDataset result;
  result.rows_ = view.rows;
  result.schema_.features_ = view.features;
  result.schema_.max_bins_ = max_bins;
  result.schema_.storage_ = max_bins <= 256 ? BinStorage::kUInt8 : BinStorage::kUInt16;
  result.source_contiguous_ = view.source_contiguous;
  result.schema_.feature_metadata_.reserve(view.features);

  const std::uint64_t value_count_u64 = internal::CheckedMultiply(
      view.rows, static_cast<std::uint64_t>(view.features), "分箱元素数量");
  const std::size_t value_count = internal::CheckedSize(value_count_u64, "分箱元素数量");
  if (result.storage() == BinStorage::kUInt8) {
    result.bins_ = std::vector<std::uint8_t>(value_count);
  } else {
    result.bins_ = std::vector<std::uint16_t>(value_count);
  }
  result.missing_ = std::vector<std::uint8_t>(value_count, 0);

  std::vector<float> feature_values(internal::CheckedSize(view.rows, "单特征工作区"));
  for (std::uint32_t feature = 0; feature < view.features; ++feature) {
    std::uint64_t missing_count = 0;
    for (std::uint64_t row = 0; row < view.rows; ++row) {
      feature_values[internal::CheckedSize(row, "行索引")] =
          ReadFiniteOrMissingFloat(view, row, feature);
      if (std::isnan(feature_values[internal::CheckedSize(row, "行索引")])) {
        ++missing_count;
      }
    }
    const std::vector<float> feature_boundaries = BuildBoundaries(feature_values, max_bins);
    const FeatureBinMetadata metadata{result.schema_.boundaries_.size(),
                                      static_cast<std::uint32_t>(feature_boundaries.size()),
                                      static_cast<std::uint32_t>(feature_boundaries.size() + 1),
                                      missing_count};
    result.schema_.feature_metadata_.push_back(metadata);
    result.schema_.boundaries_.insert(result.schema_.boundaries_.end(),
                                      feature_boundaries.begin(),
                                      feature_boundaries.end());

    for (std::uint64_t row = 0; row < view.rows; ++row) {
      const float value = feature_values[internal::CheckedSize(row, "行索引")];
      const bool missing = std::isnan(value);
      const auto iterator =
          missing ? feature_boundaries.begin()
                  : std::lower_bound(feature_boundaries.begin(), feature_boundaries.end(), value);
      const std::uint32_t bin =
          static_cast<std::uint32_t>(iterator - feature_boundaries.begin());
      const std::size_t output_index = internal::CheckedSize(
          static_cast<std::uint64_t>(feature) * view.rows + row, "分箱输出索引");
      result.missing_[output_index] = missing ? 1U : 0U;
      if (result.storage() == BinStorage::kUInt8) {
        std::get<std::vector<std::uint8_t>>(result.bins_)[output_index] =
            static_cast<std::uint8_t>(bin);
      } else {
        std::get<std::vector<std::uint16_t>>(result.bins_)[output_index] =
            static_cast<std::uint16_t>(bin);
      }
    }
  }
  internal::ValidateSchemaFields(result.schema_);
  return result;
}

BinnedDataset TransformDense(const DenseMatrixView& view,
                             const QuantizationSchema& schema) {
  ValidateDenseView(view, schema.max_bins());
  internal::ValidateSchemaFields(schema);
  if (view.features != schema.features()) {
    throw DataError("预测输入特征数量与分箱 schema 不一致");
  }

  BinnedDataset result;
  result.rows_ = view.rows;
  result.source_contiguous_ = view.source_contiguous;
  result.schema_ = schema;
  const std::uint64_t value_count_u64 = internal::CheckedMultiply(
      view.rows, static_cast<std::uint64_t>(view.features), "预测分箱元素数量");
  const std::size_t value_count = internal::CheckedSize(value_count_u64,
                                                        "预测分箱元素数量");
  if (schema.storage() == BinStorage::kUInt8) {
    result.bins_ = std::vector<std::uint8_t>(value_count);
  } else {
    result.bins_ = std::vector<std::uint16_t>(value_count);
  }
  result.missing_ = std::vector<std::uint8_t>(value_count, 0);

  for (std::uint32_t feature = 0; feature < view.features; ++feature) {
    const FeatureBinMetadata& metadata = schema.feature_metadata()[feature];
    const auto first = schema.boundaries().begin() +
                       static_cast<std::ptrdiff_t>(metadata.boundary_offset);
    const auto last = first + static_cast<std::ptrdiff_t>(metadata.boundary_count);
    for (std::uint64_t row = 0; row < view.rows; ++row) {
      const float value = ReadFiniteOrMissingFloat(view, row, feature);
      const bool missing = std::isnan(value);
      const std::uint32_t bin =
          missing ? 0U
                  : static_cast<std::uint32_t>(std::lower_bound(first, last, value) - first);
      const std::size_t output_index = internal::CheckedSize(
          static_cast<std::uint64_t>(feature) * view.rows + row,
          "预测分箱输出索引");
      result.missing_[output_index] = missing ? 1U : 0U;
      if (schema.storage() == BinStorage::kUInt8) {
        std::get<std::vector<std::uint8_t>>(result.bins_)[output_index] =
            static_cast<std::uint8_t>(bin);
      } else {
        std::get<std::vector<std::uint16_t>>(result.bins_)[output_index] =
            static_cast<std::uint16_t>(bin);
      }
    }
  }
  return result;
}

QuantizationSchema RestoreQuantizationSchema(
    std::uint32_t features,
    std::uint32_t max_bins,
    std::vector<float> boundaries,
    std::vector<FeatureBinMetadata> metadata) {
  QuantizationSchema schema;
  schema.features_ = features;
  schema.max_bins_ = max_bins;
  schema.storage_ = max_bins <= 256 ? BinStorage::kUInt8 : BinStorage::kUInt16;
  schema.boundaries_ = std::move(boundaries);
  schema.feature_metadata_ = std::move(metadata);
  internal::ValidateSchemaFields(schema);
  return schema;
}

std::uint32_t BinnedDataset::GetBin(std::uint64_t row,
                                    std::uint32_t feature) const {
  if (row >= rows_ || feature >= features()) {
    throw DataError("分箱索引越界");
  }
  const std::size_t index = internal::CheckedSize(
      static_cast<std::uint64_t>(feature) * rows_ + row, "分箱读取索引");
  if (storage() == BinStorage::kUInt8) {
    return std::get<std::vector<std::uint8_t>>(bins_)[index];
  }
  return std::get<std::vector<std::uint16_t>>(bins_)[index];
}

bool BinnedDataset::IsMissing(std::uint64_t row, std::uint32_t feature) const {
  if (row >= rows_ || feature >= features()) {
    throw DataError("missing mask index out of range");
  }
  const std::size_t index = internal::CheckedSize(
      static_cast<std::uint64_t>(feature) * rows_ + row, "missing mask index");
  return !missing_.empty() && missing_[index] != 0;
}

const void* BinnedDataset::bin_data() const noexcept {
  if (storage() == BinStorage::kUInt8) {
    return std::get<std::vector<std::uint8_t>>(bins_).data();
  }
  return std::get<std::vector<std::uint16_t>>(bins_).data();
}

std::uint64_t BinnedDataset::bin_value_count() const noexcept {
  return rows_ * static_cast<std::uint64_t>(features());
}

}  // namespace mpsboost
