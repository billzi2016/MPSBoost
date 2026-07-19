// Internal validation and overflow helpers for binned dataset code.
//
// Quantization and serialization share these helpers so safety checks stay DRY while their public
// implementation files remain small and focused.

#include "binned_dataset_internal.hpp"

#include <cmath>
#include <limits>
#include <string>

namespace mpsboost {
namespace internal {

std::uint64_t CheckedMultiply(std::uint64_t left,
                              std::uint64_t right,
                              const char* context) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw DataError(std::string(context) + "：乘法溢出");
  }
  return left * right;
}

std::uint64_t CheckedAdd(std::uint64_t left,
                         std::uint64_t right,
                         const char* context) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    throw DataError(std::string(context) + "：加法溢出");
  }
  return left + right;
}

std::size_t CheckedSize(std::uint64_t value, const char* context) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw DataError(std::string(context) + "：超出当前进程可寻址范围");
  }
  return static_cast<std::size_t>(value);
}

std::uint64_t ScalarByteSize(ScalarType type) {
  switch (type) {
    case ScalarType::kFloat32:
      return sizeof(float);
    case ScalarType::kFloat64:
      return sizeof(double);
  }
  throw DataError("不支持的输入标量类型");
}

void ValidateSchemaFields(const QuantizationSchema& schema) {
  if (schema.features() == 0 || schema.max_bins() < 2 ||
      schema.max_bins() > 65536) {
    throw DataError("分箱 schema 的特征数或 max_bins 不合法");
  }
  if (schema.feature_metadata().size() != schema.features()) {
    throw DataError("分箱 schema 的特征元数据数量不一致");
  }
  std::uint64_t expected_offset = 0;
  for (const FeatureBinMetadata& item : schema.feature_metadata()) {
    if (item.boundary_offset != expected_offset || item.bin_count == 0 ||
        item.bin_count != item.boundary_count + 1 ||
        item.bin_count > schema.max_bins()) {
      throw DataError("分箱 schema 的边界区间或 bin 数不合法");
    }
    const std::uint64_t end = CheckedAdd(
        item.boundary_offset, item.boundary_count, "分箱 schema 边界区间");
    if (end > schema.boundaries().size()) {
      throw DataError("分箱 schema 的边界区间越界");
    }
    for (std::uint32_t index = 0; index < item.boundary_count; ++index) {
      const float value = schema.boundaries()[item.boundary_offset + index];
      if (!std::isfinite(value) ||
          (index != 0 &&
           value <= schema.boundaries()[item.boundary_offset + index - 1])) {
        throw DataError("分箱 schema 的特征边界必须有限且严格递增");
      }
    }
    expected_offset = end;
  }
  if (expected_offset != schema.boundaries().size()) {
    throw DataError("分箱 schema 包含未被特征引用的边界");
  }
  const BinStorage expected_storage =
      schema.max_bins() <= 256 ? BinStorage::kUInt8 : BinStorage::kUInt16;
  if (schema.storage() != expected_storage) {
    throw DataError("分箱 schema 的存储宽度与 max_bins 不一致");
  }
}

}  // namespace internal

}  // namespace mpsboost
