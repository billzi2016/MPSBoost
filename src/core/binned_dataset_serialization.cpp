// Binary serialization for MPSBoost binned datasets.
//
// This file owns the private test/debug binned-dataset container format. Quantization and dense
// transform logic remain in binned_dataset.cpp.

#include "mpsboost/binned_dataset.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <string>
#include <type_traits>

#include "binned_dataset_internal.hpp"

namespace mpsboost {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{'M', 'P', 'S', 'B', 'I', 'N', 0, 1};

template <typename Integer>
void AppendLittleEndian(std::vector<std::uint8_t>* output, Integer value) {
  static_assert(std::is_unsigned_v<Integer>);
  for (std::size_t index = 0; index < sizeof(Integer); ++index) {
    output->push_back(static_cast<std::uint8_t>((value >> (index * 8U)) & 0xFFU));
  }
}

void AppendFloat(std::vector<std::uint8_t>* output, float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  AppendLittleEndian(output, bits);
}

class ByteReader final {
 public:
  explicit ByteReader(const std::vector<std::uint8_t>& bytes) : bytes_(bytes) {}

  template <typename Integer>
  Integer ReadUnsigned(const char* field) {
    static_assert(std::is_unsigned_v<Integer>);
    Require(sizeof(Integer), field);
    Integer value = 0;
    for (std::size_t index = 0; index < sizeof(Integer); ++index) {
      value |= static_cast<Integer>(bytes_[position_++]) << (index * 8U);
    }
    return value;
  }

  float ReadFloat(const char* field) {
    const std::uint32_t bits = ReadUnsigned<std::uint32_t>(field);
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    if (!std::isfinite(value)) {
      throw DataError(std::string("Binned serialization field is not finite: ") + field);
    }
    return value;
  }

  void ExpectMagic() {
    Require(kMagic.size(), "magic");
    for (const std::uint8_t expected : kMagic) {
      if (bytes_[position_++] != expected) {
        throw DataError("Binned serialization magic or version is unsupported");
      }
    }
  }

  bool at_end() const noexcept { return position_ == bytes_.size(); }

 private:
  void Require(std::size_t count, const char* field) {
    if (count > bytes_.size() - position_) {
      throw DataError(std::string("Binned serialization data is truncated: ") + field);
    }
  }

  const std::vector<std::uint8_t>& bytes_;
  std::size_t position_{0};
};

}  // namespace

std::vector<std::uint8_t> BinnedDataset::Serialize() const {
  std::vector<std::uint8_t> output(kMagic.begin(), kMagic.end());
  AppendLittleEndian(&output, rows_);
  AppendLittleEndian(&output, features());
  AppendLittleEndian(&output, max_bins());
  AppendLittleEndian(&output, static_cast<std::uint8_t>(storage()));
  AppendLittleEndian(&output, static_cast<std::uint8_t>(source_contiguous_ ? 1 : 0));
  AppendLittleEndian(&output, static_cast<std::uint16_t>(0));
  AppendLittleEndian(&output, static_cast<std::uint64_t>(boundaries().size()));
  AppendLittleEndian(&output, rows_ * static_cast<std::uint64_t>(features()));

  for (const FeatureBinMetadata& metadata : feature_metadata()) {
    AppendLittleEndian(&output, metadata.boundary_offset);
    AppendLittleEndian(&output, metadata.boundary_count);
    AppendLittleEndian(&output, metadata.bin_count);
    AppendLittleEndian(&output, metadata.missing_count);
  }
  for (const float boundary : boundaries()) {
    AppendFloat(&output, boundary);
  }
  if (storage() == BinStorage::kUInt8) {
    const auto& values = std::get<std::vector<std::uint8_t>>(bins_);
    output.insert(output.end(), values.begin(), values.end());
  } else {
    for (const std::uint16_t value : std::get<std::vector<std::uint16_t>>(bins_)) {
      AppendLittleEndian(&output, value);
    }
  }
  output.insert(output.end(), missing_.begin(), missing_.end());
  return output;
}

BinnedDataset BinnedDataset::Deserialize(const std::vector<std::uint8_t>& bytes) {
  ByteReader reader(bytes);
  reader.ExpectMagic();
  BinnedDataset result;
  result.rows_ = reader.ReadUnsigned<std::uint64_t>("rows");
  result.schema_.features_ = reader.ReadUnsigned<std::uint32_t>("features");
  result.schema_.max_bins_ = reader.ReadUnsigned<std::uint32_t>("max_bins");
  const auto storage_value = reader.ReadUnsigned<std::uint8_t>("storage");
  result.source_contiguous_ =
      reader.ReadUnsigned<std::uint8_t>("source_contiguous") != 0;
  static_cast<void>(reader.ReadUnsigned<std::uint16_t>("reserved"));
  const std::uint64_t boundary_count =
      reader.ReadUnsigned<std::uint64_t>("boundary_count");
  const std::uint64_t value_count = reader.ReadUnsigned<std::uint64_t>("value_count");

  if (result.rows_ == 0 || result.features() == 0 || result.max_bins() < 2 ||
      result.max_bins() > 65536) {
    throw DataError("Binned serialization header fields are invalid");
  }
  const std::uint64_t expected_values = internal::CheckedMultiply(
      result.rows_, static_cast<std::uint64_t>(result.features()), "serialized element count");
  if (value_count != expected_values) {
    throw DataError("Binned serialization element count does not match");
  }
  result.schema_.storage_ =
      storage_value == static_cast<std::uint8_t>(BinStorage::kUInt8) ? BinStorage::kUInt8
      : storage_value == static_cast<std::uint8_t>(BinStorage::kUInt16) ? BinStorage::kUInt16
      : throw DataError("Binned serialization storage type is unsupported");
  const BinStorage expected_storage =
      result.max_bins() <= 256 ? BinStorage::kUInt8 : BinStorage::kUInt16;
  if (result.storage() != expected_storage) {
    throw DataError("Binned serialization storage type does not match max_bins");
  }

  result.schema_.feature_metadata_.reserve(result.features());
  for (std::uint32_t feature = 0; feature < result.features(); ++feature) {
    FeatureBinMetadata metadata;
    metadata.boundary_offset = reader.ReadUnsigned<std::uint64_t>("boundary_offset");
    metadata.boundary_count = reader.ReadUnsigned<std::uint32_t>("feature boundary_count");
    metadata.bin_count = reader.ReadUnsigned<std::uint32_t>("feature bin_count");
    metadata.missing_count = reader.ReadUnsigned<std::uint64_t>("feature missing_count");
    const std::uint64_t end = internal::CheckedAdd(metadata.boundary_offset,
                                                   metadata.boundary_count,
                                                   "feature boundary range");
    if (end > boundary_count || metadata.bin_count != metadata.boundary_count + 1 ||
        metadata.bin_count > result.max_bins()) {
      throw DataError("Binned serialization feature metadata is invalid");
    }
    result.schema_.feature_metadata_.push_back(metadata);
  }

  result.schema_.boundaries_.reserve(internal::CheckedSize(boundary_count, "boundary count"));
  for (std::uint64_t index = 0; index < boundary_count; ++index) {
    result.schema_.boundaries_.push_back(reader.ReadFloat("boundary"));
  }
  if (result.storage() == BinStorage::kUInt8) {
    std::vector<std::uint8_t> values;
    values.reserve(internal::CheckedSize(value_count, "uint8 bin count"));
    for (std::uint64_t index = 0; index < value_count; ++index) {
      values.push_back(reader.ReadUnsigned<std::uint8_t>("uint8 bin"));
    }
    result.bins_ = std::move(values);
  } else {
    std::vector<std::uint16_t> values;
    values.reserve(internal::CheckedSize(value_count, "uint16 bin count"));
    for (std::uint64_t index = 0; index < value_count; ++index) {
      values.push_back(reader.ReadUnsigned<std::uint16_t>("uint16 bin"));
    }
    result.bins_ = std::move(values);
  }
  result.missing_.reserve(internal::CheckedSize(value_count, "missing mask size"));
  for (std::uint64_t index = 0; index < value_count; ++index) {
    const std::uint8_t missing = reader.ReadUnsigned<std::uint8_t>("missing mask");
    if (missing > 1) {
      throw DataError("missing mask value must be 0 or 1");
    }
    result.missing_.push_back(missing);
  }

  if (!reader.at_end()) {
    throw DataError("Binned serialization data contains unrecognized trailing bytes");
  }
  internal::ValidateSchemaFields(result.schema_);
  for (std::uint32_t feature = 0; feature < result.features(); ++feature) {
    const FeatureBinMetadata& metadata = result.feature_metadata()[feature];
    std::uint64_t missing_count = 0;
    for (std::uint64_t row = 0; row < result.rows_; ++row) {
      if (result.GetBin(row, feature) >= metadata.bin_count) {
        throw DataError("Binned serialization bin value exceeds the feature range");
      }
      if (result.IsMissing(row, feature)) {
        ++missing_count;
      }
    }
    if (missing_count != metadata.missing_count) {
      throw DataError("serialized missing count does not match missing mask");
    }
  }
  return result;
}

}  // namespace mpsboost
