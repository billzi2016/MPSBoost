// MPSBoost 确定性分箱实现。
//
// 本文件是分箱语义的唯一权威实现：输入验证、分位 rank、重复边界处理、lower_bound
// 映射和二进制格式都集中在此。其他后端只能消费结果，不得复制或改变这些规则。

#include "mpsboost/binned_dataset.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>

namespace mpsboost {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{'M', 'P', 'S', 'B', 'I', 'N', 0, 1};

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

  const std::uint64_t item_size = ScalarByteSize(view.scalar_type);
  if (view.row_stride_bytes < item_size || view.column_stride_bytes < item_size) {
    throw DataError("输入矩阵 stride 小于元素字节数");
  }

  // 只检查视图可能访问的最大 offset；后续逐元素读取即可避免在热循环重复检查乘法。
  const std::uint64_t last_row = CheckedMultiply(
      view.rows - 1, view.row_stride_bytes, "输入矩阵 row stride");
  const std::uint64_t last_column = CheckedMultiply(
      static_cast<std::uint64_t>(view.features - 1), view.column_stride_bytes,
      "输入矩阵 column stride");
  const std::uint64_t last_offset = CheckedAdd(
      CheckedAdd(last_row, last_column, "输入矩阵最大 offset"), item_size,
      "输入矩阵末端 offset");
  CheckedSize(last_offset, "输入矩阵最大 offset");

  const std::uint64_t value_count = CheckedMultiply(
      view.rows, static_cast<std::uint64_t>(view.features), "输入矩阵元素数量");
  CheckedSize(value_count, "分箱元素数量");
}

float ReadFiniteFloat(const DenseMatrixView& view,
                      std::uint64_t row,
                      std::uint32_t feature) {
  const std::uint64_t offset =
      row * view.row_stride_bytes +
      static_cast<std::uint64_t>(feature) * view.column_stride_bytes;
  const auto* address = static_cast<const std::uint8_t*>(view.data) + offset;

  if (view.scalar_type == ScalarType::kFloat32) {
    float value = 0.0F;
    std::memcpy(&value, address, sizeof(value));
    if (!std::isfinite(value)) {
      throw DataError("输入矩阵包含 NaN 或 Inf");
    }
    return value;
  }

  double source = 0.0;
  std::memcpy(&source, address, sizeof(source));
  if (!std::isfinite(source)) {
    throw DataError("输入矩阵包含 NaN 或 Inf");
  }
  constexpr double kFloatMax = static_cast<double>(std::numeric_limits<float>::max());
  if (source > kFloatMax || source < -kFloatMax) {
    throw DataError("float64 输入值超出有限 float32 表示范围");
  }
  return static_cast<float>(source);
}

// 精确计算 ceil(part * total / divisor) - 1，避免 part*total 在大行数时溢出。
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
  std::sort(values.begin(), values.end());
  const std::uint64_t desired_bins =
      std::min<std::uint64_t>(max_bins, values.size());
  std::vector<float> boundaries;
  boundaries.reserve(CheckedSize(desired_bins - 1, "边界数量"));

  const float maximum = values.back();
  for (std::uint64_t part = 1; part < desired_bins; ++part) {
    const std::uint64_t rank = QuantileRank(part, values.size(), desired_bins);
    const float candidate = values[CheckedSize(rank, "分位 rank")];
    // 最大值不能成为边界，否则会产生空的最右 bin；重复候选只保留一次。
    if (candidate < maximum &&
        (boundaries.empty() || candidate > boundaries.back())) {
      boundaries.push_back(candidate);
    }
  }
  return boundaries;
}

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
      throw DataError(std::string("分箱序列化字段不是有限值：") + field);
    }
    return value;
  }

  void ExpectMagic() {
    Require(kMagic.size(), "magic");
    for (const std::uint8_t expected : kMagic) {
      if (bytes_[position_++] != expected) {
        throw DataError("分箱序列化 magic 或版本不受支持");
      }
    }
  }

  bool at_end() const noexcept { return position_ == bytes_.size(); }

 private:
  void Require(std::size_t count, const char* field) {
    if (count > bytes_.size() - position_) {
      throw DataError(std::string("分箱序列化数据截断：") + field);
    }
  }

  const std::vector<std::uint8_t>& bytes_;
  std::size_t position_{0};
};

}  // namespace

void ValidateDenseView(const DenseMatrixView& view, std::uint32_t max_bins) {
  ValidateViewImpl(view, max_bins);
}

BinnedDataset QuantizeDense(const DenseMatrixView& view, std::uint32_t max_bins) {
  ValidateDenseView(view, max_bins);

  BinnedDataset result;
  result.rows_ = view.rows;
  result.features_ = view.features;
  result.max_bins_ = max_bins;
  result.storage_ = max_bins <= 256 ? BinStorage::kUInt8 : BinStorage::kUInt16;
  result.source_contiguous_ = view.source_contiguous;
  result.feature_metadata_.reserve(view.features);

  const std::uint64_t value_count_u64 = CheckedMultiply(
      view.rows, static_cast<std::uint64_t>(view.features), "分箱元素数量");
  const std::size_t value_count = CheckedSize(value_count_u64, "分箱元素数量");
  if (result.storage_ == BinStorage::kUInt8) {
    result.bins_ = std::vector<std::uint8_t>(value_count);
  } else {
    result.bins_ = std::vector<std::uint16_t>(value_count);
  }

  std::vector<float> feature_values(CheckedSize(view.rows, "单特征工作区"));
  for (std::uint32_t feature = 0; feature < view.features; ++feature) {
    for (std::uint64_t row = 0; row < view.rows; ++row) {
      feature_values[CheckedSize(row, "行索引")] = ReadFiniteFloat(view, row, feature);
    }

    const std::vector<float> feature_boundaries =
        BuildBoundaries(feature_values, max_bins);
    const FeatureBinMetadata metadata{
        result.boundaries_.size(),
        static_cast<std::uint32_t>(feature_boundaries.size()),
        static_cast<std::uint32_t>(feature_boundaries.size() + 1),
    };
    result.feature_metadata_.push_back(metadata);
    result.boundaries_.insert(result.boundaries_.end(), feature_boundaries.begin(),
                              feature_boundaries.end());

    for (std::uint64_t row = 0; row < view.rows; ++row) {
      const float value = feature_values[CheckedSize(row, "行索引")];
      const auto iterator =
          std::lower_bound(feature_boundaries.begin(), feature_boundaries.end(), value);
      const std::uint32_t bin =
          static_cast<std::uint32_t>(iterator - feature_boundaries.begin());
      const std::size_t output_index = CheckedSize(
          static_cast<std::uint64_t>(feature) * view.rows + row, "分箱输出索引");
      if (result.storage_ == BinStorage::kUInt8) {
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

std::uint32_t BinnedDataset::GetBin(std::uint64_t row, std::uint32_t feature) const {
  if (row >= rows_ || feature >= features_) {
    throw DataError("分箱索引越界");
  }
  const std::size_t index = CheckedSize(
      static_cast<std::uint64_t>(feature) * rows_ + row, "分箱读取索引");
  if (storage_ == BinStorage::kUInt8) {
    return std::get<std::vector<std::uint8_t>>(bins_)[index];
  }
  return std::get<std::vector<std::uint16_t>>(bins_)[index];
}

std::vector<std::uint8_t> BinnedDataset::Serialize() const {
  std::vector<std::uint8_t> output(kMagic.begin(), kMagic.end());
  AppendLittleEndian(&output, rows_);
  AppendLittleEndian(&output, features_);
  AppendLittleEndian(&output, max_bins_);
  AppendLittleEndian(&output, static_cast<std::uint8_t>(storage_));
  AppendLittleEndian(&output, static_cast<std::uint8_t>(source_contiguous_ ? 1 : 0));
  AppendLittleEndian(&output, static_cast<std::uint16_t>(0));
  AppendLittleEndian(&output, static_cast<std::uint64_t>(boundaries_.size()));
  AppendLittleEndian(&output, rows_ * static_cast<std::uint64_t>(features_));

  for (const FeatureBinMetadata& metadata : feature_metadata_) {
    AppendLittleEndian(&output, metadata.boundary_offset);
    AppendLittleEndian(&output, metadata.boundary_count);
    AppendLittleEndian(&output, metadata.bin_count);
  }
  for (const float boundary : boundaries_) {
    AppendFloat(&output, boundary);
  }
  if (storage_ == BinStorage::kUInt8) {
    const auto& values = std::get<std::vector<std::uint8_t>>(bins_);
    output.insert(output.end(), values.begin(), values.end());
  } else {
    for (const std::uint16_t value : std::get<std::vector<std::uint16_t>>(bins_)) {
      AppendLittleEndian(&output, value);
    }
  }
  return output;
}

BinnedDataset BinnedDataset::Deserialize(const std::vector<std::uint8_t>& bytes) {
  ByteReader reader(bytes);
  reader.ExpectMagic();

  BinnedDataset result;
  result.rows_ = reader.ReadUnsigned<std::uint64_t>("rows");
  result.features_ = reader.ReadUnsigned<std::uint32_t>("features");
  result.max_bins_ = reader.ReadUnsigned<std::uint32_t>("max_bins");
  const auto storage_value = reader.ReadUnsigned<std::uint8_t>("storage");
  result.source_contiguous_ =
      reader.ReadUnsigned<std::uint8_t>("source_contiguous") != 0;
  static_cast<void>(reader.ReadUnsigned<std::uint16_t>("reserved"));
  const std::uint64_t boundary_count =
      reader.ReadUnsigned<std::uint64_t>("boundary_count");
  const std::uint64_t value_count = reader.ReadUnsigned<std::uint64_t>("value_count");

  if (result.rows_ == 0 || result.features_ == 0 || result.max_bins_ < 2 ||
      result.max_bins_ > 65536) {
    throw DataError("分箱序列化头字段不合法");
  }
  const std::uint64_t expected_values = CheckedMultiply(
      result.rows_, static_cast<std::uint64_t>(result.features_), "序列化元素数量");
  if (value_count != expected_values) {
    throw DataError("分箱序列化元素数量不一致");
  }
  result.storage_ = storage_value == static_cast<std::uint8_t>(BinStorage::kUInt8)
                        ? BinStorage::kUInt8
                        : storage_value == static_cast<std::uint8_t>(BinStorage::kUInt16)
                              ? BinStorage::kUInt16
                              : throw DataError("分箱序列化存储类型不支持");
  const BinStorage expected_storage =
      result.max_bins_ <= 256 ? BinStorage::kUInt8 : BinStorage::kUInt16;
  if (result.storage_ != expected_storage) {
    throw DataError("分箱序列化存储类型与 max_bins 不一致");
  }

  result.feature_metadata_.reserve(result.features_);
  for (std::uint32_t feature = 0; feature < result.features_; ++feature) {
    FeatureBinMetadata metadata;
    metadata.boundary_offset = reader.ReadUnsigned<std::uint64_t>("boundary_offset");
    metadata.boundary_count = reader.ReadUnsigned<std::uint32_t>("feature boundary_count");
    metadata.bin_count = reader.ReadUnsigned<std::uint32_t>("feature bin_count");
    const std::uint64_t end = CheckedAdd(metadata.boundary_offset,
                                         metadata.boundary_count,
                                         "特征边界区间");
    if (end > boundary_count || metadata.bin_count != metadata.boundary_count + 1 ||
        metadata.bin_count > result.max_bins_) {
      throw DataError("分箱序列化特征元数据不合法");
    }
    result.feature_metadata_.push_back(metadata);
  }

  result.boundaries_.reserve(CheckedSize(boundary_count, "边界数量"));
  for (std::uint64_t index = 0; index < boundary_count; ++index) {
    result.boundaries_.push_back(reader.ReadFloat("boundary"));
  }

  if (result.storage_ == BinStorage::kUInt8) {
    std::vector<std::uint8_t> values;
    values.reserve(CheckedSize(value_count, "uint8 bin 数量"));
    for (std::uint64_t index = 0; index < value_count; ++index) {
      values.push_back(reader.ReadUnsigned<std::uint8_t>("uint8 bin"));
    }
    result.bins_ = std::move(values);
  } else {
    std::vector<std::uint16_t> values;
    values.reserve(CheckedSize(value_count, "uint16 bin 数量"));
    for (std::uint64_t index = 0; index < value_count; ++index) {
      values.push_back(reader.ReadUnsigned<std::uint16_t>("uint16 bin"));
    }
    result.bins_ = std::move(values);
  }

  if (!reader.at_end()) {
    throw DataError("分箱序列化数据包含未识别的尾部字节");
  }
  for (std::uint32_t feature = 0; feature < result.features_; ++feature) {
    const FeatureBinMetadata& metadata = result.feature_metadata_[feature];
    for (std::uint64_t row = 0; row < result.rows_; ++row) {
      if (result.GetBin(row, feature) >= metadata.bin_count) {
        throw DataError("分箱序列化 bin 值超出特征范围");
      }
    }
  }
  return result;
}

}  // namespace mpsboost
