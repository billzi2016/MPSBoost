// MPSBoost 量化数据领域模型。
//
// 职责：定义设备无关的二维数值视图、确定性分箱结果、紧凑存储和稳定序列化接口。
// 本文件不依赖 Python、Metal 或文件系统；CPU oracle 与 MPS 后端必须共享同一结果。
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <variant>
#include <vector>

namespace mpsboost {

// 输入或序列化数据违反领域契约时抛出的统一异常。
class DataError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

enum class ScalarType : std::uint8_t { kFloat32 = 1, kFloat64 = 2 };
enum class BinStorage : std::uint8_t { kUInt8 = 1, kUInt16 = 2 };

// 借用外部二维 buffer 的只读视图。量化函数只在同步调用期间读取指针，不保存它。
struct DenseMatrixView final {
  const void* data{nullptr};
  std::uint64_t rows{0};
  std::uint32_t features{0};
  std::uint64_t row_stride_bytes{0};
  std::uint64_t column_stride_bytes{0};
  ScalarType scalar_type{ScalarType::kFloat32};
  bool source_contiguous{false};
};

// 单个特征在全局边界数组中的区间以及实际 bin 数。
struct FeatureBinMetadata final {
  std::uint64_t boundary_offset{0};
  std::uint32_t boundary_count{0};
  std::uint32_t bin_count{0};
};

class BinnedDataset final {
 public:
  std::uint64_t rows() const noexcept { return rows_; }
  std::uint32_t features() const noexcept { return features_; }
  std::uint32_t max_bins() const noexcept { return max_bins_; }
  BinStorage storage() const noexcept { return storage_; }
  bool source_contiguous() const noexcept { return source_contiguous_; }

  // 量化过程直接读取 strided view，不创建完整规范化输入副本；该值用于诊断和测试。
  bool source_was_copied() const noexcept { return false; }

  const std::vector<float>& boundaries() const noexcept { return boundaries_; }
  const std::vector<FeatureBinMetadata>& feature_metadata() const noexcept {
    return feature_metadata_;
  }

  // 读取 feature-major 紧凑存储中的一个 bin。索引错误明确失败，不能产生越界读取。
  std::uint32_t GetBin(std::uint64_t row, std::uint32_t feature) const;

  // 生成确定性的版本化字节表示，用于 round-trip 测试和后续缓存层复用。
  std::vector<std::uint8_t> Serialize() const;
  static BinnedDataset Deserialize(const std::vector<std::uint8_t>& bytes);

 private:
  friend BinnedDataset QuantizeDense(const DenseMatrixView&, std::uint32_t);

  std::uint64_t rows_{0};
  std::uint32_t features_{0};
  std::uint32_t max_bins_{0};
  BinStorage storage_{BinStorage::kUInt8};
  bool source_contiguous_{false};
  std::vector<float> boundaries_;
  std::vector<FeatureBinMetadata> feature_metadata_;
  std::variant<std::vector<std::uint8_t>, std::vector<std::uint16_t>> bins_;
};

// 把有限 float32/float64 矩阵量化为拥有自身内存的 feature-major 数据集。
BinnedDataset QuantizeDense(const DenseMatrixView& view, std::uint32_t max_bins);

// 只验证 view 元数据和可访问范围，不读取输入或分配分箱内存。量化入口与边界测试共享
// 此唯一验证逻辑，避免测试通过另一套简化规则产生虚假覆盖。
void ValidateDenseView(const DenseMatrixView& view, std::uint32_t max_bins);


}  // namespace mpsboost
