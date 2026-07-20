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

enum class ScalarType : std::uint8_t { kFloat32 = 1,
                                       kFloat64 = 2 };
enum class BinStorage : std::uint8_t { kUInt8 = 1,
                                       kUInt16 = 2 };

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
  std::uint64_t missing_count{0};
};

class BinnedDataset;

// 训练期冻结的分箱规则。Schema 只保存边界和布局，不保存任何训练样本，因此可以安全
// 进入模型文件并用于新数据预测；预测阶段绝不能重新拟合分位数边界。
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

  // 量化过程直接读取 strided view，不创建完整规范化输入副本；该值用于诊断和测试。
  bool source_was_copied() const noexcept { return false; }

  const std::vector<float>& boundaries() const noexcept {
    return schema_.boundaries();
  }
  const std::vector<FeatureBinMetadata>& feature_metadata() const noexcept {
    return schema_.feature_metadata();
  }
  const QuantizationSchema& schema() const noexcept { return schema_; }

  // 返回 feature-major 紧凑 bin buffer 的只读地址和元素数量。地址仅在本对象存活且
  // 未移动时有效，MPS 后端必须同步复制或完成 command 后再返回，不能保存借用指针。
  const void* bin_data() const noexcept;
  std::uint64_t bin_value_count() const noexcept;

  // 读取 feature-major 紧凑存储中的一个 bin。索引错误明确失败，不能产生越界读取。
  std::uint32_t GetBin(std::uint64_t row, std::uint32_t feature) const;
  bool IsMissing(std::uint64_t row, std::uint32_t feature) const;

  // 生成确定性的版本化字节表示，用于 round-trip 测试和后续缓存层复用。
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

// 把有限 float32/float64 矩阵量化为拥有自身内存的 feature-major 数据集。
BinnedDataset QuantizeDense(const DenseMatrixView& view, std::uint32_t max_bins);

// 使用训练期 schema 转换新矩阵。该入口只应用既有 lower_bound 边界，不读取标签、
// 不重新估计分位数，保证 fit、predict 和模型加载后的路由语义完全一致。
BinnedDataset TransformDense(const DenseMatrixView& view,
                             const QuantizationSchema& schema);

// 从已验证的模型字段恢复 schema。所有 offset、边界单调性和 bin 数在此唯一入口检查，
// 模型 loader 不得绕过验证直接构造领域对象。
QuantizationSchema RestoreQuantizationSchema(
    std::uint32_t features,
    std::uint32_t max_bins,
    std::vector<float> boundaries,
    std::vector<FeatureBinMetadata> metadata);

// 只验证 view 元数据和可访问范围，不读取输入或分配分箱内存。量化入口与边界测试共享
// 此唯一验证逻辑，避免测试通过另一套简化规则产生虚假覆盖。
void ValidateDenseView(const DenseMatrixView& view, std::uint32_t max_bins);

}  // namespace mpsboost
