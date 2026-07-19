// MPSBoost 计算后端的最小稳定契约。
//
// 本文件只声明设备无关 POD 数据和后端入口，不暴露 Objective-C/Metal 类型。训练核心
// 将依赖此类抽象而不是具体设备对象，以满足依赖倒置并保持 CPU oracle 可独立实现。
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

// 后端执行失败的统一异常。设备层必须保留具体阶段信息，但不能让 Python 用户只看到
// 无上下文的原生错误码。
class BackendError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// 可安全返回给用户的设备能力摘要。这里故意不包含路径、用户名或可用于遥测的持久
// 标识，只提供判断能否运行和估算内存所需的信息。
struct BackendInfo final {
  bool available{false};
  std::string device_name;
  std::uint64_t recommended_max_working_set_size{0};
  bool has_unified_memory{false};
};

// 单个特征 bin 的确定性二阶统计。count 使用无符号 64 位计数，G/H 使用 FP64，
// 以便 CPU oracle 为后续设备 kernel 提供不依赖累计顺序的高精度对照。
struct HistogramBin final {
  std::uint64_t count{0};
  double gradient_sum{0.0};
  double hessian_sum{0.0};
};

using FeatureHistogram = std::vector<HistogramBin>;
using NodeHistograms = std::vector<FeatureHistogram>;

// 目标函数设备执行的最小接口。训练状态机只请求数学结果，不知道 gradient 是由 CPU
// oracle 还是 Metal kernel 产生，从而让设备选择保持在应用装配层。
class GradientComputer {
 public:
  virtual ~GradientComputer() = default;
  virtual std::vector<GradientPair> ComputeSquaredError(
      const std::vector<double>& labels,
      const std::vector<double>& predictions) const = 0;
};

// 训练核心当前只依赖 histogram 能力，避免为了未来方法建立臃肿后端接口。S4 的
// MPS 实现和本 CPU oracle 必须实现同一最小契约，后续能力通过接口隔离继续扩展。
class HistogramBuilder {
 public:
  virtual ~HistogramBuilder() = default;

  // 为指定节点的行集合构建全部特征 histogram。实现不得改变 rows 顺序或保存任何
  // 借用指针；返回形状必须与数据集特征及各自 bin_count 完全一致。
  virtual NodeHistograms BuildHistograms(
      const BinnedDataset& dataset,
      const std::vector<std::uint64_t>& rows,
      const std::vector<GradientPair>& gradients) const = 0;
};

// 唯一 CPU histogram oracle。它只实现计算能力，不包含树生长、split 选择或参数
// 语义，从而避免 CPU 与 MPS 各维护一套训练控制流。
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

// 最近一次同步设备工作的非敏感耗时。它只用于诊断和 benchmark，不参与训练结果、
// cache key 或调度决策，避免测量逻辑反向影响正确性。
struct BackendTiming final {
  double gradient_seconds{0.0};
  double histogram_encode_seconds{0.0};
  double histogram_command_seconds{0.0};
  double hot_path_encode_seconds{0.0};
  double hot_path_command_seconds{0.0};
  std::uint64_t pooled_buffer_reuse_count{0};
  std::uint64_t pooled_buffer_allocation_count{0};
};

// 单个 feature 的 GPU split scan 候选。该结构只承载设备侧扫描结果；训练核心仍会
// 按唯一 FP64 规则完成最终 split 验证，避免并行累计顺序改变模型语义。
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

// 可选的按层 histogram 能力。训练核心通过 dynamic_cast 探测该窄接口；不支持的后端
// 自动使用既有逐节点 HistogramBuilder，因此不会产生第二套训练语义。
class LayerHistogramBuilder {
 public:
  virtual ~LayerHistogramBuilder() = default;

  // 为同一层多个活跃节点构建 histogram。每个输入节点独立返回 NodeHistograms，顺序
  // 必须与 node_rows 一致；实现可以把节点批量编码到 GPU，但不得改变行集合内容。
  virtual std::vector<NodeHistograms> BuildLayerHistograms(
      const BinnedDataset& dataset,
      const std::vector<std::vector<std::uint64_t>>& node_rows,
      const std::vector<GradientPair>& gradients) const = 0;
};

// 真实 MPS 计算后端。Objective-C/Metal 对象全部隐藏在 Impl 中，稳定 C++ 头文件不
// 暴露平台类型；对象不可复制，但可在单个训练会话中重复使用 pipeline 和 command queue。
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

  // 复用同一 context/pipeline/command 实现验证 wheel 的最小 GPU 链路。该方法只由
  // 内部测试绑定调用，不属于 Python 公共数值 API。
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

// 查询默认 Metal 设备。无设备属于正常的“不可用”状态，不抛异常；运行时初始化失败
// 才抛 BackendError，从而让 is_available() 与真正故障保持不同语义。
BackendInfo QueryBackendInfo();

// 在真实 GPU 上执行逐元素加法，用于验证 shader、pipeline、buffer、command 和同步整条
// 链路。该入口仅用于后端集成测试，不是公共数值 API。
std::vector<float> RunVectorAdd(const std::vector<float>& left,
                                const std::vector<float>& right,
                                const std::string& metallib_path);

}  // namespace mpsboost
