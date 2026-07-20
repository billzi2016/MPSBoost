// MPSBoost 多轮回归训练与稳定模型契约。
//
// 职责：定义设备无关 boosting 状态机、可预测模型和模型 I/O 入口。训练核心只依赖
// GradientComputer/HistogramBuilder，不依赖 Python、Metal 对象、缓存或文件系统实现细节。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mpsboost/binned_dataset.hpp"
#include "mpsboost/tree.hpp"

namespace mpsboost {

class GradientComputer;
class HistogramBuilder;

struct TrainingParameters final {
  enum class Objective : std::uint32_t {
    kSquaredError = 0,
    kBinaryLogistic = 1,
  };

  std::uint32_t n_estimators{100};
  double learning_rate{0.1};
  std::uint32_t max_bins{256};
  Objective objective{Objective::kSquaredError};
  TreeTrainingParameters tree;
};

class RegressionModel final {
 public:
  std::uint32_t feature_count() const noexcept { return schema_.features(); }
  std::uint32_t tree_count() const noexcept {
    return static_cast<std::uint32_t>(trees_.size());
  }
  double base_score() const noexcept { return base_score_; }
  double learning_rate() const noexcept { return learning_rate_; }
  TrainingParameters::Objective objective() const noexcept { return objective_; }
  const QuantizationSchema& schema() const noexcept { return schema_; }
  const std::vector<RegressionTree>& trees() const noexcept { return trees_; }

  // 预测已使用本模型 schema 转换的数据。函数不修改模型或输入，可并发只读调用；
  // schema 不一致会明确失败，避免误用重新拟合边界的数据集。
  std::vector<double> Predict(const BinnedDataset& dataset) const;

  // 模型 loader 的唯一构造入口。所有字段在接受前完整验证，失败不产生半成品模型。
  static RegressionModel Restore(QuantizationSchema schema,
                                 double base_score,
                                 double learning_rate,
                                 TrainingParameters::Objective objective,
                                 std::vector<RegressionTree> trees);

 private:
  friend RegressionModel TrainRegressionModel(
      const BinnedDataset&,
      const std::vector<double>&,
      const std::vector<double>&,
      const TrainingParameters&,
      const GradientComputer&,
      const HistogramBuilder&);

  QuantizationSchema schema_;
  double base_score_{0.0};
  double learning_rate_{0.1};
  TrainingParameters::Objective objective_{TrainingParameters::Objective::kSquaredError};
  std::vector<RegressionTree> trees_;
};

class MulticlassModel final {
 public:
  std::uint32_t feature_count() const noexcept { return schema_.features(); }
  std::uint32_t class_count() const noexcept { return class_count_; }
  std::uint32_t tree_count() const noexcept {
    return static_cast<std::uint32_t>(trees_.size());
  }
  double learning_rate() const noexcept { return learning_rate_; }
  const QuantizationSchema& schema() const noexcept { return schema_; }
  const std::vector<RegressionTree>& trees() const noexcept { return trees_; }
  const std::vector<double>& base_scores() const noexcept { return base_scores_; }
  const std::vector<double>& class_labels() const noexcept { return class_labels_; }

  // Return row-major raw class margins with shape rows × class_count.
  std::vector<double> PredictMargins(const BinnedDataset& dataset) const;

  // Attach the user-visible numeric class mapping owned by the model file.
  // Labels must be finite, strictly increasing, and match class_count.
  void SetClassLabels(std::vector<double> class_labels);

  // Restore a validated native softmax model from versioned model bytes. Trees
  // must be stored in round-major class order.
  static MulticlassModel Restore(QuantizationSchema schema,
                                 std::uint32_t class_count,
                                 double learning_rate,
                                 std::vector<double> base_scores,
                                 std::vector<double> class_labels,
                                 std::vector<RegressionTree> trees);

 private:
  friend MulticlassModel TrainMulticlassSoftmaxModel(
      const BinnedDataset&,
      const std::vector<double>&,
      const std::vector<double>&,
      const TrainingParameters&,
      std::uint32_t,
      const HistogramBuilder&);

  QuantizationSchema schema_;
  std::uint32_t class_count_{0};
  double learning_rate_{0.1};
  std::vector<double> base_scores_;
  std::vector<double> class_labels_;
  std::vector<RegressionTree> trees_;
};

RegressionModel TrainRegressionModel(
    const BinnedDataset& dataset,
    const std::vector<double>& labels,
    const std::vector<double>& sample_weights,
    const TrainingParameters& parameters,
    const GradientComputer& gradient_computer,
    const HistogramBuilder& histogram_builder);

MulticlassModel TrainMulticlassSoftmaxModel(
    const BinnedDataset& dataset,
    const std::vector<double>& labels,
    const std::vector<double>& sample_weights,
    const TrainingParameters& parameters,
    std::uint32_t class_count,
    const HistogramBuilder& histogram_builder);

// 模型格式入口使用版本化二进制容器、完整性校验和同目录原子替换。文件中不包含训练
// 数据、路径、设备标识或缓存；加载失败不影响调用方已有模型。
std::vector<std::uint8_t> SerializeModel(const RegressionModel& model);
RegressionModel DeserializeModel(const std::vector<std::uint8_t>& bytes);
std::vector<std::uint8_t> SerializeModel(const MulticlassModel& model);
MulticlassModel DeserializeMulticlassModel(const std::vector<std::uint8_t>& bytes);
void SaveModelFile(const RegressionModel& model, const std::string& path);
RegressionModel LoadModelFile(const std::string& path);
void SaveModelFile(const MulticlassModel& model, const std::string& path);
MulticlassModel LoadMulticlassModelFile(const std::string& path);

}  // namespace mpsboost
