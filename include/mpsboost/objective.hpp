// MPSBoost 目标函数与训练数学的唯一领域契约。
//
// 职责：定义平方误差 gradient/Hessian、节点分数、叶子权重和切分增益。CPU oracle、
// 训练核心与后续 Metal kernel 必须严格对照这里的语义；Python 层不得复制这些公式。
#pragma once

#include <stdexcept>
#include <vector>

namespace mpsboost {

// 训练输入、参数或中间统计违反领域不变量时抛出的统一异常。该异常表示算法无法产生
// 完整可信模型，调用方不得捕获后继续导出部分树。
class TrainingError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// 单个样本的二阶统计。CPU 使用 double 累计形成 oracle；设备实现即使采用 float，
// 也必须在测试中与此语义逐层对照。
struct GradientPair final {
  double gradient{0.0};
  double hessian{0.0};
};

// 计算平方误差的一阶和二阶统计：g = prediction - label，h = 1。
// 两个输入必须等长、非空且全部有限；返回值拥有独立内存，无外部生命周期依赖。
std::vector<GradientPair> ComputeSquaredErrorGradients(
    const std::vector<double>& labels,
    const std::vector<double>& predictions);

// 计算节点分数 G²/(H+lambda)。H 和 lambda 必须非负，分母必须严格为正。
double NodeScore(double gradient_sum, double hessian_sum, double reg_lambda);

// 计算叶值 -G/(H+lambda)。验证规则与 NodeScore 完全一致。
double LeafWeight(double gradient_sum, double hessian_sum, double reg_lambda);

// 计算 0.5*(score_left+score_right-score_parent)-gamma。左右统计必须代表非空、
// 正 Hessian 候选；gamma 必须非负。函数不决定是否切分，调用方只接受严格正增益。
double SplitGain(double left_gradient,
                 double left_hessian,
                 double right_gradient,
                 double right_hessian,
                 double reg_lambda,
                 double gamma);

}  // namespace mpsboost
