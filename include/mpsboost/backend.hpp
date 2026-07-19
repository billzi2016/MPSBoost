// MPSBoost 计算后端的最小稳定契约。
//
// 本文件只声明设备无关 POD 数据和后端入口，不暴露 Objective-C/Metal 类型。训练核心
// 将依赖此类抽象而不是具体设备对象，以满足依赖倒置并保持 CPU oracle 可独立实现。
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

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

// 查询默认 Metal 设备。无设备属于正常的“不可用”状态，不抛异常；运行时初始化失败
// 才抛 BackendError，从而让 is_available() 与真正故障保持不同语义。
BackendInfo QueryBackendInfo();

// 在真实 GPU 上执行逐元素加法，用于验证 shader、pipeline、buffer、command 和同步整条
// 链路。该入口仅用于后端集成测试，不是公共数值 API。
std::vector<float> RunVectorAdd(const std::vector<float>& left,
                                const std::vector<float>& right,
                                const std::string& metallib_path);

}  // namespace mpsboost
