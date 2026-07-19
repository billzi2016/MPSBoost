// MPSBoost 最小 Metal 运行时实现。
//
// 职责：提供无副作用设备能力查询，并把历史 smoke 入口装配到唯一 MpsBackend runtime。
// pipeline、buffer 与 command 的正式实现位于 mps_backend.mm，本文件不得复制设备逻辑。

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "mpsboost/backend.hpp"

namespace mpsboost {

BackendInfo QueryBackendInfo() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      return BackendInfo{};
    }

    BackendInfo info;
    info.available = true;
    info.device_name = [[device name] UTF8String];
    info.recommended_max_working_set_size =
        static_cast<std::uint64_t>([device recommendedMaxWorkingSetSize]);
    info.has_unified_memory = [device hasUnifiedMemory];
    return info;
  }
}

std::vector<float> RunVectorAdd(const std::vector<float>& left,
                                const std::vector<float>& right,
                                const std::string& metallib_path) {
  MpsBackend backend(metallib_path);
  return backend.RunVectorAddForTest(left, right);
}

}  // namespace mpsboost
