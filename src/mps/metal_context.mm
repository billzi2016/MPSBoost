// MPSBoost minimal Metal runtime implementation.
//
// Responsibility: provides side-effect-free device capability queries and connects
// the legacy smoke entry to the sole MpsBackend runtime. The production pipeline,
// buffer, and command implementation is in mps_backend.mm; do not duplicate device
// logic in this file.

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
