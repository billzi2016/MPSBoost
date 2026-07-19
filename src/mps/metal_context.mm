// MPSBoost 最小 Metal 运行时实现。
//
// 职责：发现默认设备、加载构建期 metallib、创建 pipeline、管理共享 buffer、提交并
// 检查 command。算法参数与树训练语义不得进入此文件；后续资源复用应抽取到独立
// MetalContext/BufferPool，而不是复制本文件的初始化逻辑。

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "mpsboost/backend.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <sstream>

namespace mpsboost {
namespace {

// 把 NSError 转成包含阶段信息的 UTF-8 文本。错误对象可能为空，因此不能无条件解引用。
std::string DescribeError(const char* stage, NSError* error) {
  std::ostringstream message;
  message << stage;
  if (error != nil) {
    message << ": " << [[error localizedDescription] UTF8String];
  }
  return message.str();
}

// 获取默认设备并把“无设备”提升为执行错误。查询接口允许不可用，但真正计算入口必须
// 早失败，避免后续对 nil Objective-C 对象发送消息后产生模糊结果。
id<MTLDevice> RequireDefaultDevice() {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (device == nil) {
    throw BackendError("MPS 后端不可用：系统没有返回默认 Metal 设备");
  }
  return device;
}

// 检查元素数量到字节数的乘法，防止超大输入在分配前发生 size_t 溢出。
std::size_t CheckedByteSize(std::size_t count) {
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw BackendError("GPU smoke 输入过大：字节数计算溢出");
  }
  return count * sizeof(float);
}

}  // namespace

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
  if (left.size() != right.size()) {
    throw BackendError("GPU smoke 输入长度不一致");
  }
  if (left.empty()) {
    return {};
  }

  const std::size_t byte_size = CheckedByteSize(left.size());

  @autoreleasepool {
    id<MTLDevice> device = RequireDefaultDevice();
    NSString* library_path = [NSString stringWithUTF8String:metallib_path.c_str()];
    if (library_path == nil || ![[NSFileManager defaultManager] fileExistsAtPath:library_path]) {
      throw BackendError("MPS shader library 不存在或路径不是有效 UTF-8");
    }

    // Metal 的路径加载接口已经弃用。使用 file URL 能保留路径语义，并避免把空 URL
    // 传入设备 API；路径存在性检查仍在前面完成，以提供更清楚的用户错误。
    NSURL* library_url = [NSURL fileURLWithPath:library_path isDirectory:NO];
    if (library_url == nil) {
      throw BackendError("无法为 MPS shader library 创建文件 URL");
    }

    NSError* library_error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:library_url error:&library_error];
    if (library == nil) {
      throw BackendError(DescribeError("加载 MPS shader library 失败", library_error));
    }

    id<MTLFunction> function = [library newFunctionWithName:@"vector_add"];
    if (function == nil) {
      throw BackendError("MPS shader library 缺少 vector_add 函数");
    }

    NSError* pipeline_error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&pipeline_error];
    if (pipeline == nil) {
      throw BackendError(DescribeError("创建 vector_add pipeline 失败", pipeline_error));
    }

    const MTLResourceOptions shared = MTLResourceStorageModeShared;
    id<MTLBuffer> left_buffer = [device newBufferWithBytes:left.data()
                                                    length:byte_size
                                                   options:shared];
    id<MTLBuffer> right_buffer = [device newBufferWithBytes:right.data()
                                                     length:byte_size
                                                    options:shared];
    id<MTLBuffer> output_buffer = [device newBufferWithLength:byte_size options:shared];
    if (left_buffer == nil || right_buffer == nil || output_buffer == nil) {
      throw BackendError("分配 MPS smoke 共享 buffer 失败");
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (queue == nil || command == nil || encoder == nil) {
      throw BackendError("创建 MPS command queue/buffer/encoder 失败");
    }

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:left_buffer offset:0 atIndex:0];
    [encoder setBuffer:right_buffer offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];

    // threadgroup 宽度不超过 pipeline 上限。dispatchThreads 会自动生成最后一个不足整组
    // 的 threadgroup，shader 中仍保留 count 边界检查作为内存安全的第二道防线。
    const NSUInteger group_width =
        std::min<NSUInteger>(256, [pipeline maxTotalThreadsPerThreadgroup]);
    const MTLSize grid = MTLSizeMake(left.size(), 1, 1);
    const MTLSize group = MTLSizeMake(group_width, 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];

    if ([command status] != MTLCommandBufferStatusCompleted) {
      throw BackendError(DescribeError("MPS vector_add command 执行失败", [command error]));
    }

    std::vector<float> output(left.size());
    std::memcpy(output.data(), [output_buffer contents], byte_size);
    return output;
  }
}

}  // namespace mpsboost
