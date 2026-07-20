# 模块设计：MPS 后端

## 1. 职责

实现 Apple GPU 设备发现、资源管理、shader 加载、kernel 调度、同步和错误转换。对用户统一称为 MPS 后端；内部可使用适合的 MPS 优化能力和自定义 Metal Compute kernel。

## 2. 最小权限

后端只使用公开用户态 GPU API，不需要管理员权限、系统扩展、后台服务、网络、相机、文件访问 entitlement 或遥测。导入包不创建设备；`fit()` 或显式诊断时才初始化。

## 3. 运行时对象

- `MetalContext`：device、command queue、shader library 与 capability；
- `PipelineRegistry`：按 kernel、function constants、ABI 版本缓存 pipeline；
- `BufferPool`：复用临时 buffer，按大小等级管理；
- `MPSBackend`：实现 `ComputeBackend`，不暴露 Objective-C 对象给核心。

对象所有权使用 RAII 包装。command 未完成前，相关 buffer 不得归还池或析构。

## 4. Shader 发布

- 构建期编译 `.metal → .air → .metallib`；
- wheel 包含版本匹配的 `.metallib`；
- 普通用户机器不运行 shader 编译器；
- native ABI、shader ABI 和包版本必须校验；
- resource 缺失或 function 不存在时在训练前失败。

## 5. 核心 kernel

### 梯度

一线程处理一行，平方误差输出 `float2(grad, hess)`。长度检查在 host，kernel 仍使用 grid 边界保护。

### 直方图

正确性基线允许全局原子；生产版本必须使用 threadgroup 局部直方图和第二阶段归并，降低热点冲突。布局与 tile 参数由真实 benchmark 冻结。

### 后续热路径

split scan、partition、histogram subtraction 和 prediction update 只有相应任务完成后进入生产。不存在的 kernel 不得用 host 实现报告为 GPU 能力。

## 6. 内存

- 控制数据优先 shared storage；
- GPU 热 workspace 对比 shared/private 后冻结；
- `uint8` bins 减少带宽；
- 训练开始预分配最大已知 workspace；
- 每层只清理实际区域；
- 记录峰值 allocated size 并在 OOM 前估算。

统一内存不等于无成本：同步、cache coherence、页面驻留和重复 buffer 仍必须 profiling。

## 7. 调度与同步

- 按层批量编码 command，避免每节点独立提交；
- CPU 只在需要紧凑 split 结果或阶段边界时同步；
- 每个 command buffer 检查 status/error；
- debug 开启验证，release 保留边界和错误上下文；
- 不允许 fire-and-forget 后立即释放资源。

## 8. Capability

后端必须验证 arm64 macOS、Metal device、所需 GPU family/原子能力、最大 threadgroup memory 和 shader 资源。不能只依据芯片名称猜测。诊断返回非敏感能力摘要。

## 9. 性能门

每项优化记录数据规模、芯片、冷/热缓存、kernel 与端到端时间、内存和误差。至少两个规模有效才可默认启用。特殊设备参数必须有保守通用路径并进入 L2 调优缓存 key。

## 10. 验收

- 真实设备运行，不用 mock command queue；
- 非 threadgroup 整倍数、偏斜 bin、最大 bin 和大梯度通过；
- command 失败可复现且资源安全；
- 重复训练无 buffer 泄漏；
- wheel 环境无需开发工具即可加载 shader。
