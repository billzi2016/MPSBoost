# 模块设计：测试、基准与质量

## 1. 测试金字塔

- 单元：数学、分箱、参数、模型格式；
- CPU oracle：手算树与确定性；
- MPS kernel：真实设备逐 buffer 对照；
- 集成：Python → native → MPS → model；
- 安装：干净 wheel 环境 smoke fit/predict；
- 稳定：重复训练、异常、缓存损坏、内存压力；
- benchmark：独立进程、固定数据与参数。

## 2. 禁止事项

- 不 mock GPU 验收真实设备功能；
- 不删除或跳过失败测试掩盖缺陷；
- 不用实现本身计算期望值；
- 不为通过测试任意放宽浮点容差；
- 不只报告最快一次或排除预处理。

## 3. 数值比较

使用 `abs(actual-expected) <= atol + rtol*abs(expected)`，按 kernel、累计规模和模型 metric 分别冻结容差。调整容差必须附误差分布和根因。

比较层级：逐 bin `count/G/H`、split feature/bin/gain、树结构/叶值、逐样本预测、最终 metric。

## 4. benchmark

预登记 Small、Medium、Large、Wide 四类合成数据与至少两个可合法分发/下载的真实数据集。记录：

- 设备、系统、版本、数据哈希和参数；
- CPU 线程；
- cold/warm cache；
- 输入转换、分箱、初始化、kernel、同步、总 fit；
- 峰值 RSS 与设备内存；
- 模型质量。

公开结论必须展示小数据退化区间，不使用“最高可达”代替完整结果。

## 5. 代码质量

- C++ 开启严格 warning 并把项目 warning 视为错误；
- Python lint、类型检查和测试；
- shader 编译警告审计；
- sanitizer 能运行的 CPU/native 路径纳入 CI；
- 中文文件头、函数契约和关键点注释由 review 检查。

## 6. 安全

fuzz 模型 loader 和尺寸计算；验证缓存清理路径；wheel 审计秘密、绝对路径和不可分发内容；运行时验证无隐式网络和额外权限。

## 7. 完成门

模块测试、跨模块集成、安装 smoke、性能 sanity 和注释审查全部通过，任务才可打勾。测试环境阻塞必须记录为未完成。

