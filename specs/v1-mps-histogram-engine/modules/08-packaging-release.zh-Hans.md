# 模块设计：构建、打包与发布

## 1. 构建目标

使用 `scikit-build-core`、CMake、Objective-C++ 和构建期 Metal shader 编译生成 arm64 macOS wheel。版本只有一个权威来源，并注入 Python、native 与 shader ABI。

## 2. wheel 内容

只包含：

- Python 公共模块；
- 单个必要 native extension；
- 匹配的 `.metallib`；
- 包元数据和许可证。

不包含 specs、AGENTS、测试、benchmark、源码中间件、缓存、调试日志、凭据、SDK 文件或构建机绝对路径。

## 3. 体积

发布 job 记录 wheel 压缩/解压体积、最大文件和动态依赖。release 构建处理调试符号但保留崩溃诊断所需版本信息。新增依赖必须说明体积与功能收益。

## 4. CI

CI 使用最小权限 `contents: read`。测试矩阵覆盖支持 Python 版本和受支持 macOS runner。发布 workflow 与 CI 分离，只有版本 tag 和受保护环境可以获得发布权限。

优先使用 PyPI Trusted Publishing；若环境暂不具备，token 必须最小项目权限、只存在 secret、不得输出。不得重复交互登录或把凭据写入配置文件。

## 5. Git

- `specs/` 与 `specs/AGENTS.md` 作为项目规则资产进入源码；缓存、`dist/` 和本地构建产物保持忽略；
- 每个内聚模块独立中文 commit；
- commit 总计不超过 10 行，说明结果、原因和验证；
- push 前审查 status、diff、忽略规则、敏感信息与测试结果；
- 不提交未验证 artifact。

## 6. 发布顺序

1. 所有发布任务实际完成；
2. 从干净 commit 构建一次 artifact；
3. 对该 artifact 运行 metadata、内容、链接、体积和安装测试；
4. 记录 SHA-256；
5. 隔离环境验证发布安装；
6. 用户确认包名、版本和哈希；
7. 上传同一 artifact 到正式 PyPI；
8. 从 PyPI 新建环境安装并运行真实 smoke fit/predict；
9. 发布英文 release notes。

任何一步失败都停止，不得重新上传相同版本或跳过验证。

## 7. 0.2.0 规则

0.2.0 必须包含真实训练能力。禁止发布只有命名空间、假 API 或 `NotImplementedError` 训练的占位包。若只需提前保留名称，应使用明确的预发布版本和完全诚实的元数据，但仍须用户在最终上传前单独确认。

## 8. 验收

- 一条 pip 命令完成安装；
- 无管理员权限、网络运行依赖或本地编译；
- wheel 体积与依赖审计通过；
- 干净环境真实训练和预测通过；
- artifact 与测试哈希完全一致；
- GitHub 与 PyPI 页面只声明已实现能力。

## 9. 大模块交付与 PyPI 里程碑

每个完成验收的大模块都必须形成 GitHub 模块 commit/push，并由 CI 保存已验证 wheel artifact。内部实现模块不单独上传 PyPI，避免堆积用户无法直接使用的版本。

PyPI 里程碑固定为：S5 首个真实 estimator 发布 `0.2.0a0`；S6 性能验收发布 `0.2.0b0`；S7 稳定性验收发布 `0.2.0rc0`；完整发布门发布 `0.2.0`。每个 PyPI artifact 必须来自对应 GitHub 源码状态，上传前完成哈希、内容、安装与真实测试审计，上传后从正式 PyPI 全新安装复验。
