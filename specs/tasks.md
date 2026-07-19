# 实施任务清单

> **打勾纪律：只有代码、中文注释、测试、文档和验收全部完成后，才能把 `[ ]` 改为 `[x]`。部分完成、尚未运行或环境阻塞都不得打勾。**

> **版本纪律：在 `v3-real-world-tests/prd.md` 的真实世界数据集验收全部完成前，只允许发布 `0.x` 版本。`1.x` 只能作为真实用户稳定承诺，不能仅凭 synthetic benchmark、功能数量或普通 CI 通过发布。**

> **发布节奏：`0.2.0` 正式版完成后，不再为小型内部模块频繁发布 PyPI。下一次公开功能里程碑固定为 `0.3.0`，目标是完成 `v2-arboretum-implementation/prd.md` 中规划的 all trees 主体能力；中间开发只 commit、push 和保留 CI artifact，除非用户明确批准新的预发布。**

## S0：V0 Constitution 规格与仓库准备

- [x] S0.1 建立宪法、PRD、项目树和任务清单。
- [x] S0.2 建立每个核心模块的设计规格。
- [x] S0.3 建立 `specs/AGENTS.md` 项目 Agent 规则，并保持普通用户入口简洁。
- [x] S0.4 对照规格审计并移除既有 mock、占位 API 和重复逻辑。
- [x] S0.5 建立公开 README、许可证和真实状态说明。
- [x] S0.6 初始化 Git 仓库并确认公开远端。

## S1：V1 MPS Histogram Engine 构建与最小真实后端

- [x] S1.1 建立 CMake + Python 构建系统，版本单一来源。
- [x] S1.2 实现 Objective-C++ 设备发现与 capability 检查。
- [x] S1.3 构建期编译并打包最小真实 `.metallib`。
- [x] S1.4 实现真实向量 smoke kernel 与同步错误处理。
- [x] S1.5 构建 arm64 wheel 并在干净环境安装验证。
- [x] S1.6 验收 G0：无重量级运行依赖、无本地编译要求、无 mock。

## S2：数据与分箱

- [x] S2.1 实现输入形状、dtype、有限值和溢出验证。
- [x] S2.2 实现确定性分位数边界与重复边界处理。
- [x] S2.3 实现 `uint8/uint16` 唯一分箱表示。
- [x] S2.4 实现数据所有权、连续布局和复制诊断。
- [x] S2.5 完成常量、偏斜、空输入和边界测试。
- [x] S2.6 验收 G1：分箱完全确定、可序列化、无生命周期错误。
- [x] S2.7 提交并 push GitHub 模块，由 CI 保存已验证 wheel artifact。

## S3：CPU oracle 与单树

- [x] S3.1 实现平方误差 gradient/Hessian。
- [x] S3.2 实现唯一 split gain 与叶子权重函数。
- [x] S3.3 实现 CPU histogram 和稳定 tie-break。
- [x] S3.4 实现深度受限单棵回归树。
- [x] S3.5 实现扁平节点预测。
- [x] S3.6 用手算数据验证每个 split、叶值和预测。
- [x] S3.7 验收 G2：领域语义被冻结并有 oracle 保护。
- [x] S3.8 提交并 push GitHub 模块，由 CI 保存已验证 wheel artifact。

## S4：MPS 直方图

- [x] S4.1 实现真实 gradient/Hessian kernel。
- [x] S4.2 实现正确性基线 histogram kernel。
- [x] S4.3 实现 threadgroup partial histogram。
- [x] S4.4 实现 partial histogram reduction。
- [x] S4.5 完成多 bin、偏斜、非整组长度和数值对照测试。
- [x] S4.6 完成 profiler 记录和内存安全检查。
- [x] S4.7 验收 G3：预登记大规模 histogram 相对 CPU oracle 达到目标。
- [x] S4.8 提交并 push GitHub 模块及 benchmark，由 CI 保存 wheel artifact。

## S5：真实回归 GBDT

- [x] S5.1 实现 Trainer 状态机和多轮 boosting。
- [x] S5.2 实现树级 prediction update。
- [x] S5.3 实现 `MPSBoostRegressor.fit/predict`。
- [x] S5.4 实现未拟合、设备失败和参数冲突异常。
- [x] S5.5 实现模型保存/加载与 round-trip 测试。
- [x] S5.6 验收 G4：端到端真实训练正确，无 mock 或静默回退。
- [x] S5.7 提交并 push GitHub，发布 PyPI `0.2.0a0` 并从 PyPI 复验。

## S6：GPU 热路径优化

- [x] S6.1 实现 split scan kernel。
- [x] S6.2 实现 partition/compaction kernel。
- [x] S6.3 实现 histogram subtraction。
- [x] S6.4 实现按层活跃节点批处理。
- [x] S6.5 实现 buffer pool 和同步压缩。
- [x] S6.6 量化小节点退化区间并确定唯一调度策略。
- [x] S6.7 验收 G5：预登记端到端场景达到性能目标。
- [x] S6.8 提交并 push GitHub，发布 PyPI `0.2.0b0` 并从 PyPI 复验。

## S7：缓存与稳定性

- [x] S7.1 实现 L1 pipeline 与 buffer 进程缓存。
- [x] S7.2 实现 L2 版本 key、原子写入和损坏重建。
- [x] S7.3 实现 L3 CI 构建缓存和严格失效条件。
- [x] S7.4 实现只查询不创建、显式创建和安全清理 API。
- [x] S7.5 完成重复训练、缓存损坏和内存增长测试。
- [x] S7.6 验收 G6：删除全部缓存不影响结果。
- [x] S7.7 提交并 push GitHub，发布 PyPI `0.2.0rc0` 并从 PyPI 复验。

## S8：0.2.0 发布

- [x] S8.1 完成公共英文 README、API 和限制说明。
- [x] S8.2 完成 Apache-2.0 与依赖许可证审计。
- [x] S8.3 完成 wheel 内容、动态链接和绝对路径审计。
- [x] S8.4 完成 CPU、MPS、集成、安装和稳定性测试。
- [x] S8.5 完成可复现 benchmark 与原始报告。
- [x] S8.6 用不超过 10 行的中文 commit 提交各内聚模块。
- [x] S8.7 push 到已确认的公开 GitHub 远端。
- [x] S8.8 使用已验证 artifact 完成隔离发布验证。
- [x] S8.9 用户最终确认包名、版本和 artifact 哈希。
- [x] S8.10 发布正式 PyPI 0.2.0，并验证全新安装。

## S9：V2 Arboretum Implementation 规格与统一底座

- [x] S9.1 完成 `v2-arboretum-implementation/prd.md` 规格审查，确认 v2 范围、非目标和验收门。
- [x] S9.2 设计统一树模型族抽象：目标函数、采样、split 策略、树生长和预测聚合。
- [x] S9.3 保持现有回归 GBDT 行为与模型格式向后兼容。
- [x] S9.4 建立模型族能力矩阵和未实现能力早失败错误。
- [x] S9.5 明确 `0.3.0` all trees 发布范围，禁止为单个小模块频繁发布 PyPI。
- [x] S9.6 验收 G8：新增抽象不复制数学公式，不引入第二套训练语义。

## S10：分类 GBDT 与完整 boosting

- [x] S10.1 实现二分类 logistic objective 的唯一数学语义。
- [x] S10.2 Implement `GradientBoostingClassifier` / `MPSBoostClassifier` fit, predict, and predict_proba for strict binary 0/1 labels.
- [x] S10.3 实现 early stopping、验证集监控和训练诊断。
- [x] S10.4 Complete CPU oracle, real MPS, model I/O, and integration tests.
- [ ] S10.5 验收 G9：分类 GBDT 正确、稳定、真实 MPS 加速且无静默回退。

## S11：Random Forest

- [x] S11.1 实现 bootstrap/无放回采样和特征子采样。
- [x] S11.1a Implement `DecisionTreeRegressor` and `DecisionTreeClassifier` as one-tree native estimators.
- [x] S11.2 Implement `RandomForestRegressor.fit/predict` using independent native decision trees.
- [x] S11.3 Implement `RandomForestClassifier.fit/predict/predict_proba` using independent native decision trees.
- [ ] S11.4 实现独立树批量训练调度和共享 MPS 热路径。
- [x] S11.5 Complete regression averaging, classification probability aggregation, random seed, and model I/O tests.
- [ ] S11.6 验收 G10：Random Forest 端到端真实可用，性能和退化区间有记录。

## S12：ExtraTrees

- [x] S12.1 实现可复现随机 threshold 候选生成。
- [ ] S12.2 实现 `MPSExtraTreesRegressor`。
- [ ] S12.3 实现 `MPSExtraTreesClassifier`。
- [ ] S12.4 实现 GPU 批量评估随机 split。
- [ ] S12.5 与 Random Forest 共享采样、聚合和预测格式。
- [ ] S12.6 验收 G11：ExtraTrees 正确、稳定、真实 MPS 路径可观测。

## S13：LightGBM-like 与 CatBoost-like 训练策略

- [ ] S13.1 实现受控 leaf-wise 生长策略。
- [ ] S13.2 实现活跃叶队列、内存上限和过拟合控制。
- [ ] S13.3 扩展 histogram subtraction 与小叶调度策略。
- [ ] S13.4 对 level-wise 与 leaf-wise 做端到端 benchmark。
- [x] S13.5 设计 CatBoost-like ordered boosting、类别特征友好路径和 permutation 语义。
- [ ] S13.6 实现 `CatBoostRegressor` 与 `CatBoostClassifier` 的受控兼容入口。
- [ ] S13.7 验收 G12：LightGBM-like 与 CatBoost-like 策略真实加速且公开限制诚实。

## S14：统一推理热路径

- [ ] S14.1 实现多模型族共享扁平预测格式。
- [ ] S14.2 实现 MPS 批量 tree traversal 或等价预测优化。
- [ ] S14.3 完成保存/加载后 CPU 与 MPS 预测一致性测试。
- [ ] S14.4 量化训练加速与推理加速的适用边界。
- [ ] S14.5 验收 G13：所有已交付树模型族共享稳定推理路径。

## S15：工业表格语义

- [ ] S15.1 实现缺失值检测、默认方向训练和模型保存。
- [ ] S15.2 实现样本权重贯穿 objective、histogram、split gain、叶值和指标。
- [ ] S15.3 实现类别特征元数据、类别 split 和未知类别处理。
- [ ] S15.4 实现单调约束并验证所有相关 split 与叶值满足约束。
- [ ] S15.5 实现交互约束并验证路径特征组合不越界。
- [ ] S15.6 实现 L1、max leaves、叶值裁剪和更多正则化控制。
- [ ] S15.7 验收 G14：工业表格语义在 CPU oracle 与 MPS 后端一致。

## S16：高级目标函数与解释

- [ ] S16.1 实现 quantile objective。
- [ ] S16.2 实现 Poisson objective。
- [ ] S16.3 实现 Tweedie objective。
- [x] S16.4 Complete feature importance: gain, split count, and permutation.
- [x] S16.4a Implement gain and split-count feature importance from native fitted tree nodes.
- [x] S16.4b Implement permutation importance without duplicating prediction or scoring logic.
- [ ] S16.5 设计并实现 SHAP-like 近似解释的受控版本。
- [ ] S16.6 验收 G15：高级目标与解释能力有真实测试、文档和性能边界。

## S17：异常检测与排序学习

- [ ] S17.1 实现 `MPSIsolationForest`。
- [ ] S17.2 实现路径长度、异常分数和批量预测热路径。
- [ ] S17.3 设计 ranking 输入契约：group/query、label 和指标。
- [ ] S17.4 实现基础 ranking objective 与验证集监控。
- [ ] S17.5 完成异常检测与排序的 CPU oracle、MPS 测试和 benchmark。
- [ ] S17.6 验收 G16：异常检测与排序学习真实可用且限制清晰。

## S18：V3 Real World Tests 与 1.x 发布门

- [ ] S18.1 完成 `v3-real-world-tests/prd.md` 规格审查，确认 1.x 发布纪律。
- [ ] S18.2 建立合法、可复现的真实世界数据集矩阵。
- [ ] S18.3 覆盖回归、分类、异常检测和排序中已实现的模型族。
- [ ] S18.4 建立强 CPU 基线、项目 CPU oracle 和真实 MPS 对照报告。
- [ ] S18.5 完成训练、预测、保存、加载、缓存删除和重复训练稳定性测试。
- [ ] S18.6 完成模型质量、端到端性能、内存峰值、wheel 体积和权限审计。
- [ ] S18.7 公开真实数据集报告，诚实记录成功、退化和不支持场景。
- [ ] S18.8 验收 G17：真实世界测试矩阵全部通过，才允许规划 `1.0.0`。
- [ ] S18.9 用户最终确认 1.x 公开承诺范围、版本号和 artifact 哈希。
- [ ] S18.10 发布 PyPI `1.0.0` 并从正式 PyPI 全新环境复验。
