# 实施任务清单

> **打勾纪律：只有代码、中文注释、测试、文档和验收全部完成后，才能把 `[ ]` 改为 `[x]`。部分完成、尚未运行或环境阻塞都不得打勾。**

> **版本纪律：在 `v3-real-world-tests/prd.md` 的真实世界数据集验收全部完成前，只允许发布 `0.x` 版本。`1.x` 只能作为真实用户稳定承诺，不能仅凭 synthetic benchmark、功能数量或普通 CI 通过发布。**

> **当前发布纪律：`0.3.0` 是大规模验证前的 all-trees 功能里程碑。`0.4.0` 是大规模和真实世界数据集验证之后的版本。`0.5.0` 是 known issue 清零和体验兜底版本。所有计划能力、真实世界矩阵、性能/内存/权限审计、artifact hash、安装/环境 fallback、客户侧失败路径、文档页面、release audit、CI run、PyPI artifact 和 fresh-install verification 全部完成并明确关闭前，不得发布 `1.0.0`。**

> **发布节奏：`0.2.0` 正式版完成后，不再为小型内部模块频繁发布 PyPI。下一次公开功能里程碑固定为 `0.3.0`，目标是完成 `v2-arboretum-implementation/prd.md` 中规划的 all trees 主体能力；中间开发只 commit、push 和保留 CI artifact，除非用户明确批准新的预发布。**

> **当前执行顺序：S15 完成后必须优先推进 S18 真实世界数据集测试与报告，然后推进 S16 高级目标函数与解释，最后推进 S17 异常检测与排序学习。除非用户明确调整顺序，否则不得按编号顺序直接从 S15 跳到 S16 或 S17。**

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
- [x] S10.5 验收 G9：分类 GBDT 正确、稳定、真实 MPS 加速且无静默回退。

## S11：Random Forest

- [x] S11.1 实现 bootstrap/无放回采样和特征子采样。
- [x] S11.1a Implement `DecisionTreeRegressor` and `DecisionTreeClassifier` as one-tree native estimators.
- [x] S11.2 Implement `RandomForestRegressor.fit/predict` using independent native decision trees.
- [x] S11.3 Implement `RandomForestClassifier.fit/predict/predict_proba` using independent native decision trees.
- [x] S11.4a Implement deterministic `n_jobs` scheduling for independent random-forest trees.
- [x] S11.4 实现独立树批量训练调度和共享 MPS 热路径。
- [x] S11.5 Complete regression averaging, classification probability aggregation, random seed, and model I/O tests.
- [x] S11.6 验收 G10：Random Forest 端到端真实可用，性能和退化区间有记录。

## S12：ExtraTrees

- [x] S12.1 实现可复现随机 threshold 候选生成。
- [x] S12.2 Implement `ExtraTreesRegressor` using native random-threshold split candidates.
- [x] S12.3 Implement `ExtraTreesClassifier` using native random-threshold split candidates.
- [x] S12.4 Implement native random-threshold split policy for CPU and MPS tree training.
- [x] S12.5 Share sampling, aggregation, and forest container prediction format with Random Forest.
- [x] S12.6 验收 G11：ExtraTrees 正确、稳定、真实 MPS 路径可观测。

## S13：LightGBM-like 与 CatBoost-like 训练策略

- [x] S13.1 实现受控 leaf-wise 生长策略。
- [x] S13.2 实现活跃叶队列、内存上限和过拟合控制。
- [x] S13.3 扩展 histogram subtraction 与小叶调度策略。
- [x] S13.4 对 level-wise 与 leaf-wise 做端到端 benchmark。
- [x] S13.5 设计 CatBoost-like ordered boosting、类别特征友好路径和 permutation 语义。
- [x] S13.6 实现 `CatBoostRegressor` 与 `CatBoostClassifier` 的受控兼容入口。
- [x] S13.7 验收 G12：LightGBM-like 与 CatBoost-like 策略真实加速且公开限制诚实。

## S14：统一推理热路径

- [x] S14.1 实现多模型族共享扁平预测格式。
- [x] S14.2 实现 MPS 批量 tree traversal 或等价预测优化。
- [x] S14.3 完成保存/加载后 CPU 与 MPS 预测一致性测试。
- [x] S14.4 量化训练加速与推理加速的适用边界。
- [x] S14.5 验收 G13：所有已交付树模型族共享稳定推理路径。

## S15：工业表格语义

- [x] S15.1 实现缺失值检测、默认方向训练和模型保存。
- [x] S15.2 实现样本权重贯穿 objective、histogram、split gain、叶值和指标。
- [x] S15.3 实现类别特征元数据、类别 split 和未知类别处理。
- [x] S15.4 实现单调约束并验证所有相关 split 与叶值满足约束。
- [x] S15.5 实现交互约束并验证路径特征组合不越界。
- [x] S15.6 实现 L1、max leaves、叶值裁剪和更多正则化控制。
- [x] S15.7 验收 G14：工业表格语义在 CPU oracle 与 MPS 后端一致。

## S16：高级目标函数与解释

> 排期：本阶段在 S18 真实世界数据集测试完成后推进。

- [x] S16.1 实现 quantile objective。
- [x] S16.2 实现 Poisson objective。
- [x] S16.3 实现 Tweedie objective。
- [x] S16.4 Complete feature importance: gain, split count, and permutation.
- [x] S16.4a Implement gain and split-count feature importance from native fitted tree nodes.
- [x] S16.4b Implement permutation importance without duplicating prediction or scoring logic.
- [x] S16.5 设计并实现 SHAP-like 近似解释的受控版本。
- [x] S16.5a 设计官方 SHAP 集成路径：可选依赖 `mpsboost[shap]`、native tree adapter/export、TreeExplainer 语义验证和科研示例；不得把近似解释声明为官方 SHAP。
- [x] S16.6 验收 G15：高级目标与解释能力有真实测试、文档和性能边界。

## S17：异常检测与排序学习

> 排期：本阶段在 S16 高级目标函数与解释完成后推进。

- [x] S17.1 实现 `MPSIsolationForest`。
- [x] S17.2 实现路径长度、异常分数和批量预测热路径。
- [x] S17.3 设计 ranking 输入契约：group/query、label 和指标。
- [x] S17.4 实现基础 ranking objective 与验证集监控。
- [x] S17.5 完成异常检测与排序的 CPU oracle、MPS 测试和 benchmark。
- [x] S17.6 验收 G16：异常检测与排序学习真实可用且限制清晰。

## S18：V3 Real World Tests 与 1.x 发布门

> 排期：S15 工业表格语义完成后立即进入本阶段，用真实世界数据集优先验证已完成模型族。

- [x] S18.1 完成 `v3-real-world-tests/prd.md` 规格审查，确认 1.x 发布纪律。
- [x] S18.2 建立合法、可复现的真实世界数据集矩阵。
- [x] S18.2a 建立 `tests/real_world/` 目录和真实世界测试规则。
- [x] S18.2b 实现内置数据集验收：Iris、Breast Cancer、Diabetes、Digits。
- [x] S18.2c 实现缓存下载数据集验收：California Housing。
- [x] S18.2d 实现 opt-in 外部数据集验收：MNIST subset、Titanic、Adult Income、Covertype subset、Higgs subset。
- [x] S18.2d-1 实现 MNIST subset 验收。
- [x] S18.2d-2 实现 Titanic 验收。
- [x] S18.2d-3 实现 Adult Income 验收。
- [x] S18.2d-4 实现 Covertype subset 大行数验收和真实 MPS parity smoke。
- [x] S18.2d-5 实现 Higgs subset 验收。
- [x] S18.3 覆盖回归、分类、异常检测和排序中已实现的模型族。
- [x] S18.4 建立强 CPU 基线、项目 CPU oracle 和真实 MPS 对照报告。
- [x] S18.5 完成训练、预测、保存、加载、缓存删除和重复训练稳定性测试。
- [x] S18.5a 覆盖真实数据集上的模型保存、加载、缓存删除、缓存损坏和重复训练稳定性。
- [ ] S18.6 完成模型质量、端到端性能、内存峰值、wheel 体积和权限审计。
- [ ] S18.6a 记录真实数据集训练时间、预测时间、内存峰值、模型大小、wheel 体积和权限范围。
- [x] S18.7 公开真实数据集报告，诚实记录成功、退化和不支持场景。
- [ ] S18.8 验收 G17：真实世界测试矩阵全部通过后才允许规划 `1.0.0`，并且包含质量、性能、内存、模型体积、wheel 体积和权限证据。
- [ ] S18.9 用户最终确认 1.x 公开承诺范围、版本号、artifact hash、文档完整性和客户侧失败路径行为。
- [ ] S18.10 只有所有文档和 release audit 最终完成后，才能发布 PyPI `1.0.0`，并从正式 PyPI 全新环境复验。

## S24：0.4.0 大规模验证版本

- [x] S24.1 明确 `0.4.0` 是大规模和真实世界数据集验证之后的 0.x 版本，区别于大规模验证前的 `0.3.0` all-trees 里程碑。
- [ ] S24.2 只有当前 0.4.0 wheel artifact、CI 结果和 smoke verification 都记录后，才能发布 PyPI `0.4.0`。

## S25：0.5.0 Known Issue 清零加固版本

- [x] S25.1 把所有已知 runtime、documentation、packaging、environment 和 user-experience issue 分类为已修复、有意延期或当前平台不可能支持。
- [x] S25.2 确保缺可选依赖、缺 Metal toolchain、不支持的 Linux/CUDA 环境、适合 CPU 的 workload 都给出可复制 guidance 或 warning，而不是让用户遇到困惑失败。
- [ ] S25.3 只有没有已知 blocking customer-facing issue 时，才能发布 PyPI `0.5.0`。

## S26：1.0.0 最终客户承诺门

- [ ] S26.1 冻结最终公开范围，确认没有计划能力被静默排除在 `1.0.0` 承诺之外。
- [ ] S26.2 确认文档完整、双语、链接正确，且没有过期版本声明。
- [ ] S26.3 确认 release audit、known-issue audit、performance report、artifact hash、CI 结果和 PyPI fresh-install verification 全部完成。
- [ ] S26.4 确认客户侧失败在无法继续执行时使用 warning、可复制 setup command 或清晰 external-dependency attribution。
- [ ] S26.5 只有 S26.1-S26.4 完成且用户明确批准最终发布后，才能发布 PyPI `1.0.0`。

## S19：文件结构达到发布维护标准

- [x] S19.1 建立文件长度规则：默认 200 行目标，超过 300 行必须拆分或登记例外。
- [x] S19.2 拆分 Python estimator 实现，保持 public API 与历史 import 路径不变。
- [x] S19.3 拆分 estimator unit tests，避免单个测试文件继续膨胀。
- [x] S19.4 拆分 Python binding 文件，隔离 buffer、dataset、model、backend test helpers。
- [x] S19.5 拆分 native binned dataset 实现，隔离 validation、quantization、schema 和 serialization。
- [x] S19.6 拆分 native tree 实现，隔离 split scan、growth、prediction 和 restore validation。
- [x] S19.7 验收 G18：所有新增/重构文件满足长度规则，例外均有说明和后续任务。
- [x] S19.8 拆分 MPS backend 实现，隔离 Metal context、gradient、histogram、split partition 与 lifetime glue。

## S20：清理程序文件中文遗留问题

- [x] S20.1 开始前必须阅读 `specs/legacy-issues.md`，并按其中的搜索范围、禁止事项和完成标准执行。
- [x] S20.2 程序文件中的注释、docstring、测试说明、断言消息、运行时错误和包元数据必须全部翻译为英文。
- [x] S20.3 文档文件不纳入本任务；`specs/`、README、未来双语站点内容允许继续保留中文。
- [x] S20.4 必须使用 `specs/legacy-issues.md` 中规定的 `rg` 命令查找中文，并在新增程序目录后同步扩展搜索范围。
- [x] S20.5 完成后只能在搜索命令无程序文件中文命中、相关测试通过、且没有行为改动时打勾。

## S21：原生多分类 Softmax 与 sklearn 兼容接口

- [x] S21.1 明确多分类路线：OvR 只能作为 compatibility layer/fallback，不能作为最终默认最佳实现。
- [x] S21.2 保持 sklearn/XGBoost 风格 public API 不变：`fit`、`predict`、`predict_proba`、`decision_function`、`score`、`get_params`、`set_params`、`GridSearchCV` 必须继续可用。
- [x] S21.3 增加 native multiclass softmax objective 规格：`num_class`、class 编码、base margin、softmax probability、gradient/Hessian、样本权重和数值稳定规则必须写清楚。
- [x] S21.4 实现 CPU oracle 的 native softmax 多分类训练，不允许用 OvR 测试假装原生 softmax 已完成。
- [x] S21.5 扩展 native model 格式，保存 `num_class`、多分类 objective、class mapping 和多 class tree/update 结构，并保持旧模型格式兼容读取。
- [x] S21.6 实现 Python classifier 的 `multi_strategy="auto" | "softmax" | "ovr"`，其中 `auto` 在 native softmax 可用时默认选择 softmax。
- [x] S21.7 实现 `predict_proba` 的 native softmax 输出，概率每行必须归一、有限，并与 `predict` 的 argmax class 一致。
- [x] S21.8 增加 MPS native softmax 路径或明确分阶段门槛；在 MPS 未完成前不得把 CPU softmax 报告为 MPS softmax。
- [x] S21.9 覆盖 Iris、Digits、Covertype subset 等真实多分类数据集，并同时验证 CPU oracle、MPS 行为和 sklearn model-selection 兼容性。
- [x] S21.10 验收 G19：默认多分类实现达到 native softmax，OvR 仅保留为显式 fallback/兼容策略。

## S22：跨平台兼容后端与统一入口

- [x] S22.1 设计 portable backend policy：MPSBoost native CPU/MPS 后端继续作为默认与 correctness oracle；Apple Silicon 优先 native，Linux CUDA 可选择 XGBoost GPU，通用 CPU 可选择 native CPU 或 sklearn/XGBoost CPU，并且必须在 summary 中暴露实际 backend。
- [x] S22.2 增加可选依赖 extras：`mpsboost[xgboost]`、`mpsboost[sklearn]`、`mpsboost[cuda]`，默认安装仍保持轻量，不强制拉取重型依赖。
- [x] S22.3 实现统一 estimator adapter：保持 `fit`、`predict`、`predict_proba`、`score`、`get_params`、`set_params` 和 model-selection 行为一致。
- [x] S22.4 实现环境诊断与安装提示：缺 CUDA/XGBoost/sklearn 时给出复制即用安装命令，不使用交互 `input()`，可通过环境变量跳过诊断。
- [x] S22.5 明确边界：外部后端必须是显式 portable mode 或 `device="auto"` 的可观测选择，不替代 native CPU oracle；summary 必须报告实际 backend 和策略。
- [x] S22.6 通过 native macOS 测试、显式 external-backend adapter policy、可选依赖诊断和 backend summary 断言覆盖 macOS MPS、macOS CPU、Linux CPU、Linux CUDA smoke matrix。Linux/CUDA runtime failure 归属所选 external sklearn/XGBoost/CUDA stack，不归因于 native MPSBoost CPU/MPS。
- [x] S22.7 验收 G20：同一用户接口可把 Apple Silicon 路由到 native CPU/MPS，把普通 Linux 或 CUDA Linux 路由到显式 external backend，且依赖、性能预期和实际 backend 对用户透明可查。

## S23：文档站点翻译与国际化

- [x] S23.1 梳理所有必须进入文档站点的 Markdown 源文件。项目文档和项目 specs 必须在源文件所在目录原地翻译，例如 `README.md` 与 `README.zh-Hans.md` 并列，`specs/tasks.md` 与 `specs/tasks.zh-Hans.md` 并列。
- [x] S23.2 `docs-site` 自己的 PRD 源文件必须保留在 `docs-site/specs/`，英文 `*.md` 与简体中文 `*.zh-Hans.md` 同目录并列；这些文件不得移动到根目录 `specs/`。
- [x] S23.3 `docs-site/docs/en/` 与 `docs-site/docs/zh-Hans/` 只作为语言导航树；凡是源文件已存在于项目其他目录的页面，必须使用 symlink 指向源文件，不得复制一份 Markdown 到 `docs-site/docs/`。
- [x] S23.4 文档站点 PRD 导航目录使用 `docs-site/docs/en/docs-site-prd/` 与 `docs-site/docs/zh-Hans/docs-site-prd/`，两边都通过 symlink 指回 `docs-site/specs/`。
- [x] S23.5 翻译 README、`docs/`、`ai-skills/`、核心 specs、docs-site PRD、benchmark 文档、测试文档和用户指南页面。翻译必须发生在对应源文件归属目录，不得在 `docs-site/` 下维护另一套重复翻译副本。
- [x] S23.6 增加并维护 MkDocs i18n 配置，建立 `en/` 与 `zh-Hans/` 平行导航结构。英文导航不得指向中文文件名，中文导航不得指向仅英文内容；尚未翻译的页面必须显式标记，不得静默混用。
- [x] S23.7 校验中英文页面链接、术语、版本号、后端策略、PyPI 安装命令和环境诊断命令一致。
- [x] S23.8 翻译纪律：禁止缩减、摘要化、删段、合并要点、简化警告、删除限制说明，或用概括性文字替代具体命令；英文页面与中文页面必须保持章节结构、信息量、代码块、命令、约束、限制说明和验收口径一致。
- [x] S23.9 验收 G21：文档站点具备可维护的双语矩阵，所有既有 Markdown 源文件都有明确双语路径，symlink 全部有效，MkDocs strict build 通过，且 release history 从 `0.1.0a0` 到 `0.4.0` 保持 append-only。
