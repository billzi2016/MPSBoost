# All Trees v2 规格

## 1. 目标

All Trees v2 的目标是把所有有实际价值、适合本地表格学习的树模型能力逐步迁移到
Apple Silicon 的 MPS/Metal 后端上。MPSBoost 不应只停留在单一回归 GBDT，而应把
通用树训练热路径做成可复用底座，让 boosting、bagging、随机切分和推理服务共享
同一套数据、内存、调度和模型语义。

本文件是 0.2.0 之后的 v2 产品方向规格。实现必须继续遵守宪法、PRD、SOLID/DRY、
真实测试和不可 mock 规则。

## 1.1 v2 及以后语言规范

从 v2 开始，除 Agent 与用户在对话中继续使用中文外，项目持久材料必须使用英文。
这包括代码文件头部意图注释、函数注释、关键实现注释、commit message、README、
release notes、CI 文案、错误文档和公开示例说明。这样做是为了让项目后续面向国际
用户、外部 contributor、PyPI/GitHub 生态和学术/工程复现时保持一致。

v0/v1 已存在的中文材料可以保留作为历史规格和开发记录；v2 及以后的新增或大幅修改
代码、公开文档和发布材料必须转为英文。specs 不强制就地翻译；中英文站点通过 MkDocs
维护或生成独立语言版本。若需要在对话中解释设计取舍，Agent 仍然使用中文。

## 2. 范围

### 2.1 必须覆盖的树模型族

- Histogram GBDT：当前 MPSBoost 回归 GBDT 的主线，继续扩展分类、early stopping、
  正则化、采样和更完整的训练热路径。
- LightGBM-like leaf-wise / level-wise boosting：支持直方图复用、histogram
  subtraction、按层或按叶调度和小叶子退化策略，但不得复制任何第三方实现。
- XGBoost-like estimator：提供熟悉的参数命名、fit/predict/save/load 体验和清晰
  错误语义；兼容体验不等于复制源码或承诺完全 API 等价。
- Random Forest：支持独立树并行训练、行采样、列采样、回归和分类投票/平均。
- ExtraTrees：支持随机 threshold 候选、随机特征子集和高并行独立树训练。
- CatBoost-like ordered boosting：支持 ordered boosting 语义、类别特征友好路径、
  可复现 permutation 策略和分类/回归入口，但不得复制任何第三方实现或承诺二进制兼容。
- CART 单树：作为调试、教学、oracle 和轻量模型路径保留，但不得形成第二套训练语义。
- 推理优化：所有训练得到的树模型必须进入统一扁平模型格式，并支持批量预测热路径。
- Isolation Forest：用于异常检测，利用独立树并行训练和批量路径长度预测。
- Quantile / Poisson / Tweedie boosting：覆盖常见非平方误差回归目标。
- Ranking trees：用于排序学习，但必须在基础分类/回归稳定后再进入实现。

### 2.2 明确暂缓的能力

- 分布式训练、多 GPU 和非 Apple Silicon 平台；
- 多输出、多标签和复杂类别特征组合；
- 与第三方库的完全参数兼容、模型二进制兼容或源码兼容；
- 为了营销提前暴露未完成模型入口。

## 3. 统一底座

所有模型族必须共享以下底座，不允许为每个模型复制一套低质量实现：

- 输入验证、dtype、有限值、连续性和生命周期检查；
- 确定性分箱、schema、`uint8/uint16` 紧凑表示和模型内边界保存；
- split gain、叶子权重、最小样本、最小 Hessian 和稳定 tie-break 的领域函数；
- MPS gradient、histogram、split scan、partition/compaction、prediction update；
- buffer pool、pipeline cache、dispatch plan 和同步错误处理；
- 模型 I/O、版本校验、权限最小化和无遥测原则；
- benchmark、CPU oracle、真实 Metal 测试和干净 wheel 安装验证。

## 3.1 表格学习必备语义

以下能力是把树模型从“能跑”提升到“工业可用”的关键语义，必须集中设计，不能在
每个 estimator 内临时拼凑：

- 缺失值：支持 NaN 检测、默认方向、训练期最佳默认方向选择和模型保存。
- 类别特征：支持显式类别元数据、低基数类别 split、高基数降级策略和未知类别处理。
- 样本权重：所有 objective、histogram、split gain、叶值和评价指标必须一致处理权重。
- 单调约束：支持特征级 monotonic constraints，并在 split 选择和叶值更新中强制满足。
- 交互约束：支持限制特征组合进入同一路径，避免模型违反业务约束。
- 正则化：支持 L1、L2、min gain、min child weight、max leaves 和叶值裁剪。
- 解释能力：支持 gain、split count、permutation importance 和后续 SHAP-like 近似。
- 训练可观测性：记录每轮指标、树数量、有效 split、退化原因和设备阶段耗时。

这些语义必须进入 CPU oracle，再进入 MPS 后端。禁止只在 Python 层预处理后声明为
后端支持。

## 4. 设计原则

### 4.1 用户入口

公共入口保持 estimator 风格。v2 可以新增专用 estimator，例如：

- `GradientBoostingRegressor`
- `GradientBoostingClassifier`
- `RandomForestRegressor`
- `RandomForestClassifier`
- `ExtraTreesRegressor`
- `ExtraTreesClassifier`
- `CatBoostRegressor`
- `CatBoostClassifier`
- `MPSBoostRegressor`
- `MPSBoostClassifier`
- `MPSRandomForestRegressor`
- `MPSRandomForestClassifier`
- `MPSExtraTreesRegressor`
- `MPSExtraTreesClassifier`

简洁名称是主入口，`MPS*` 名称作为项目品牌和向后兼容别名保留。参数命名应尽量接近用户熟悉的树模型生态，但未知参数必须早失败。兼容层只做用户体验
映射，不得把第三方项目的内部结构、名称或源码变成项目依赖。

模型选择默认复用标准 sklearn 协议和工具链，例如 `GridSearchCV`、`RandomizedSearchCV`
和 `cross_val_score`。项目不得轻易新增 `MPSGridSearchCV` 这类专用搜索 API；只有未来
真实 benchmark 证明 GPU 批量超参数调度需要独立 API，且该 API 能显著降低用户复杂度
或提升端到端性能时，才允许进入规格评审。

### 4.2 后端能力

MPS 后端只暴露计算能力，不拥有产品参数语义。训练核心决定模型族、采样、目标函数、
树生长策略和最终 split 验证；后端负责真实执行可批量化的热路径。

GPU 热路径优先级：

1. 分箱后数据常驻或低复制上传；
2. 多节点、多树、多 feature 的批量 histogram；
3. split scan 与 histogram subtraction；
4. 稳定 partition/compaction；
5. 独立树并行调度；
6. 预测 traversal 批处理；
7. buffer pool 和同步压缩。

### 4.3 正确性

每个模型族都必须先有 CPU oracle 和小数据手算/性质测试，再接入 MPS 热路径。MPS 结果
可以存在有依据的浮点容差，但树结构、节点约束和错误语义必须可解释、可复现。

### 4.4 性能

性能结论必须按模型族分别记录。小数据、浅树、窄特征和极端稀疏有效 split 可能不适合
GPU，必须量化退化区间并写入公开文档。任何“更快”声明都必须包含端到端时间，而不是
只展示单个 kernel。

### 4.5 安装与体积

v2 仍保持轻依赖和预编译 wheel 原则。不得为了覆盖更多模型族引入重量级运行时依赖。
新增模型不能把测试数据、benchmark、规格、缓存或开发工具链打进 wheel。

## 5. 路线

### V2-A：统一树后端抽象

- 抽象目标函数、采样策略、split 策略、树生长策略和预测聚合；
- 保持现有回归 GBDT 行为不变；
- 增加模型族能力矩阵和明确错误信息；
- 测试证明新增抽象没有复制数学公式。

### V2-B：分类与更完整 GBDT

- 实现二分类 logistic objective；
- 增加 `predict_proba`；
- 实现 early stopping、验证集监控和训练诊断；
- 保持 MPS 热路径真实执行。

### V2-C：Random Forest

- 实现 bootstrap/无放回采样；
- 实现特征子采样；
- 支持回归平均和分类投票；
- 多树训练按 batch 调度到 MPS，避免一棵树一个 command 的低效路径。

### V2-D：ExtraTrees

- 实现随机 threshold 候选生成；
- 支持可复现随机种子；
- GPU 批量评估随机 split；
- 与 Random Forest 共享采样、聚合和预测格式。

### V2-E：LightGBM-like 训练策略

- 支持 leaf-wise 生长的受控版本；
- 支持 histogram subtraction、活跃叶队列和小叶调度策略；
- 明确内存上限和过拟合控制；
- 对 level-wise 与 leaf-wise 进行端到端 benchmark。

### V2-E2：CatBoost-like ordered boosting

- 支持 ordered boosting 语义的受控版本；
- 支持类别特征友好训练路径和可复现 permutation 策略；
- 与统一分箱、缺失值、类别特征和模型 I/O 语义共享底座；
- 明确不同于任何第三方实现的兼容边界。

### V2-F：推理热路径

- 实现 MPS 批量 tree traversal 或等价优化；
- 支持多模型族共享预测 kernel；
- 验证加载模型后的预测与训练后一致；
- 记录 CPU/MPS 推理适用边界。

### V2-G：工业语义补全

- 实现缺失值默认方向；
- 实现样本权重；
- 实现类别特征 split；
- 实现单调约束和交互约束；
- 实现非平方误差回归目标；
- 实现基础解释能力。

### V2-H：异常检测与排序

- 实现 Isolation Forest；
- 实现 ranking objective 和 group/query 输入契约；
- 实现排序指标与验证集监控；
- 明确这些模型的 MPS 适用边界。

## 6. 验收门

- 每个模型族都有 CPU oracle、真实 MPS 测试、集成测试和模型 I/O 测试；
- 每个模型族都有至少两个预登记 benchmark 场景和退化区间；
- 公共 README 只描述已经完成的模型族，不提前承诺；
- wheel 安装仍然简单，体积和依赖审计通过；
- 删除缓存不影响任何模型结果；
- 所有任务在 `tasks.md` 逐项完成后才能打勾。
