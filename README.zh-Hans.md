# MPSBoost

<p align="center">
  <img src="assets/icons/mpsboost-icon.png" alt="MPSBoost 图标" width="160">
</p>

[![PyPI](https://img.shields.io/pypi/v/mpsboost)](https://pypi.org/project/mpsboost/)
[![Python](https://img.shields.io/pypi/pyversions/mpsboost)](https://pypi.org/project/mpsboost/)
[![README](https://img.shields.io/badge/README-English%20%7C%20%E4%B8%AD%E6%96%87-blue)](#mpsboost)

MPSBoost 是一个面向 Apple Silicon 的早期梯度提升项目。当前加速后端使用自定义 Metal compute kernel 计算 squared-error gradient 和两阶段 histogram，同时保留一个与 CPU oracle 共享的确定性树构建实现。

文档站点位于 [billzi2016.github.io/MPSBoost](https://billzi2016.github.io/MPSBoost/)。

> **开发状态：** `1.0.0` 是 stable customer-commitment release。它包含 0.x line 加固后的 native CPU/MPS tree-estimator 范围：sklearn 风格 classifier、native CPU multiclass softmax、random forest、ExtraTrees、decision tree、CatBoost-like numeric estimator、高级回归目标、特征解释、适合 CPU 的 anomaly/ranking estimator、明确的环境提示、有文档记录的后端选择边界、可选 SHAP/portable-backend 诊断、S18 真实世界验收报告、性能与 known-issue 报告，以及 external backend policy 的用户侧 fallback 行为。

## 项目来源

MPSBoost 由一名 Purdue CS 博士生发起，工作方向横跨算法、系统、AI、编译器和形式化验证。直接动机很实际：Apple Silicon 拥有很强的 GPU 栈，但常见树模型机器学习工作负载仍缺少简单、快速、低权限的 MPS 加速路径。

项目采用 SDD 工作流。工作被拆成清晰阶段：先验证真实 MPS/Metal kernel 和运行时行为，再锁定规格和产品需求，然后确定技术栈，最后按任务清单执行。规格被视为项目契约，而不是事后补写的说明。

对 AI agent 和自动化而言，[mps_boost_skill.md](https://github.com/billzi2016/MPSBoost/blob/main/ai-skills/mps_boost_skill.md) 是规范用法入口。它描述完整目标用法、导入风格、estimator 名称、后端策略、sklearn model selection、模型持久化、诊断和实现约束。

## 安装

```bash
python -m pip install mpsboost
```

加速版本提供预构建 Apple Silicon wheel；普通用户不需要 heavyweight framework、包管理器、CMake 或本地 Metal shader compiler。

## Estimator 风格 API

```python
import mpsboost as mb

model = mb.GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    device="mps",
)

model.fit(X_train, y_train)
prediction = model.predict(X_test)
importance = model.feature_importances_
permutation = model.permutation_importance(X_test, y_test)
model.save_model("model.mb")

restored = mb.GradientBoostingRegressor(device="mps")
restored.load_model("model.mb")
```

`MPSBoostRegressor` 继续作为向后兼容的项目品牌别名，指向同一实现。

共享 CPU/MPS 训练路径支持 `fit(..., sample_weight=...)` 和 `score(..., sample_weight=...)`。权重会进入 native gradient、Hessian、split gain、leaf value 和默认 estimator score，而不是作为 Python-only 后处理。

## sklearn model selection

MPSBoost estimator 设计为遵循 sklearn estimator protocol，所以用户应能使用标准 sklearn model-selection 栈，而不需要学习项目专用搜索 API。

```python
from sklearn.model_selection import GridSearchCV
import mpsboost as mb

search = GridSearchCV(
    mb.GradientBoostingRegressor(device="auto"),
    param_grid={
        "max_depth": [3, 6, 9],
        "learning_rate": [0.03, 0.1],
        "n_estimators": [100, 300],
    },
    cv=3,
    n_jobs=2,
)

search.fit(X_train, y_train)
best_model = search.best_estimator_
```

同一方向适用于 `RandomizedSearchCV`、`cross_val_score` 和 classifier estimator。当前 regressor 与 classifier 暴露标准搜索工具需要的 `get_params()`、`set_params()`、`fit()`、`predict()`、适用时的 `predict_proba()`，以及默认 `score()`。真实世界 Iris 验收包含标准 `GridSearchCV` 多分类运行。

多进程通过 sklearn/joblib 在外层搜索级别支持。CPU job 可以在多个进程中运行。MPS job 应更保守调度：多个 Python worker 争用同一个 Apple GPU 时，command queue、统一内存带宽和同步开销会成为瓶颈，可能比一个尺寸合适的 GPU worker 更慢。预期策略是让 `device="auto"` 为小型搜索任务选择 CPU，只在测量到 tree hot path 足够大时保留 MPS。

## Tree estimator 名称

主要 public API 使用简洁的 sklearn 风格 estimator 名称，因此用户通常只需替换 import，保留熟悉的模型名称。

| Model family | Primary names | Status |
| --- | --- | --- |
| Histogram gradient boosting | `GradientBoostingRegressor` | Available |
| Histogram gradient boosting classification | `GradientBoostingClassifier` | Available for binary labels and multiclass one-vs-rest |
| Random forest | `RandomForestRegressor`, `RandomForestClassifier` | Available |
| Extra trees | `ExtraTreesRegressor`, `ExtraTreesClassifier` | Available |
| Single decision tree | `DecisionTreeRegressor`, `DecisionTreeClassifier` | Available |
| CatBoost-like ordered boosting | `CatBoostRegressor`, `CatBoostClassifier` | Available for numeric features |
| Isolation forest | `IsolationForest`, `MPSIsolationForest` | Available; CPU is selected for this branch-heavy workload because it is expected to be faster than Apple GPU |
| Ranking trees | `LearningToRankRegressor` | Available; CPU is selected for this latency-sensitive workflow because it is expected to be faster than Apple GPU |

quantile、Poisson、Tweedie、logistic、ranking 等 objective variant 在共享同一 tree engine 时应通过 estimator 参数选择。只有当模型族具有不同 fit/predict 语义时，才添加单独类名。

Python 中也能查询同一状态信息：

```python
import mpsboost as mb

print(mb.available_estimators())
print(mb.planned_estimators())
mb.require_estimator_supported("CatBoostRegressor")
```

`1.0.0` 发布交付 v2 arboretum 基础并补上显式可选 SHAP、portable-backend 诊断和用户侧 fallback 行为：一个共享 tree-family registry、boosting/bagging/random-split 模型的统一语义、诚实的后端策略，以及没有 placeholder estimator class。

Random forest 行采样、特征子采样、ExtraTrees 随机 threshold 和 CatBoost-like ordered permutation 共享一个确定性随机化契约。类别特征可以用 `categorical_features=[...]` 标记；CatBoost-like estimator 也接受 `cat_features=[...]` 别名。类别会按训练目标统计排序，然后进入同一 native histogram split engine。预测时未知类别会编码为 missing value，并使用 native missing-value default direction。类别模型持久化会有意失败，直到公开模型格式保存 category mapping。

验证指标历史和 early stopping 也共享 estimator-independent monitoring 契约，未来 classifier 和 tree ensemble 不需要重复 callback 语义。

LightGBM-like 受控 leaf-wise growth 可通过 `growth_strategy="leaf_wise"` 使用。实现使用与 level-wise boosting 相同的 native tree engine，并增加 `max_leaves`、`max_active_leaves` 和 `min_gain_to_split` 显式控制。Benchmark 脚本会端到端比较 level-wise 与 leaf-wise；用户应选择自己工作负载上实际胜出的策略，而不是假设某种 growth policy 永远更快。

## CPU 和 MPS 后端

MPSBoost 将 CPU 视为 first-class backend，而不是临时 fallback。CPU oracle/backend 在项目内实现，并与 MPS backend 共享 quantization、objective、sampling、monitoring、split-gain 和 model-format 契约。MPSBoost 不会调用 XGBoost、LightGBM、CatBoost 或 scikit-learn 作为隐藏训练引擎。

长期策略如下：

- `device="cpu"` 强制项目内 CPU backend。
- `device="mps"` 强制项目内 MPS/Metal backend。
- `device="auto"` 为小型或同步开销重的工作负载选择 CPU，为测得 tree hot path 能覆盖传输和 launch 开销的工作负载选择 MPS。

MPS 是加速后端，不是运行要求。如果某个工作负载 CPU 更快或更稳定，项目应明确说明并在 `auto` 下使用 CPU。

## 预测路径

所有已交付 tree family 共享一个预测契约。Gradient boosting、single tree、random forest、ExtraTrees 和 CatBoost-like numeric estimator 都复用 native flat-tree model format，以及共享的 Python aggregation helper，用于 feature-count validation、feature-subset slicing 和 forest averaging。

当前 MPS 加速目标是训练 hot path。预测使用共享 native tree traversal path，小 batch 上可能 CPU 更快。未来 MPS batch traversal kernel 可以替换 traversal backend，而不改变 estimator API 或模型文件。

## 后端诊断

native backend 暴露非敏感设备和缓存诊断：

```python
import mpsboost as mb

print(mb.__version__)
print(mb.is_available())
print(mb.system_info())
print(mb.mps_setup_instructions())
print(mb.cache_info())
```

如果用户在 Apple GPU 加速不可用的机器或 session 上请求 `device="mps"`，MPSBoost 会报告可复制粘贴的 setup 命令：

```bash
xcode-select --install
xcodebuild -downloadComponent MetalToolchain
python -m pip install --upgrade --force-reinstall mpsboost
python -c "import mpsboost as mb; print(mb.system_info())"
```

CPU-only 脚本、托管 CI 和多进程 model-selection worker 可以跳过导入期 GPU 环境检查，同时不禁用 CPU 训练：

```bash
MPSBOOST_SKIP_ENV_CHECK=1 python your_script.py
```

`cache_info()` 只报告路径和存在性，不创建目录。`create_cache()` 显式创建 L2 cache 目录，`clear_cache()` 在拒绝危险目标如文件系统根目录、用户 home 目录或 symlink 后安全删除 MPSBoost cache root。删除 cache 永远不改变模型结果。

可选科研和 portable-backend 依赖默认不安装：

```bash
python -m pip install 'mpsboost[shap]'
python -m pip install 'mpsboost[xgboost]'
python -m pip install 'mpsboost[sklearn]'
python -m pip install 'mpsboost[cuda]'
```

`mpsboost[shap]` 保留给官方 SHAP 集成路径。当前 `approximate_shap_values(...)` 输出是受控近似解释，不得引用为官方 SHAP TreeExplainer 输出。公共 helper `export_native_trees_for_shap(...)` 会导出 native tree structure 用于 adapter validation，并且不包含训练数据、telemetry、credential 或 device identifier。

Portable backend policy 是显式且可观测的。Native CPU/MPS 继续作为 MPSBoost 默认实现和 correctness oracle。后续 sklearn/XGBoost/CUDA adapter 必须通过 portable policy 选择，并在 summary 中报告实际 backend；它们不会隐藏在 native MPSBoost backend 名称后面。

## 项目原则

- 熟悉的 XGBoost/scikit-learn 风格 Python 入口。
- `device="mps"` 作为稳定的用户侧 Apple GPU 后端名称。
- 面向树模型不规则计算的自定义 Metal kernel。
- 预构建 wheel，且没有 heavyweight Python runtime dependency。
- 可观测后端策略：非法输入清晰失败；适合 CPU 的 MPS 请求会 warning，并在 training summary 中记录后端选择。
- 端到端 benchmark，包括 preprocessing 和 synchronization。

## 状态

当前 public API 包含 `GradientBoostingRegressor`、`GradientBoostingClassifier`、它们向后兼容的项目品牌别名、estimator capability registry、确定性 randomization 与 monitoring helper、cache diagnostics 与 management helper、`is_available`、`system_info` 和 `__version__`。训练支持 dense finite `float32`/`float64` compatible data、ordered categorical feature encoding、feature-level monotonic constraints、path-level interaction constraints、L1/L2/gamma regularization、leaf-value clipping、squared error regression、binary-logistic classification、multiclass one-vs-rest classification、quantile、Poisson、Tweedie、isolation-forest anomaly scoring、pointwise ranking、deterministic quantization、depth-limited histogram trees、sklearn-compatible `score()`、numeric non-categorical binary/native model save/load、gain/split/permutation feature importance、approximate SHAP-like explanations、random forest `n_jobs`、显式 `device="mps"`、显式 `device="cpu"` 和初始 `device="auto"` selection。

已提交的 S6 benchmark 记录了回归区间和胜出区间。在 M2 Ultra 验证机器上，小型端到端训练在 MPS 上仍然更慢，而 `gbdt-large-wide` 场景相对 CPU oracle 达到 1.629x median speedup，最大预测差异约 `5.4e-6`。

Sparse matrix、categorical model persistence、public GPU prediction 和完整 third-party API compatibility 不属于本里程碑。小数据集预计在 GPU 上更慢，因为固定设备 setup 和同步成本占主导；已提交 benchmark report 会保留这个退化区域，同时记录更大工作负载上的胜出结果。

## 发布审计

`1.0.0` release gate 包含：

- Python 3.10 和 3.13 上的 CPU、packaging、integration 和真实 Metal GPU 测试。
- Wheel 内容检查，排除 specs、tests、caches 和 build artifacts。
- Native extension 的 dynamic-link 检查。
- Apache-2.0 项目许可证和 runtime dependency review。
- Fresh PyPI installation 和真实 MPS smoke validation。

## 独立性声明

MPSBoost 是独立开源项目。它不隶属于 Apple Inc.、XGBoost project、LightGBM project、CatBoost project 或 scikit-learn project，也未获得这些项目认可或赞助。MPS/Metal backend 是基于公开论文、公开 API 文档和原创工程工作的独立实现，不派生自这些库。Apple、Metal 和 Metal Performance Shaders 可能是 Apple Inc. 的商标。

## License

Apache-2.0
