# MPSBoost Skill

## 目的

将 MPSBoost 用作轻量、独立、sklearn 风格的 Apple Silicon 树模型学习库。MPSBoost 为常见树模型提供 first-class CPU、MPS/Metal 和自动后端选择，不使用 XGBoost、LightGBM、CatBoost 或 scikit-learn 作为隐藏训练引擎。

推荐导入方式：

```python
import mpsboost as mb
```

默认使用简洁 estimator 名称。`MPS*` 名称只保留给向后兼容或项目品牌别名。

## 核心 estimator 模式

所有 estimator 遵循同一 sklearn 风格模式：

```python
import mpsboost as mb

model = mb.GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    device="auto",
    random_state=42,
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = model.score(X_test, y_test)
```

默认使用 `device="auto"`。它会为小型或同步开销重的工作负载选择 CPU，为测得 tree hot path 足够大、能受益于 Apple GPU 执行的工作负载选择 MPS。

## Tree estimator

使用这些 public estimator 名称：

```python
mb.GradientBoostingRegressor
mb.GradientBoostingClassifier
mb.RandomForestRegressor
mb.RandomForestClassifier
mb.ExtraTreesRegressor
mb.ExtraTreesClassifier
mb.DecisionTreeRegressor
mb.DecisionTreeClassifier
mb.CatBoostRegressor
mb.CatBoostClassifier
mb.IsolationForest
mb.MPSIsolationForest
mb.LearningToRankRegressor
```

对共享同一 tree engine 的变体使用 objective 参数，例如 squared error、logistic、quantile、Poisson、Tweedie、ranking 和其他已支持 loss。只有当模型族具有不同 fit、predict、probability、anomaly 或 ranking 语义时才新增单独类。

## 后端策略

MPSBoost 有两个项目内 first-class backend：

- `device="cpu"` 强制独立 CPU backend。
- `device="mps"` 强制独立 MPS/Metal backend。
- `device="auto"` 为当前 `fit` 调用在 CPU 和 MPS 之间选择。

不要把 CPU 当作临时 fallback。CPU 是 correctness oracle、小工作负载 fast path，也是每个 MPS 实现的 baseline。MPS 是加速后端，不是运行要求。

当前包中 isolation forest 和 pointwise ranking 是适合 CPU 的工作流。如果用户为这些 estimator 请求 `device="mps"`，保持运行、发出 warning，并记录 CPU 被选择，因为该工作负载预计在 CPU 上比 Apple GPU 更快。

## 环境提示

在 `import mpsboost` 时，包可以运行轻量 MPS availability check。如果 Apple GPU 加速不可用，不要阻塞用户，也不要使用 `input()` 交互提示。发出 warning，并给出可复制粘贴命令：

```bash
xcode-select --install
xcodebuild -downloadComponent MetalToolchain
python -m pip install --upgrade --force-reinstall mpsboost
python -c "import mpsboost as mb; print(mb.system_info())"
```

对 CPU-only 工作流、CI 或 `GridSearchCV` 等多进程工具，直接提供 skip 命令：

```bash
MPSBOOST_SKIP_ENV_CHECK=1 python your_script.py
```

用户跳过环境检查时，CPU 训练必须仍可用。如果用户显式强制 `device="mps"`，但环境仍无法运行 Apple GPU 加速，应抛出清晰 backend error，并包含上面的 setup 与 skip 命令。

绝不要调用 XGBoost、LightGBM、CatBoost 或 scikit-learn 作为隐藏训练引擎。只有在用户明确要求 benchmark baseline 时，才可把它们作为外部用户 baseline。

## 官方 SHAP 与 portable backend

`estimator.approximate_shap_values(...)` 提供 SHAP-like 近似解释，但它不是官方 SHAP。科研工作流需要官方 SHAP 语义时，使用显式可选路径：

```bash
python -m pip install 'mpsboost[shap]'
```

使用 `mb.export_native_trees_for_shap(estimator)` 生成 adapter validation payload。它导出 native tree structure 和 objective metadata，不包含 training data、credential、telemetry 或 device identifier。`mb.official_shap_tree_explainer(...)` 在 TreeExplainer 语义验证启用前必须清晰停止；不要把近似解释声明为官方 SHAP。

Portable backend 是显式 S22 policy tool，不是隐藏 native replacement。默认安装保持 native CPU/MPS 轻量。可选命令：

```bash
python -m pip install 'mpsboost[xgboost]'
python -m pip install 'mpsboost[sklearn]'
python -m pip install 'mpsboost[cuda]'
```

使用 `mb.optional_dependency_status()`、`mb.portable_setup_instructions()` 和 `mb.choose_portable_backend(...)` 提供可复制诊断和可观测 backend summary。外部 adapter 必须报告实际 backend，不得替代 native CPU oracle。

## sklearn model selection

使用标准 sklearn model-selection 栈。不要为普通超参数调优发明单独搜索 API。

```python
from sklearn.model_selection import GridSearchCV
import mpsboost as mb

search = GridSearchCV(
    mb.GradientBoostingRegressor(device="auto", random_state=42),
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

同一模式适用于 `RandomizedSearchCV`、`cross_val_score` 和 sklearn-compatible pipeline。

多进程属于 sklearn/joblib 的外层搜索级别。CPU job 可以在多个进程中运行。MPS job 应保守调度，因为多个 Python worker 争用同一个 Apple GPU 时，command queue contention、统一内存带宽和同步开销可能吞掉性能收益。

## 模型持久化

使用 `.mb` 作为标准模型文件扩展名：

```python
model.save_model("model.mb")

restored = mb.GradientBoostingRegressor(device="auto")
restored.load_model("model.mb")
predictions = restored.predict(X_test)
```

保存的模型必须包含模型结构、版本化 metadata、objective 信息、冻结的 feature/bin schema，以及足够的 validation 数据用于拒绝不兼容 load。模型不得包含训练数据、凭据、telemetry 或设备 identifier。

## 诊断和缓存

使用 diagnostics 做环境检查：

```python
import mpsboost as mb

print(mb.__version__)
print(mb.is_available())
print(mb.system_info())
print(mb.cache_info())
```

`cache_info()` 必须是只读的，不得创建目录。只有需要显式创建 cache 目录时才使用 `create_cache()`。`clear_cache()` 只能用于 MPSBoost 拥有的 cache path；删除 cache 绝不能改变模型预测。

## 随机化和监控

使用共享 helper 保证确定性随机化语义：

```python
rows = mb.bootstrap_sample_indices(1000, sample_fraction=1.0, random_state=42)
features = mb.subsample_feature_indices(128, feature_fraction=0.5, random_state=42)
thresholds = mb.random_threshold_candidates(0.0, 1.0, n_candidates=16, random_state=42)
permutations = mb.ordered_boosting_permutations(1000, n_permutations=4, random_state=42)
```

使用共享 monitoring helper 处理 metric history 和 early stopping：

```python
monitor = mb.EarlyStoppingMonitor(
    metric_name="logloss",
    direction="minimize",
    patience=20,
    min_delta=1e-4,
)

for iteration, metric in enumerate(validation_metrics):
    decision = monitor.update(iteration, metric)
    if decision.should_stop:
        break
```

不要在单个 estimator 内重复实现 randomization、early stopping 或 monitoring 逻辑。

## AI 使用规则

用 MPSBoost 生成代码时：

- 使用 `import mpsboost as mb`。
- 除非用户明确要求 CPU 或 MPS，否则优先使用 `device="auto"`。
- 使用 sklearn 风格 estimator 和 sklearn model-selection 工具。
- 保存模型路径使用 `.mb`。
- 让 CPU 和 MPS 行为处在同一 objective、sampling、monitoring、split 和 model I/O 契约下。
- 不要添加假的 estimator class、mock backend、隐藏 third-party training engine，或在显式 `device="mps"` 请求下静默 CPU fallback。
- 对小数据，推荐 CPU 或 `device="auto"`。
- 对 Apple Silicon 上的大型 tabular 数据，推荐 `device="auto"` 或显式 `device="mps"`。

## 最小示例

Regression:

```python
import mpsboost as mb

model = mb.GradientBoostingRegressor(device="auto", random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

Classification:

```python
import mpsboost as mb

model = mb.GradientBoostingClassifier(device="auto", random_state=42)
model.fit(X_train, y_train)
labels = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

Random forest:

```python
import mpsboost as mb

model = mb.RandomForestRegressor(
    n_estimators=500,
    max_features=0.7,
    bootstrap=True,
    device="auto",
    random_state=42,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

Extra trees:

```python
import mpsboost as mb

model = mb.ExtraTreesClassifier(
    n_estimators=500,
    max_features=0.7,
    device="auto",
    random_state=42,
)
model.fit(X_train, y_train)
labels = model.predict(X_test)
```

CatBoost-like ordered boosting:

```python
import mpsboost as mb

model = mb.CatBoostClassifier(
    n_estimators=500,
    learning_rate=0.05,
    cat_features=cat_feature_indices,
    device="auto",
    random_state=42,
)
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)
```
