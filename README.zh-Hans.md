# MPSBoost

MPSBoost 是面向 Apple Silicon 的树模型学习库。`0.3.0` 是 v2 arboretum 里程碑，包含 native CPU/MPS 后端、sklearn 风格 estimator、native CPU 多分类 softmax、随机森林、ExtraTrees、DecisionTree、CatBoost-like numeric estimator、高级回归目标、解释工具、异常检测和排序学习入口。

文档站点：[billzi2016.github.io/MPSBoost](https://billzi2016.github.io/MPSBoost/)。

## 安装

```bash
python -m pip install mpsboost
```

Apple GPU 加速需要 Apple Silicon、Metal 和可用本机环境。CPU 训练可直接使用。

## 环境诊断

如果 MPS 环境不可用，按提示执行：

```bash
xcode-select --install
xcodebuild -downloadComponent MetalToolchain
python -m pip install --upgrade --force-reinstall mpsboost
python -c "import mpsboost as mb; print(mb.system_info())"
```

CPU-only、CI 或 `GridSearchCV` worker 可跳过导入期环境检查：

```bash
MPSBOOST_SKIP_ENV_CHECK=1 python your_script.py
```

## 最小示例

```python
import numpy as np
import mpsboost as mb

X = np.arange(24, dtype=np.float32).reshape(12, 2)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

model = mb.GradientBoostingClassifier(
    n_estimators=3,
    max_depth=1,
    min_samples_leaf=1,
    min_child_weight=0.0,
    device="auto",
)
model.fit(X, y)
print(model.predict_proba(X).shape)
print(model.training_summary_)
```

## 公开 estimator

- `GradientBoostingRegressor`
- `GradientBoostingClassifier`
- `MPSBoostRegressor`
- `MPSBoostClassifier`
- `DecisionTreeRegressor`
- `DecisionTreeClassifier`
- `RandomForestRegressor`
- `RandomForestClassifier`
- `ExtraTreesRegressor`
- `ExtraTreesClassifier`
- `ExtraTreeRegressor`
- `ExtraTreeClassifier`
- `CatBoostRegressor`
- `CatBoostClassifier`
- `IsolationForest`
- `MPSIsolationForest`
- `LearningToRankRegressor`

## 后端策略

CPU backend 是项目内 native correctness oracle，不是临时替代品。MPS backend 用于适合 Apple GPU 的 histogram 热路径。`IsolationForest` 和 `LearningToRankRegressor` 属于 CPU-suitable workflow，请求 `device="mps"` 时会提示 CPU 更适合并继续运行。

S22 portable backend 是后续规划，不替代当前 native CPU/MPS 后端；运行摘要必须明确报告实际使用的 backend。
