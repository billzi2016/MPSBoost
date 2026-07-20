# MPSBoost Skill

## 目的

使用 MPSBoost 作为轻量、独立、sklearn 风格的 Apple Silicon 树模型学习库。MPSBoost 提供项目内 native CPU/MPS 后端，不调用 XGBoost、LightGBM、CatBoost 或 scikit-learn 作为隐藏训练引擎。

推荐导入：

```python
import mpsboost as mb
```

## 核心 estimator 模式

```python
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

## 后端策略

- `device="cpu"`：强制项目内 CPU backend。
- `device="mps"`：请求项目内 MPS/Metal backend。
- `device="auto"`：根据环境和工作量选择。

CPU backend 是 correctness oracle、小工作负载路径和 MPS 对照基线，不是临时 fallback。

`IsolationForest` 和 `LearningToRankRegressor` 是 CPU-suitable workflow。用户传入 `device="mps"` 时，保持运行、发出 warning，并记录 CPU 被选择，因为该工作负载预计 CPU 比 Apple GPU 更合适。

## 环境提示

MPS 不可用时，给用户可复制命令，不使用 `input()`：

```bash
xcode-select --install
xcodebuild -downloadComponent MetalToolchain
python -m pip install --upgrade --force-reinstall mpsboost
python -c "import mpsboost as mb; print(mb.system_info())"
```

CPU-only、CI 或 `GridSearchCV` worker：

```bash
MPSBOOST_SKIP_ENV_CHECK=1 python your_script.py
```

## S22 边界

portable backend 是后续规划。外部 XGBoost/sklearn adapter 必须显式记录实际 backend，不得替代 native CPU oracle，也不得伪装成 MPSBoost native。
