# sklearn 兼容

MPSBoost estimator 遵循 sklearn 风格接口：

- `fit`
- `predict`
- `predict_proba`
- `decision_function`
- `score`
- `get_params`
- `set_params`

## Model selection

```python
from sklearn.model_selection import GridSearchCV
import mpsboost as mb

search = GridSearchCV(
    mb.GradientBoostingClassifier(device="auto"),
    param_grid={"max_depth": [2, 3], "n_estimators": [20, 50]},
    cv=3,
)
search.fit(X_train, y_train)
```

导入期 MPS 环境检查不会使用交互输入。CPU-only worker 可以通过以下命令跳过环境检查：

```bash
MPSBOOST_SKIP_ENV_CHECK=1 python your_script.py
```
