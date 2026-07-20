# 最小示例

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

`training_summary_` 会记录实际后端、策略和关键训练参数。跨平台 portable backend 属于 S22 后续规划，不替代当前 native CPU/MPS 后端。
