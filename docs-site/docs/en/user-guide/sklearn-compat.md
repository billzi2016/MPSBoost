# sklearn Compatibility

MPSBoost estimators follow the sklearn-style interface:

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

The import-time MPS environment check does not use interactive input. CPU-only workers can skip
the environment check with:

```bash
MPSBOOST_SKIP_ENV_CHECK=1 python your_script.py
```
