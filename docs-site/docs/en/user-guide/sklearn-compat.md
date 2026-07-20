# sklearn Compatibility

MPSBoost estimators support `fit`, `predict`, `predict_proba`, `decision_function`,
`score`, `get_params`, and `set_params`, including `GridSearchCV`. CPU-only workers
can skip the import-time environment check with:

```bash
MPSBOOST_SKIP_ENV_CHECK=1 python your_script.py
```
