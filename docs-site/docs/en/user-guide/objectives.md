# Objectives

MPSBoost currently supports these objectives:

- squared error
- binary logistic
- native CPU multiclass softmax
- quantile
- Poisson
- Tweedie

## Parameter entry

Regression estimators choose advanced objectives through `loss`:

```python
model = mb.GradientBoostingRegressor(loss="quantile", quantile_alpha=0.9)
model = mb.GradientBoostingRegressor(loss="poisson")
model = mb.GradientBoostingRegressor(loss="tweedie", tweedie_variance_power=1.5)
```

## Model persistence

Advanced objective parameters are written into the native model payload. Model loading validates
that estimator parameters and model objectives are compatible, preventing predictions from being
interpreted under the wrong objective.
