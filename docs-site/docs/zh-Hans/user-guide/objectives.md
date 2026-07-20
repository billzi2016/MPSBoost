# 目标函数

MPSBoost 当前支持以下目标函数：

- squared error
- binary logistic
- native CPU multiclass softmax
- quantile
- Poisson
- Tweedie

## 参数入口

回归 estimator 通过 `loss` 选择高级目标：

```python
model = mb.GradientBoostingRegressor(loss="quantile", quantile_alpha=0.9)
model = mb.GradientBoostingRegressor(loss="poisson")
model = mb.GradientBoostingRegressor(loss="tweedie", tweedie_variance_power=1.5)
```

## 模型持久化

高级目标参数写入 native model payload。加载模型时会校验 estimator 参数和模型 objective 是否兼容，避免误用错误目标函数解释预测值。
