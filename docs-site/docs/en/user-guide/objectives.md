# Objectives

MPSBoost supports squared error, binary logistic, native CPU multiclass softmax,
quantile, Poisson, and Tweedie objectives. Advanced regression objectives are chosen
with `loss`; their metadata is persisted in the native model payload and validated on load.
