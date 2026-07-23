# MPSBoost 0.2.0 发布审计

`0.2.0` 是第一个正式 MPS histogram engine 版本。它建立了真实 compiled Apple Silicon backend、deterministic data/binning path、regression GBDT、model save/load、cache safety 和 release verification workflow。

包含：

- 真实 compiled MPS/Metal backend foundation；
- native device diagnostics 和 GPU smoke kernel；
- deterministic quantization 和 compact binned representation；
- 真实 regression GBDT training 和 prediction；
- model save/load；
- S6 benchmark evidence，记录 MPS 胜出和 small-data regression；
- cache diagnostics、explicit cache creation 和 safe cache clearing；
- wheel、license、dynamic-link 和 PyPI verification。

不包含：

- classifier；
- forest/ExtraTrees/CatBoost-like family；
- advanced objective；
- anomaly/ranking estimator；
- 大规模真实世界数据集矩阵。
