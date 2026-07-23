# 真实世界验收报告

本报告记录当前 S18 release-gate 证据，并保持诚实：默认 CI 不联网，opt-in 外部数据集只从已忽略的本地 cache 运行；在完整矩阵、性能审计和用户公开承诺完成前，`1.0.0` 仍然阻塞。

## 默认 no-network 矩阵

- Iris：active multiclass native CPU softmax acceptance。
- Digits：active flattened-image multiclass native CPU softmax acceptance。
- Breast Cancer Wisconsin：active binary classification acceptance，并记录 sklearn CPU baseline。
- Diabetes：active regression acceptance、advanced objective acceptance、ranking smoke 和 sklearn CPU baseline。

## Cached opt-in 矩阵

- California Housing：通过 sklearn fetcher 的 active cached regression acceptance。
- Covertype subset：active cached large-row multiclass acceptance 和 real MPS parity smoke。
- MNIST subset：active cached OpenML flattened-image multiclass acceptance。
- Titanic：active cached OpenML missing-value 和 categorical workflow acceptance。
- Adult Income：active cached OpenML categorical binary-classification acceptance。
- HIGGS subset：active local-file large numeric binary-classification acceptance。

## 退化和边界说明

- 小数据集可能 CPU 更快，因为 GPU launch 和 synchronization overhead 占主导。
- IsolationForest 和 LearningToRankRegressor 是 CPU-suitable workflow；MPS 请求会 warning，并记录 CPU backend decision。
- MPS native multiclass softmax 仍是分阶段能力。CPU native softmax 是当前默认 native 多分类实现；MPS 多分类在 native MPS softmax kernel 完成前使用可观测 OvR compatibility。
- 普通测试不会下载外部数据集。缺 cache 时测试会 skip，并给出可复制 setup 指令。

## 未关闭 release gate

- S18.6/S18.6a：完整 model quality、end-to-end performance、peak memory、wheel size、model size 和 permission audit。
- S18.8：规划 `1.0.0` 前必须完成 G17 全矩阵验收。
- S18.9/S18.10：用户确认和正式 PyPI `1.0.0` 发布。
