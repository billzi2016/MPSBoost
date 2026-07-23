# MPSBoost 0.4.0 发布审计

本文件记录 `0.4.0` 的发布门：这是完成大规模真实世界验收之后的 0.x 版本。它不改写
`0.3.0` 审计；`0.3.0` 仍然是 all-trees 功能里程碑，位于 S18 大规模验证之前。

## 范围

`0.4.0` 包含 `0.3.0` all-trees 范围，并额外包含：

- 可选官方 SHAP 集成诊断和 native tree export payload；
- 显式 portable-backend 诊断，以及 XGBoost、sklearn、CUDA 可选 extras；
- MNIST、Titanic、Adult Income、Covertype、HIGGS 的项目内 opt-in 真实世界数据 cache；
- S18 真实世界验收报告，记录成功、退化和边界；
- 文档明确 `1.0.0` 仍要等待最终稳定发布门关闭。

## 边界

Native CPU/MPS 仍然是 MPSBoost 实现和 correctness oracle。Linux CPU 与 Linux CUDA 执行通过显式 external backend policy 和 adapter 暴露；外部 sklearn/XGBoost/CUDA runtime 内部失败，不写成 native MPSBoost backend 失败。

`1.0.0` 仍然被阻塞，直到模型质量、端到端性能、内存峰值、wheel/model 体积、权限、artifact hash 和用户明确最终确认全部完成。

## 验证矩阵

发布 `0.4.0` 前必须完成：

- 本地 non-GPU unit、integration、packaging 和真实世界验收测试；
- 基于项目内 cache 的 opt-in 外部真实世界数据集验收；
- Python 3.10 和 3.13 上的 GitHub hosted CPU/package test；
- Python 3.10 和 3.13 上的 self-hosted real Metal GPU test；
- exact uploaded wheels 的 `twine check`；
- fresh wheel install 和 import/training smoke verification。

## Artifact 规则

Release wheel 不得包含 `specs/`、`tests/`、`benchmarks/`、`.github/`、build directory、cache directory、raw dataset、credential 或 runner file。
