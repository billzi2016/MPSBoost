# MPSBoost 0.5.0 性能报告

本报告总结 0.x 加固线的性能证据。口径刻意保守：已提交的 S4/S6 数字是 v2/v3 优化前的历史 baseline；`0.5.0` 重点是客户 workflow 可运行、后端选择透明、性能边界有文档记录。

## 已有证据

- Apple M2 Ultra、Python 3.13.5、MPSBoost `0.2.0a0` 上的 S4 histogram benchmark。
- Apple M2 Ultra、Python 3.13.5、后续 v2/v3 cleanup 前的 S6 end-to-end regressor benchmark。
- built-in、cached 和 opt-in 数据集的真实世界验收报告。
- hosted CPU runner 和 self-hosted real Metal GPU runner 上的 CI package tests。

## 历史 MPS 边界

已提交 S6 report 显示预期形态：

| Scenario | Rows | Features | CPU median (s) | MPS median (s) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `gbdt-medium` | 16,384 | 32 | 0.067630 | 0.089597 | 0.755x |
| `gbdt-wide` | 16,384 | 128 | 0.283176 | 0.249502 | 1.135x |
| `gbdt-large-wide` | 32,768 | 256 | 1.031002 | 0.633006 | 1.629x |

这些不是最终 `0.5.0` speed claim。它们用于保留 workload boundary：小型或同步开销重的任务在 GPU 上可能更慢；更宽、更大的 histogram workload 可以受益于 MPS。

## 当前 0.5.0 策略

- 普通用户使用 `device="auto"`。
- CPU 是 correctness oracle，也是小数据、branch-heavy anomaly detection 和 latency-sensitive ranking 的优先路径。
- MPS 是 acceleration backend，适合足够大的 workload，用来摊薄 launch、transfer 和 synchronization overhead。
- 显式 external portable policy 会记录 requested/effective backend。如果 external runtime 没有对当前 estimator 激活，MPSBoost 会 warning，并通过 native CPU compatibility 继续运行，而不是中断 workflow。
- Linux CPU/CUDA 性能取决于所选 external sklearn/XGBoost/CUDA stack，并报告为 external backend，不报告成 native MPSBoost CPU/MPS。

## 1.0.0 之前

最终稳定版本需要在 current HEAD 上重新完成完整性能审计：

- built-in、cached 和 opt-in 真实世界数据集上的 train time 和 predict time；
- peak memory；
- model size；
- wheel size；
- CPU/MPS applicability boundary；
- Linux CPU/CUDA 路径的 external backend attribution；
- 原始命令、机器信息、package version 和 artifact hash。
