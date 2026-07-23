# MPSBoost 1.0.0 性能报告

本报告记录发布机器上可用的最终 1.0.0 性能与体积证据，并区分 current-HEAD measurement 和 v2/v3 前历史 benchmark baseline。

## 发布机器

- 日期：2026-07-23
- 平台：Apple Silicon macOS
- 本地审计 Python：Python 3.13
- Dataset cache policy：项目内已忽略 `tests/real_world/data/` 和 `tests/real_world/cache/`
- 本地已检查 wheel artifact：
  `dist/mpsboost-1.0.0-cp310-cp310-macosx_13_0_arm64.whl` 为 280K，
  `dist/mpsboost-1.0.0-cp313-cp313-macosx_13_0_arm64.whl` 为 284K

## Current-HEAD CPU 审计

这些 measurement 使用最终 `1.0.0` wheel rebuild 前的 current source。Peak RSS 从进程最大 resident set size 换算，并四舍五入为 MiB。

| Dataset | Rows | Features | Estimator | Train (s) | Predict (s) | Metric | Score | Model size |
| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| Iris | 150 | 4 | GradientBoostingClassifier | 0.000965 | 0.000044 | accuracy | 0.973684 | 9,460 B |
| Breast Cancer | 569 | 30 | GradientBoostingClassifier | 0.006503 | 0.000106 | accuracy | 0.930070 | 11,048 B |
| Diabetes | 442 | 10 | GradientBoostingRegressor | 0.003520 | 0.000048 | R2 | 0.424435 | 11,156 B |
| Digits | 1,797 | 64 | GradientBoostingClassifier | 0.106488 | 0.001003 | accuracy | 0.740000 | 49,528 B |
| California Housing | 20,640 | 8 | GradientBoostingRegressor | 0.079778 | 0.003057 | R2 | 0.726251 | 33,000 B |
| Covertype subset | 30,000 | 54 | GradientBoostingClassifier | 0.423550 | 0.006479 | accuracy | 0.473067 | 19,604 B |

本次审计期间 peak RSS 低于 495 MiB。最大值来自加载和切分 cached Covertype subset 后的运行。

## 历史 MPS Baseline

已提交 S4/S6 结果仍可用于 workload-shape guidance，但不是最终 1.0.0 speed claim。它们记录于后续 v2/v3 实现和 cleanup 之前，作用是展示稳定边界：小型或同步开销重的任务在 MPS 上可能更慢；更宽、更大的 histogram workload 可以受益于 Apple GPU。

历史 S6 large-wide regressor case 在记录的 M2 Ultra 机器上达到相对 CPU oracle 的 1.629x median speedup。

## HIGGS 边界

HIGGS 明确作为 local-file、opt-in performance-boundary dataset。官方 raw dataset 是 multi-gigabyte scale，不会自动下载到用户环境，也不会打包进 wheel。可执行测试保留在 `tests/real_world/test_external_opt_in_datasets.py`，并说明运行它所需的本地文件路径。

## 1.0.0 策略

- 默认使用 `device="auto"`。
- CPU 是 correctness oracle，也是小型、branch-heavy 和 latency-sensitive workload 的优先路径。
- MPS 是 acceleration backend，适合足够大的 workload 来摊薄 launch、transfer 和 synchronization overhead。
- Linux CPU/CUDA 性能属于所选 external sklearn/XGBoost/CUDA stack，并按该 external backend 报告。
