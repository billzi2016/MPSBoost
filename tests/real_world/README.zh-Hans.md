# 真实世界测试套件

本目录保留给真实世界数据集验收测试。

该套件用于在任何 `1.x` release 前验证 MPSBoost 能在实际数据集和工作流上运行。它独立于 unit test、integration test 和 synthetic benchmark。

Dataset matrix：

- `dataset_matrix.py` 是可执行 S18 dataset matrix。
- `REPORT.zh-Hans.md` 记录当前 S18 验收证据和仍未关闭的发布门。
- 默认 no-network acceptance 目前只运行 active built-in datasets。
- Multiclass dataset 默认使用 native CPU softmax。MPS multiclass 当前使用显式 staged OvR compatibility path，直到 native MPS softmax kernel 实现。

初始 dataset target：

- Iris：小型 multiclass sanity test。
- Breast Cancer Wisconsin：binary classification baseline。
- Diabetes：小型 regression sanity test。
- Diabetes advanced objectives：quantile、Poisson 和 Tweedie 在 positive real targets 上通过共享 native CPU trainer 运行。这些检查验证正确性和有限输出，不验证 objective leaderboard quality。
- Breast Cancer isolation forest：适合 CPU 的 anomaly scoring，并输出有限 path-length score。
- Diabetes pointwise ranking：query-group validation 和 finite full-list NDCG。
- California Housing：medium regression baseline。
- Digits：轻量 flattened-image multiclass test。
- MNIST subset：opt-in flattened-image stress test。
- Titanic：missing-value 和 categorical-feature workflow test。
- Adult Income：更大的 categorical binary-classification test。
- Covertype subset：更大的 multiclass throughput test。
- Higgs subset：opt-in large numeric binary-classification performance-boundary test。

规则：

- 不要在本目录 mock CPU 或 MPS backend。
- 不要把 raw external dataset 提交进 repository 或 wheel。
- 默认 CI 覆盖优先使用 built-in sklearn datasets。
- External dataset 必须 versioned、hash-checked、cached，并且可离线复现。
- Long-running test 必须 opt-in 并清晰标记。
- Advanced objective 当前共享 histogram tree path，并使用适中的 CPU acceptance 设置；release notes 宣称 speedup 前需要单独 end-to-end benchmark。
- Isolation forest 和 pointwise ranking 会把 MPS 请求路由到 CPU 并发出 warning，因为这些 branch-heavy 或 latency-sensitive workflow 预计在 CPU 上比 Apple GPU 更快。
- Blocked dataset 必须在 matrix 中保持可见，不得被 synthetic 或 binary-subset stand-in 静默替代。

显式下载：

```bash
python tests/real_world/download_datasets.py california-housing
python tests/real_world/download_datasets.py covertype-subset
python tests/real_world/download_datasets.py mnist-subset
python tests/real_world/download_datasets.py titanic
python tests/real_world/download_datasets.py adult-income
```

下载文件位于已 ignore 的 `tests/real_world/data/` 下，生成的 manifest 位于已 ignore 的 `tests/real_world/cache/` 下。删除这两个目录即可删除本地数据集产物；下载器不使用用户全局 sklearn cache path。

HIGGS 作为显式本地文件数据集处理。运行 opt-in HIGGS subset 测试前，将 `HIGGS.csv.gz`
放到 `tests/real_world/data/higgs/`。
