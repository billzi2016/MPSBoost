# MPSBoost 0.2.0b0 发布审计

`0.2.0b0` 是 GPU hot-path beta。重点是 Metal histogram 和 split-processing pipeline，用于让大型 tabular workload 快于 CPU oracle。

包含：

- 真实 MPS gradient 和 histogram kernel；
- split-scan、partition/compaction、histogram subtraction 和 buffer-pool 工作；
- S6 benchmark evidence，记录胜出场景和 small-data regression；
- 持续 CPU oracle validation。

不包含：

- 最终 cache invalidation 和 stability 工作；
- 正式 0.2.0 release hardening；
- broad tree-family expansion。
