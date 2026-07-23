# MPSBoost 0.2.0b0 Release Audit

`0.2.0b0` was the GPU hot-path beta. It focused on the Metal histogram and split-processing
pipeline needed to make large tabular workloads faster than the CPU oracle.

Included:

- real MPS gradient and histogram kernels;
- split-scan, partition/compaction, histogram subtraction, and buffer-pool work;
- S6 benchmark evidence recording both wins and small-data regressions;
- continued CPU oracle validation.

Not included:

- final cache invalidation and stability work;
- formal 0.2.0 release hardening;
- broad tree-family expansion.
