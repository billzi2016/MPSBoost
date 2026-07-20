# S4 histogram benchmark

历史基线说明：该结果记录于后续 histogram 和端到端优化之前，仅用于可追溯记录，不代表当前 HEAD 或最终 `0.3.0` 性能。

本报告记录完整 preregistered run；没有移除任何 scenario 或 repetition。Wall time 包含主机端转换、buffer 传输、command submission、synchronization 和结果 materialization。

| Scenario | Rows | Features | CPU median (s) | MPS median (s) | Speedup |
|---|---:|---:|---:|---:|---:|
| small | 4096 | 16 | 0.000782 | 0.001723 | 0.454x |
| medium | 32768 | 32 | 0.004980 | 0.004473 | 1.113x |
| large | 131072 | 64 | 0.034616 | 0.013495 | 2.565x |
| wide | 32768 | 256 | 0.040275 | 0.015453 | 2.606x |

Device：Apple M2 Ultra；Python 3.13.5；MPSBoost 0.2.0a0。
