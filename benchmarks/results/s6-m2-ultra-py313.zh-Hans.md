# MPSBoost histogram benchmark

历史基线说明：这些结果记录于后续 v2/v3 优化和 estimator 清理之前，仅用于可追溯记录，不代表当前 HEAD 或最终 `0.3.0` 性能承诺。

Wall time 包含主机端准备、buffer 传输、command submission 和 synchronization。

| 场景 | 行数 | 特征数 | CPU median (s) | MPS median (s) | 加速比 | Buffer pool 复用 |
|---|---:|---:|---:|---:|---:|---:|
| small，小型输入 | 4096 | 16 | 0.000736 | 0.001613 | 0.456x | 6 |
| medium，中型输入 | 32768 | 32 | 0.005142 | 0.004310 | 1.193x | 13 |
| large，大行数输入 | 131072 | 64 | 0.035795 | 0.012662 | 2.827x | 20 |
| wide，宽表输入 | 32768 | 256 | 0.042130 | 0.015056 | 2.798x | 27 |

# MPSBoost end-to-end regressor benchmark

历史基线说明：这些结果记录于后续 v2/v3 优化和 estimator 清理之前，仅用于可追溯记录，不代表当前 HEAD 或最终 `0.3.0` 性能承诺。

Wall time 包含 Python 输入适配、quantization、training、model assembly 和 synchronization。

| 场景 | 行数 | 特征数 | CPU median (s) | MPS median (s) | 加速比 | 最大预测差异 |
|---|---:|---:|---:|---:|---:|---:|
| gbdt-medium，中型 GBDT | 16384 | 32 | 0.067630 | 0.089597 | 0.755x | 3.36394e-06 |
| gbdt-wide，宽表 GBDT | 16384 | 128 | 0.283176 | 0.249502 | 1.135x | 4.35114e-06 |
| gbdt-large-wide，大型宽表 GBDT | 32768 | 256 | 1.031002 | 0.633006 | 1.629x | 5.36442e-06 |
