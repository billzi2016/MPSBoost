# MPSBoost histogram benchmark

Historical baseline note: these measurements were recorded before the later v2/v3 optimization
and estimator cleanup work. They are retained for traceability and should not be read as current
HEAD or final `0.3.0` performance claims.

Wall time includes host preparation, buffer transfer, command submission, and synchronization.

| Scenario | Rows | Features | CPU median (s) | MPS median (s) | Speedup | Pool reuse |
|---|---:|---:|---:|---:|---:|---:|
| small | 4096 | 16 | 0.000736 | 0.001613 | 0.456x | 6 |
| medium | 32768 | 32 | 0.005142 | 0.004310 | 1.193x | 13 |
| large | 131072 | 64 | 0.035795 | 0.012662 | 2.827x | 20 |
| wide | 32768 | 256 | 0.042130 | 0.015056 | 2.798x | 27 |

# MPSBoost end-to-end regressor benchmark

Historical baseline note: these measurements were recorded before the later v2/v3 optimization
and estimator cleanup work. They are retained for traceability and should not be read as current
HEAD or final `0.3.0` performance claims.

Wall time includes Python input adaptation, quantization, training, model assembly, and synchronization.

| Scenario | Rows | Features | CPU median (s) | MPS median (s) | Speedup | Max prediction diff |
|---|---:|---:|---:|---:|---:|---:|
| gbdt-medium | 16384 | 32 | 0.067630 | 0.089597 | 0.755x | 3.36394e-06 |
| gbdt-wide | 16384 | 128 | 0.283176 | 0.249502 | 1.135x | 4.35114e-06 |
| gbdt-large-wide | 32768 | 256 | 1.031002 | 0.633006 | 1.629x | 5.36442e-06 |
