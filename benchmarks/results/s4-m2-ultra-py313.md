# S4 histogram benchmark

This report records the complete preregistered run; no scenario or repetition was removed.
Wall time includes host conversion, buffer transfer, command submission, synchronization, and result materialization.

| Scenario | Rows | Features | CPU median (s) | MPS median (s) | Speedup |
|---|---:|---:|---:|---:|---:|
| small | 4096 | 16 | 0.000782 | 0.001723 | 0.454x |
| medium | 32768 | 32 | 0.004980 | 0.004473 | 1.113x |
| large | 131072 | 64 | 0.034616 | 0.013495 | 2.565x |
| wide | 32768 | 256 | 0.040275 | 0.015453 | 2.606x |

Device: Apple M2 Ultra; Python 3.13.5; MPSBoost 0.2.0a0.

