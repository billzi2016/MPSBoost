# Real-World Acceptance Report

This report records the current S18 release-gate evidence. It is intentionally honest: default CI
uses no network, opt-in external datasets run only from ignored local caches, and `1.0.0` remains
blocked until the complete matrix, performance audit, and user commitment are finished.

## Default No-Network Matrix

- Iris: active multiclass native CPU softmax acceptance.
- Digits: active flattened-image multiclass native CPU softmax acceptance.
- Breast Cancer Wisconsin: active binary classification acceptance and sklearn CPU baseline record.
- Diabetes: active regression acceptance, advanced objective acceptance, ranking smoke, and sklearn CPU baseline record.

## Cached Opt-In Matrix

- California Housing: active cached regression acceptance through sklearn fetcher.
- Covertype subset: active cached large-row multiclass acceptance and real MPS parity smoke.
- MNIST subset: active cached OpenML flattened-image multiclass acceptance.
- Titanic: active cached OpenML missing-value and categorical workflow acceptance.
- Adult Income: active cached OpenML categorical binary-classification acceptance.
- HIGGS subset: active local-file large numeric binary-classification acceptance.

## Degradation and Boundary Notes

- Small datasets may run faster on CPU because GPU launch and synchronization overhead dominate.
- IsolationForest and LearningToRankRegressor are CPU-suitable workflows; MPS requests warn and record the CPU backend decision.
- MPS native multiclass softmax remains phased. CPU native softmax is the default native multiclass implementation; MPS multiclass uses observable OvR compatibility until native MPS softmax kernels are ready.
- Optional external datasets are not downloaded during ordinary tests. Missing caches skip with copy-paste setup instructions.

## Open Release Gates

- S18.6/S18.6a: full model-quality, end-to-end performance, peak-memory, wheel-size, model-size, and permission audit.
- S18.8: complete G17 matrix acceptance before planning `1.0.0`.
- S18.9/S18.10: user confirmation and formal PyPI `1.0.0` release.
