# Changelog

All notable changes to this project will be documented in this file.

## Unreleased - 0.2.0a0 development

- Replace preview mock APIs with a real compiled MPS/Metal backend foundation.
- Add native device diagnostics and a real GPU vector smoke kernel.
- Build Metal shaders at package build time and bundle the resulting library.
- Add deterministic native quantization for strided float32/float64 buffers.
- Add compact uint8/uint16 feature-major bins and validated serialization.
- Reject invalid values, unsupported layouts, corrupted data, and size overflows.

## 0.1.0a0 - 2026-07-19

- Reserve the `mpsboost` package namespace.
- Publish the planned sklearn/XGBoost-style Python API surface.
- Add backend diagnostics and layered cache-path definitions.
- Fail explicitly for training operations that are not implemented yet.
