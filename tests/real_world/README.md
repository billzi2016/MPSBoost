# Real-world test suite

This directory is reserved for real-world dataset acceptance tests.

The suite verifies that MPSBoost works on practical datasets and workflows before any `1.x`
release. It is separate from unit tests, integration tests, and synthetic benchmarks.

Dataset matrix:

- `dataset_matrix.py` is the executable S18 dataset matrix.
- Default no-network acceptance currently runs only active built-in datasets.
- Multiclass datasets default to native CPU softmax. MPS multiclass currently uses the explicit
  staged OvR compatibility path until native MPS softmax kernels are implemented.

Initial dataset targets:

- Iris: small multiclass sanity test.
- Breast Cancer Wisconsin: binary classification baseline.
- Diabetes: small regression sanity test.
- California Housing: medium regression baseline.
- Digits: lightweight flattened-image multiclass test.
- MNIST subset: opt-in flattened-image stress test.
- Titanic: missing-value and categorical-feature workflow test.
- Adult Income: larger categorical binary-classification test.
- Covertype subset: larger multiclass throughput test.
- Higgs subset: opt-in large numeric binary-classification performance-boundary test.

Rules:

- Do not mock CPU or MPS backends in this directory.
- Do not commit raw external datasets into the repository or wheel.
- Prefer built-in sklearn datasets for default CI coverage.
- External datasets must be versioned, hash-checked, cached, and reproducible offline.
- Long-running tests must be opt-in and clearly marked.
- Blocked datasets must stay visible in the matrix instead of being silently replaced by
  synthetic or binary-subset stand-ins.

Explicit downloads:

```bash
python tests/real_world/download_datasets.py california-housing
python tests/real_world/download_datasets.py covertype-subset
```

The downloaded files live under ignored `tests/real_world/data/`, and generated manifests live
under ignored `tests/real_world/cache/`.
