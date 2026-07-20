# Module Design: Inputs and Binning

## 1. Responsibility
Convert user numeric matrices into a stable, compact binned representation and define
ownership, missing-value, boundary, and serialization semantics. Binning output is
the shared input of the CPU oracle and MPS backend.

## 2. Input Invariants
- `rows > 0`, `features > 0`;
- `rows × features × itemsize` uses checked 64-bit arithmetic;
- all 0.2.0 inputs are finite;
- reject unsupported dtypes before allocating large memory;
- rows and features must not exceed internal index-type limits.

## 3. Binning Algorithm
Version 0.2.0 uses deterministic quantile boundaries:
1. Read finite values by feature;
2. Generate at most `max_bins - 1` split boundaries with stable rules;
3. Remove equal boundaries; constant features produce only one valid bin;
4. Map values using fixed `upper_bound/lower_bound` semantics;
5. Identical input, parameters, and version produce identical boundaries and bins.

Implement the algorithm once. Parallelization may change execution order but not
boundary rules.

## 4. Internal Representation
- Use `uint8_t` when `max_bins <= 256`;
- use `uint16_t` for `257..65536`;
- reject larger values in 0.2.0;
- boundaries use contiguous `float32` or frozen precision;
- feature metadata records offsets and valid-bin counts.

The default training layout is feature-major for contiguous reads of samples for one
feature; the final decision requires layout benchmarks. Model format stores only
boundaries, not training layout.

## 5. Data Ownership
- Borrow Python input only during synchronous validation and conversion;
- `BinnedDataset` owns its binned memory;
- do not retain raw pointers to temporary user buffers;
- when zero-copy conditions fail, copy explicitly and make it diagnosable;
- after device upload, the training session uniformly manages host-data release timing.

## 6. Cache
Version 0.2.0 first supports in-process reuse. Enable disk binned caches only after:
- fingerprints cover data content, shape, dtype, binning parameters, and format version;
- atomic writes and validation;
- labels are not cached;
- users can inspect and clear them;
- corruption or mismatches rebuild them.

## 7. Errors and Safety
- Check every offset and stride calculation for overflow;
- do not allow NaN to map to different bins on different platforms;
- estimate host and device peaks before allocation;
- errors state data shape, required types, and limits without printing raw data.

## 8. Acceptance
- Test manually calculated boundaries, constants, duplicate values, extreme skew,
  maximum bins, and non-contiguous input;
- repeated runs on the same data produce exactly identical bins;
- the CPU oracle and MPS use the same read-only view;
- no out-of-bounds access, dangling pointers, or hidden multiple full copies.
