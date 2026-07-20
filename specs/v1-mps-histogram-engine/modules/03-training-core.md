# Module Design: Training Core

## 1. Responsibility
The training core defines objectives, split gain, leaf weights, tree structure, and
the boosting state machine. It depends only on abstract `ComputeBackend`, not
Python, Metal objects, cache paths, or the file system.

## 2. Single Mathematical Semantics
Squared error:
```text
g_i = prediction_i - label_i
h_i = 1
```
Node score, leaf weight, and split gain:
```text
score(G, H) = G² / (H + λ)
weight(G, H) = -G / (H + λ)
gain = 0.5 × (score_left + score_right - score_parent) - γ
```
All implementations call shared domain functions or strictly match their kernel
contracts. Do not maintain separate formulas in Python, CPU, and GPU.

## 3. Tree Model
Flat nodes contain feature index, threshold bin, left/right child indices, leaf value,
gain, and flags. A flag distinguishes leaves; do not use special floating values.
Validate all indices on creation and loading.

Stable tie-break order:
1. larger gain;
2. smaller feature index;
3. smaller threshold bin;
4. retain the first candidate when all other conditions are identical.

Even for nearby floating values, use frozen comparison rules and never thread
completion order. The frozen rule is strict FP64 `gain >`; enter feature/bin
tie-break only when bit-level results are equal, without scale-dependent epsilon.
Candidate gain must be strictly positive; zero or negative values remain leaves.
Binned values `bin <= threshold_bin` go left; all others go right.

Training parameters require `min_samples_leaf >= 1`, `min_child_weight >= 0`,
`reg_lambda >= 0`, and `gamma >= 0`; every floating parameter is finite. Both
children meet minimum samples, minimum Hessian, and strictly positive Hessian;
violating candidates never enter gain comparison.

## 4. Backend Interface
```text
ComputeBackend
├── compute_gradients(...)
├── build_histograms(...)
├── update_predictions(...)
├── synchronize()
└── diagnostics()
```
The interface passes stable POD views and preallocated outputs. The core decides when
to call; backends decide how to compute. Extend fine-grained capability interfaces
when split/partition move to GPU, without exposing command buffers to the core.

## 5. State Machine
```text
Created → Validated → Quantized → BackendReady
        → Iterating → ModelFinalized → Completed
                         ↘ Failed
```
- Validate preconditions on every state transition;
- `Failed` must not export a model;
- the estimator atomically receives a model only after `Completed`;
- the training session owns all temporary resources and uniformly releases exception paths.

## 6. Tree Growth
Version 0.2.0 grows depth-limited trees level by level:
1. obtain active nodes in the current level;
2. build node histograms;
3. select splits by stable rules;
4. check minimum samples and Hessian;
5. partition samples;
6. create the next level or freeze leaves;
7. update training predictions after the tree completes.

The level-wise strategy facilitates batching multiple nodes and is the sole default
strategy; do not retain another unspecified depth-first production path.

## 7. CPU Oracle
The CPU backend prioritizes clarity and determinism and accumulates FP64 as the
oracle. It is not a hidden fallback. After a user chooses `mps`, unavailable
backends fail; the core may let CPU handle control flow by specified policy, but must
not replace the entire GPU hot path and still report successful `mps`.

The CPU histogram accumulates each bin's `count/G/H` in supplied row-index order,
and tree nodes are written breadth-first to a flat array. Tree-growth control flow
exists only in core; the CPU backend builds histograms only and must not copy split
selection, partitioning, or node assembly.

## 8. Acceptance
- Hand-calculated small trees agree node by node;
- prediction updates across multiple boosting rounds are correct;
- repeated training with identical parameters satisfies deterministic constraints;
- backend exceptions do not produce partial models;
- core tests need neither Python nor a real GPU; GPU integration is verified separately with real hardware.
