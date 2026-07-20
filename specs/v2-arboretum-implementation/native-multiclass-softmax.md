# Native Multiclass Softmax Specification

## Goal

The final default multiclass implementation in MPSBoost must be native multiclass
softmax, not a wrapper around several binary models.

OvR may remain a compatibility layer and fallback, but must not be presented in
documentation, tests, or release notes as native softmax.

## Public API Constraints

The user interface remains sklearn/XGBoost-style:

- `fit(X, y, sample_weight=None)`
- `predict(X)`
- `predict_proba(X)`
- `decision_function(X)`
- `score(X, y, sample_weight=None)`
- `get_params(deep=True)`
- `set_params(**params)`
- `GridSearchCV` / `RandomizedSearchCV` / `cross_val_score`

Switching from OvR to native softmax must not require users to change training code.

## Strategy Parameter

`multi_strategy` values:

- `auto`: default strategy. Choose native softmax when available; otherwise choose
  a clearly documented fallback.
- `softmax`: force native softmax. When native softmax is not implemented on the
  target device, fail clearly or report the actual compatibility strategy.
- `ovr`: explicitly use the OvR compatibility layer.

Current stage:

- CPU multiclass `auto` uses native softmax.
- Before MPS native softmax completes, CPU softmax or OvR must not be reported as MPS softmax.

## Label Encoding

The Python layer encodes user multiclass labels as contiguous class IDs in
`[0, num_class)`.

Models must store or hold the original `classes_` and map argmax class IDs back to
user labels during prediction.

## Base Margin

The CPU oracle computes class priors using sample weights.

For class `k`:

```text
base_score[k] = log(max(epsilon, weighted_count[k] / total_weight))
```

Softmax is insensitive to uniform translation, so base scores need not normalize to
sum to zero; every value must be finite.

## Softmax Probabilities

For each row of raw margin `m`:

```text
p[k] = exp(m[k] - max(m)) / sum_j exp(m[j] - max(m))
```

Every probability row must:

- be entirely finite;
- be entirely nonnegative;
- sum approximately to 1;
- satisfy `predict == classes_[argmax(p)]`.

## Gradient / Hessian

Each training round first computes one row's softmax probabilities from all current
class margins.

For target class `k`:

```text
g[k] = p[k] - 1(label == k)
h[k] = p[k] * (1 - p[k])
```

This is the diagonal-Hessian approximation for multiclass softmax; it is not
independent OvR binary logistic.

Sample weights multiply gradients and Hessians uniformly in native code:

```text
weighted_g = sample_weight * g
weighted_h = sample_weight * h
```

## Tree-Update Structure

The CPU oracle currently uses round-major structure:

```text
round 0: tree(class 0), tree(class 1), ..., tree(class K-1)
round 1: tree(class 0), tree(class 1), ..., tree(class K-1)
...
```

Each tree updates one class margin, but its gradient comes from same-round all-class
softmax probabilities.

## Model Format

The native model header uses explicit model kinds to distinguish legacy
regression/binary-logistic payloads from multiclass-softmax payloads.

- `model kind = 0`: legacy regression/binary-logistic format; backward-compatible
  loading is mandatory.
- `model kind = 1`: native multiclass softmax format.

Multiclass-softmax payloads must store:

- `class_count`;
- `learning_rate`;
- every class's `base_score`;
- numeric `classes_` mapping used by the Python public API;
- frozen training quantization schema;
- round-major class tree/update structure.

Legacy regression/classifier loaders must clearly fail on `model kind = 1`; do not
parse a multiclass-margin file as regression or binary logistic. Native softmax
loaders also clearly fail on legacy `model kind = 0`; Python classifiers then enter
the binary-logistic compatibility loading path.

## MPS Gate

MPS native softmax requires separate acceptance:

- multiclass margin buffers;
- softmax gradient/Hessian kernels;
- class-major or row-major histogram input ABI;
- CPU-oracle comparison;
- real-dataset parity;
- explicit performance report.

Before these complete, MPS must not claim native softmax is complete.

Current-stage gates:

- `multi_strategy="softmax", device="mps"` must clearly indicate and use the OvR
  compatibility strategy, with the actual strategy shown in the training summary.
- `multi_strategy="auto", device="mps"` may use the OvR compatibility fallback,
  but the training summary must show `strategy = "one_vs_rest"`.
- CPU `auto` and CPU `softmax` must use native softmax.
- Before native softmax kernels complete, MPS parity compares CPU OvR with MPS OvR
  only; CPU native softmax is not an MPS-native-softmax reference.

## Acceptance

Minimum acceptance for native softmax:

- Directly test softmax probabilities, gradients, and Hessians in objective helpers;
- CPU native-softmax estimators produce no `estimators_` OvR submodels;
- `multi_strategy="softmax", device="mps"` clearly indicates and uses the OvR
  compatibility strategy, preventing silent CPU-softmax masquerading;
- after save/load, class mapping, raw margins, probabilities, and predictions agree;
- `predict_proba` rows normalize;
- `GridSearchCV` can tune parameters;
- Iris / Digits / Covertype subset cover the default CPU native-softmax path;
- Covertype subset MPS current behavior covers the OvR compatibility fallback;
- OvR remains explicitly available through `multi_strategy="ovr"`.
