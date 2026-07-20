"""The sole MPSBoost Python input-adaptation implementation.

This module only normalizes dense two-dimensional numeric input and one-dimensional
labels into contiguous arrays supported by the native layer. C++ retains complete
size, finite-value, stride, and overflow validation. Do not implement binning,
objectives, or training logic here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def as_dense_matrix(value: Any) -> NDArray[np.float32] | NDArray[np.float64]:
    """Convert user input into a two-dimensional floating-point array borrowable by native training.

    Positive-stride float32/float64 arrays retain dtype and view semantics; other
    real dtypes convert to float32. Objects, complex values, booleans, and non-2D
    inputs are rejected without implicit sparse expansion.
    """

    if hasattr(value, "toarray") and not isinstance(value, np.ndarray):
        raise TypeError("Sparse matrices are not supported")
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError("X must be a two-dimensional dense numeric array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("X must contain at least one row and one feature")
    if array.dtype.kind not in "iuf":
        raise TypeError("X dtype must be a real numeric type")
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        array = array.astype(np.float32)
    if any(stride <= 0 for stride in array.strides):
        # The native layer rejects negative/zero strides. Normalize at the user API
        # into a C-contiguous copy so identical input behaves the same in fit and
        # predict. training_summary_ explicitly reports the copy.
        array = np.ascontiguousarray(array)
    return array


def as_labels(value: Any, expected_rows: int) -> NDArray[np.float64]:
    """Return contiguous one-dimensional float64 labels and validate row count.

    Parameter errors occur before device initialization. Native objectives validate
    finite values and extreme ranges again, preventing callers that bypass the
    Python entry point from creating a second semantic path.
    """

    labels = np.asarray(value)
    if labels.ndim != 1:
        raise ValueError("y must be a one-dimensional numeric array")
    if labels.shape[0] != expected_rows:
        raise ValueError("X and y sample counts do not match")
    if labels.dtype.kind not in "iuf":
        raise TypeError("y dtype must be a real numeric type")
    return np.ascontiguousarray(labels, dtype=np.float64)


def as_sample_weight(value: Any, expected_rows: int) -> NDArray[np.float64]:
    """Return contiguous non-negative sample weights for native objective statistics.

    The native trainer revalidates weights before multiplying gradients and Hessians. Python keeps
    the adapter small so every estimator can share the same input contract without copying weight
    validation logic into each ``fit`` method.
    """

    if value is None:
        return np.ones(expected_rows, dtype=np.float64)
    weights = np.asarray(value)
    if weights.ndim != 1:
        raise ValueError("sample_weight must be a one-dimensional numeric array")
    if weights.shape[0] != expected_rows:
        raise ValueError("sample_weight length must match X rows")
    if weights.dtype.kind not in "iuf":
        raise TypeError("sample_weight dtype must be numeric")
    result = np.ascontiguousarray(weights, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("sample_weight values must be finite")
    if np.any(result < 0.0):
        raise ValueError("sample_weight values must be non-negative")
    if float(result.sum()) <= 0.0:
        raise ValueError("sample_weight total must be positive")
    return result
