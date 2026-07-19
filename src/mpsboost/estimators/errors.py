"""Estimator-specific exceptions for MPSBoost public APIs.

This module stays dependency-free so all estimator families can share fitted-state errors without
importing native bindings, device helpers, or model-family implementations.
"""


class NotFittedError(RuntimeError):
    """Raised when fitted-only functionality is called before a complete model exists."""
