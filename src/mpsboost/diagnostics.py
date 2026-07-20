"""MPSBoost backend diagnostics and real GPU smoke validation.

This module locates shader resources within the wheel and exposes non-sensitive
native device information as stable Python dictionaries. It does not create
caches, access the network, initialize training state, or export test entry
points through the top-level API.
"""

from __future__ import annotations

from contextlib import contextmanager
from importlib.resources import as_file, files
import os
from typing import Any, Iterator, Sequence
import warnings

from . import _native
from ._cache import (
    cache_info as _cache_info,
    clear_cache as _clear_cache,
    ensure_cache_directories as _ensure_cache_directories,
)


def is_available() -> bool:
    """Return whether this wheel and machine have a usable real MPS backend.

    The result comes from a native Metal-device query rather than platform-string
    inference. The query has no file-write or network side effects. Missing native
    extensions or shaders indicate an invalid installation and fail explicitly.
    """

    return bool(_native._backend_info()["available"])


def system_info() -> dict[str, Any]:
    """Return non-sensitive backend information for installation diagnostics.

    The result excludes persistent device identifiers, usernames, paths, and
    environment variables. Byte sizes remain integers so callers own display units.
    """

    info = dict(_native._backend_info())
    info.update(
        {
            "backend": "mps",
            "package_version": _native.__version__,
        }
    )
    return info


def mps_setup_instructions() -> str:
    """Return copy-paste setup commands for users whose local MPS environment is unavailable."""

    return (
        "Apple GPU acceleration is unavailable in this environment. To enable MPSBoost GPU "
        "training on an Apple Silicon Mac, run:\n"
        "  xcode-select --install\n"
        "  xcodebuild -downloadComponent MetalToolchain\n"
        "Then reinstall or verify the package with:\n"
        "  python -m pip install --upgrade --force-reinstall mpsboost\n"
        "  python -c \"import mpsboost as mb; print(mb.system_info())\"\n"
        "To skip this import-time environment check for CPU-only jobs, GridSearchCV workers, "
        "or managed CI, run:\n"
        "  MPSBOOST_SKIP_ENV_CHECK=1 python your_script.py\n"
        'CPU training remains available with device="cpu" or device="auto".'
    )


def warn_if_mps_unavailable() -> None:
    """Warn once at import when GPU acceleration is unavailable, without blocking CPU workloads."""

    if os.environ.get("MPSBOOST_SKIP_ENV_CHECK") == "1":
        return
    if is_available():
        return
    warnings.warn(mps_setup_instructions(), RuntimeWarning, stacklevel=2)


def cache_info() -> dict[str, Any]:
    """Return a non-sensitive L2 cache-path summary without creating directories."""

    info = _cache_info()
    return {
        "root": str(info.root),
        "exists": info.exists,
        "pipelines": str(info.pipelines),
        "quantization": str(info.quantization),
        "tuning": str(info.tuning),
    }


def create_cache() -> dict[str, str]:
    """Explicitly create L2 cache directories and return their paths.

    This public entry point has file-system side effects, so ``import``,
    ``system_info``, and ``cache_info`` never call it automatically.
    """

    layout = _ensure_cache_directories()
    return {
        "root": str(layout.root),
        "pipelines": str(layout.pipelines),
        "quantization": str(layout.quantization),
        "tuning": str(layout.tuning),
    }


def clear_cache() -> int:
    """Explicitly clear the L2 cache root and estimate removed regular files."""

    return _clear_cache()


@contextmanager
def _metallib_path() -> Iterator[str]:
    """Yield the real path to the wheel's sole shader library for the context lifetime.

    Estimators, kernel tests, and smoke tests share this locator so entry points
    cannot make incompatible assumptions about compressed resources or temporary
    file lifetime. Callers must complete GPU work inside the ``with`` block.
    """

    resource = files("mpsboost").joinpath("_kernels.metallib")
    with as_file(resource) as metallib_path:
        yield str(metallib_path)


def _run_vector_add_for_test(
    left: Sequence[float], right: Sequence[float]
) -> list[float]:
    """Run the internal vector-add integration test on a real GPU.

    Args:
        left: Left input sequence.
        right: Right input sequence with the same length as left.

    Returns:
        Element-wise sum computed by the GPU.

    Raises:
        BackendError: A device, shader, pipeline, buffer, or command stage fails.

    ``as_file`` supports ordinary wheel files and possible future compressed
    resources. The native call must complete within the context so temporary paths
    remain valid while the GPU loads the resource.
    """

    with _metallib_path() as metallib_path:
        return list(_native._vector_add(left, right, metallib_path))
