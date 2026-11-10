"""qkern C++ extension: streaming and rolling time-series kernels."""

from typing import Annotated, overload

import numpy
from numpy.typing import NDArray


def _build_info() -> dict[str, object]:
    """Build metadata used by qkern.build_info()."""

def _set_gil_release_min_n(n: int) -> None:
    """Private test/benchmark hook: GIL release threshold in elements."""

def _get_gil_release_min_n() -> int:
    """Private test/benchmark hook: current GIL release threshold."""

def _debug_counters() -> dict[str, int]:
    """
    Private test hook: input conversions and output copies made by the binding layer.
    """

def _reset_debug_counters() -> None:
    """Private test hook: zero the debug counters."""

@overload
def rolling_mean(x: Annotated[NDArray[numpy.float64], dict(shape=(None,), device='cpu', writable=False)], window: int, min_periods: int, nan_policy: str) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]: ...

@overload
def rolling_mean(x: Annotated[NDArray[numpy.float32], dict(shape=(None,), device='cpu', writable=False)], window: int, min_periods: int, nan_policy: str) -> Annotated[NDArray[numpy.float64], dict(shape=(None,))]:
    """
    Rolling mean over the last `window` observations (see qkern.rolling_mean).
    """

@overload
def rolling_mean_into(x: Annotated[NDArray[numpy.float64], dict(shape=(None,), device='cpu', writable=False)], window: int, min_periods: int, nan_policy: str, out: Annotated[NDArray[numpy.float64], dict(shape=(None,), device='cpu')]) -> None: ...

@overload
def rolling_mean_into(x: Annotated[NDArray[numpy.float32], dict(shape=(None,), device='cpu', writable=False)], window: int, min_periods: int, nan_policy: str, out: Annotated[NDArray[numpy.float64], dict(shape=(None,), device='cpu')]) -> None:
    """Rolling mean written into a caller-supplied float64 array."""
