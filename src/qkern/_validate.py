"""Argument normalisation shared by the batch wrappers (spec DQ-02, DQ-03, DQ-05, DQ-11).

Range checks (window >= 1, 1 <= min_periods <= window, ddof >= 0, alpha/span/halflife) live in
the C++ extension, shared with the streaming classes; this module handles Python types.
"""

from __future__ import annotations

import operator
from typing import Literal, TypeAlias, cast, get_args

import numpy as np
import numpy.typing as npt

NanPolicy: TypeAlias = Literal["skip", "propagate", "raise"]
Interp: TypeAlias = Literal["linear", "lower", "higher", "nearest", "midpoint"]

_NAN_POLICIES: tuple[str, ...] = get_args(NanPolicy)


def as_input(x: object, name: str = "x") -> npt.NDArray[np.generic]:
    """Return ``x`` as a 1-D or 2-D NumPy array the extension accepts, without copying.

    float64 and float32 arrays pass through untouched (any stride); integer and bool arrays
    pass through too and are converted once in C++ (counted by the debug counters). Other
    real floats (float16, longdouble) are converted here to float64. Complex, object, string,
    datetime and non-native byte-order inputs raise TypeError.
    """
    arr = np.asarray(x)
    if arr.ndim not in (1, 2):
        raise TypeError(f"{name} must be 1-D or 2-D, got an array with ndim={arr.ndim}")
    dtype = arr.dtype
    if dtype.kind not in "iubf":
        raise TypeError(f"{name} has unsupported dtype {dtype}; expected a real numeric dtype")
    if not dtype.isnative:
        raise TypeError(f"{name} has non-native byte order {dtype.str!r}; convert it first")
    if dtype.kind == "f" and dtype not in (np.float64, np.float32):
        return arr.astype(np.float64)
    return arr


def check_int(value: object, name: str) -> int:
    """An integer argument (bool is rejected); range checks happen in C++."""
    if isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be an integer, got bool")
    try:
        return operator.index(cast(int, value))
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from None


def resolve_min_periods(min_periods: object, window: int) -> int:
    """Default ``min_periods`` to ``window``; the range is checked in C++."""
    if min_periods is None:
        return window
    return check_int(min_periods, "min_periods")


def check_nan_policy(nan_policy: object) -> NanPolicy:
    """Validate ``nan_policy``."""
    if nan_policy not in _NAN_POLICIES:
        raise ValueError(f"nan_policy must be one of {_NAN_POLICIES}, got {nan_policy!r}")
    return cast(NanPolicy, nan_policy)


def check_out(out: object, shape: tuple[int, ...]) -> npt.NDArray[np.float64]:
    """Validate an ``out=`` target: writeable native float64 of the input's shape.

    Overlap with the input is checked in C++ (DQ-11).
    """
    if not isinstance(out, np.ndarray):
        raise TypeError(f"out must be a numpy.ndarray, got {type(out).__name__}")
    if out.dtype != np.float64 or not out.dtype.isnative:
        raise TypeError(f"out must have native float64 dtype, got {out.dtype}")
    if out.ndim != len(shape):
        raise TypeError(f"out must have ndim={len(shape)}, got ndim={out.ndim}")
    if not out.flags.writeable:
        raise TypeError("out must be writeable")
    if out.shape != shape:
        raise ValueError(f"out has shape {out.shape}, expected {shape}")
    return cast(npt.NDArray[np.float64], out)
