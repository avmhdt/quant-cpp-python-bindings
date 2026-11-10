"""Argument normalisation shared by the batch wrappers (spec DQ-02, DQ-03, DQ-05, DQ-11)."""

from __future__ import annotations

import operator
from typing import Literal, TypeAlias, cast, get_args

import numpy as np
import numpy.typing as npt

NanPolicy: TypeAlias = Literal["skip", "propagate", "raise"]
FloatArray: TypeAlias = npt.NDArray[np.float64] | npt.NDArray[np.float32]

_NAN_POLICIES: tuple[str, ...] = get_args(NanPolicy)


def as_input(x: object, name: str = "x") -> FloatArray:
    """Return ``x`` as a 1-D float64 or float32 NumPy array.

    float64 and float32 arrays are returned as they are (no copy, any stride). Integer,
    unsigned, bool and other real float dtypes are converted once to float64. Complex,
    object, datetime and non-native byte-order inputs raise TypeError.
    """
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise TypeError(f"{name} must be 1-D, got an array with ndim={arr.ndim}")
    dtype = arr.dtype
    if not dtype.isnative:
        raise TypeError(f"{name} has non-native byte order {dtype.str!r}; convert it first")
    if dtype == np.float64:
        return cast(npt.NDArray[np.float64], arr)
    if dtype == np.float32:
        return cast(npt.NDArray[np.float32], arr)
    if dtype.kind in "iubf":
        return arr.astype(np.float64)
    raise TypeError(f"{name} has unsupported dtype {dtype}; expected a real numeric dtype")


def check_window(window: object) -> int:
    """Validate a count-based window: an integer >= 1."""
    if isinstance(window, bool | np.bool_):
        raise TypeError("window must be an integer, got bool")
    try:
        value = operator.index(cast(int, window))
    except TypeError:
        raise TypeError(f"window must be an integer, got {type(window).__name__}") from None
    if value < 1:
        raise ValueError(f"window must be >= 1, got {value}")
    return value


def resolve_min_periods(min_periods: object, window: int) -> int:
    """Default ``min_periods`` to ``window`` and check ``1 <= min_periods <= window``."""
    if min_periods is None:
        return window
    if isinstance(min_periods, bool | np.bool_):
        raise TypeError("min_periods must be an integer or None, got bool")
    try:
        value = operator.index(cast(int, min_periods))
    except TypeError:
        raise TypeError(
            f"min_periods must be an integer or None, got {type(min_periods).__name__}"
        ) from None
    if not 1 <= value <= window:
        raise ValueError(f"min_periods must satisfy 1 <= min_periods <= window, got {value}")
    return value


def check_nan_policy(nan_policy: object) -> NanPolicy:
    """Validate ``nan_policy``."""
    if nan_policy not in _NAN_POLICIES:
        raise ValueError(f"nan_policy must be one of {_NAN_POLICIES}, got {nan_policy!r}")
    return cast(NanPolicy, nan_policy)


def check_out(out: object, n: int) -> npt.NDArray[np.float64]:
    """Validate an ``out=`` target: writeable 1-D native float64 of length ``n``.

    Overlap with the input is checked in C++ (DQ-11).
    """
    if not isinstance(out, np.ndarray):
        raise TypeError(f"out must be a numpy.ndarray, got {type(out).__name__}")
    if out.dtype != np.float64 or not out.dtype.isnative:
        raise TypeError(f"out must have native float64 dtype, got {out.dtype}")
    if out.ndim != 1:
        raise TypeError(f"out must be 1-D, got ndim={out.ndim}")
    if not out.flags.writeable:
        raise TypeError("out must be writeable")
    if out.shape[0] != n:
        raise ValueError(f"out has length {out.shape[0]}, expected {n}")
    return cast(npt.NDArray[np.float64], out)
