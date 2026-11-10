"""Public batch wrappers. Each normalises its arguments and calls the C++ extension once."""

from __future__ import annotations

from typing import Any, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

from qkern import _qkern
from qkern._validate import (
    NanPolicy,
    as_input,
    check_nan_policy,
    check_out,
    check_window,
    resolve_min_periods,
)

F64: TypeAlias = npt.NDArray[np.float64]


class ArrowArrayExportable(Protocol):
    """Any object implementing the Arrow PyCapsule interface for arrays."""

    def __arrow_c_array__(
        self, requested_schema: object | None = None
    ) -> tuple[object, object]: ...


ArrayIn: TypeAlias = (
    npt.NDArray[np.floating[Any]] | npt.NDArray[np.integer[Any]] | ArrowArrayExportable
)


def rolling_mean(
    x: ArrayIn,
    window: int,
    *,
    min_periods: int | None = None,
    nan_policy: NanPolicy = "skip",
    out: F64 | None = None,
) -> F64:
    """Rolling mean over the last ``window`` observations.

    Definition: at index ``t`` the output is the mean of the valid (non-NaN) values among
    positions ``t - window + 1 .. t``. Cost: O(1) per element (sliding Welford update on a
    ring buffer); one pass over the input.

    NaN and ``min_periods``: with ``nan_policy="skip"`` NaN values are left out of the
    window; ``"propagate"`` gives NaN while any NaN is inside the window; ``"raise"``
    raises ValueError if the input holds any NaN. The output is NaN while fewer than
    ``min_periods`` valid values are in the window (default ``min_periods = window``).
    ``+inf`` and ``-inf`` are valid values: the mean is ``±inf`` while one sign is present
    and NaN when both are.

    dtypes: float64 and float32 inputs of any stride are read in place; other real dtypes
    are converted once to float64. The result is float64. With ``out=`` (writeable float64,
    same length, not overlapping ``x``) the result is written in place and ``out`` is
    returned.

    Example:
        >>> import numpy as np, qkern
        >>> qkern.rolling_mean(np.array([1.0, 2.0, 3.0, 4.0]), 2)
        array([nan, 1.5, 2.5, 3.5])
    """
    arr = as_input(x)
    w = check_window(window)
    mp = resolve_min_periods(min_periods, w)
    policy = check_nan_policy(nan_policy)
    if out is None:
        return _qkern.rolling_mean(arr, w, mp, policy)
    target = check_out(out, arr.shape[0])
    _qkern.rolling_mean_into(arr, w, mp, policy, target)
    return target


def build_info() -> dict[str, str | int | bool]:
    """Return build metadata: version, git sha, compiler, flags, ABI and GIL threshold."""
    info: dict[str, str | int | bool] = {}
    for key, value in _qkern._build_info().items():
        if not isinstance(value, str | int):  # bool is an int subclass
            raise TypeError(f"unexpected build_info value for {key!r}: {value!r}")
        info[key] = value
    return info


def show_config() -> None:
    """Print :func:`build_info` as aligned ``key: value`` lines."""
    info = build_info()
    width = max(len(k) for k in info)
    for key, value in info.items():
        print(f"{key:<{width}} : {value}")
