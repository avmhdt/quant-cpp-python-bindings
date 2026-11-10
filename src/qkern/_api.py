"""Public batch functions. Each normalises its arguments and calls the C++ extension once.

Shared rules (spec DQ-01 to DQ-06, DQ-11):

* Windows count observations: the window at row ``t`` is rows ``t - window + 1 .. t``.
* ``nan_policy="skip"`` (default) leaves NaN out of the window's valid values;
  ``"propagate"`` gives NaN while any NaN is in the window; ``"raise"`` raises ValueError
  before computing if the input holds a NaN.
* Outputs are NaN while fewer than ``min_periods`` valid values are in the window
  (default ``min_periods = window``; ``1 <= min_periods <= window``).
* ``+inf`` and ``-inf`` are valid values (pandas treats them as missing; see the docs'
  Semantics page).
* float64 and float32 inputs of any stride are read in place; integer and bool inputs are
  converted once to float64. 2-D inputs of shape ``(n, k)`` are processed column by column
  along axis 0. Results are float64 arrays that own their memory.
* ``out=`` takes a writeable float64 array of the input's shape that does not overlap it;
  the result is written there and ``out`` is returned.
* Calls on 1,024 or more elements release the GIL.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

from qkern import _qkern
from qkern._results import Moments, OLSResult
from qkern._validate import (
    NanPolicy,
    as_input,
    check_int,
    check_nan_policy,
    check_out,
    resolve_min_periods,
)

F64: TypeAlias = npt.NDArray[np.float64]


class ArrowArrayExportable(Protocol):
    """Any object implementing the Arrow PyCapsule interface for arrays."""

    def __arrow_c_array__(
        self, requested_schema: object | None = None
    ) -> tuple[object, object]: ...


ArrayIn: TypeAlias = (
    npt.NDArray[np.floating[Any]]
    | npt.NDArray[np.integer[Any]]
    | npt.NDArray[np.bool_]
    | ArrowArrayExportable
)


def _finish(result: F64 | None, out: F64 | None) -> F64:
    if out is not None:
        return out
    assert result is not None
    return result


def _rolling_args(
    x: ArrayIn, window: int, min_periods: int | None, nan_policy: NanPolicy
) -> tuple[npt.NDArray[np.generic], int, int, NanPolicy]:
    arr = as_input(x)
    w = check_int(window, "window")
    return arr, w, resolve_min_periods(min_periods, w), check_nan_policy(nan_policy)


def _out(out: F64 | None, arr: npt.NDArray[np.generic]) -> F64 | None:
    return None if out is None else check_out(out, arr.shape)


# ---------------------------------------------------------------------------------------
# Rolling moments
# ---------------------------------------------------------------------------------------


def rolling_mean(
    x: ArrayIn,
    window: int,
    *,
    min_periods: int | None = None,
    nan_policy: NanPolicy = "skip",
    out: F64 | None = None,
) -> F64:
    """Rolling mean of the valid values in each window.

    Algorithm: sliding sum (compensated) on a ring buffer, O(1) per element, one pass.
    A window holding both ``+inf`` and ``-inf`` gives NaN; one sign gives that infinity.
    See the module notes for NaN, ``min_periods``, dtype, 2-D and ``out=`` rules.

    Example:
        >>> import numpy as np, qkern
        >>> qkern.rolling_mean(np.array([1.0, 2.0, 3.0, 4.0]), 2)
        array([nan, 1.5, 2.5, 3.5])
    """
    arr, w, mp, policy = _rolling_args(x, window, min_periods, nan_policy)
    target = _out(out, arr)
    return _finish(_qkern.rolling_mean(arr, w, mp, policy, target), target)


def rolling_var(
    x: ArrayIn,
    window: int,
    *,
    min_periods: int | None = None,
    ddof: int = 1,
    nan_policy: NanPolicy = "skip",
    out: F64 | None = None,
) -> F64:
    """Rolling variance ``sum((v - mean)^2) / (n_valid - ddof)`` of each window.

    Algorithm: sliding Welford update with an exact two-pass recomputation every 65,536
    rows and whenever the sum of squares collapses (a large value left the window); O(1)
    amortised per element. NaN while any infinity is in the window, and while
    ``n_valid - ddof < 1``. A window of equal values gives exactly 0.

    Example:
        >>> import numpy as np, qkern
        >>> qkern.rolling_var(np.array([1.0, 2.0, 4.0, 8.0]), 3)
        array([       nan,        nan, 2.33333333, 9.33333333])
    """
    arr, w, mp, policy = _rolling_args(x, window, min_periods, nan_policy)
    d = check_int(ddof, "ddof")
    target = _out(out, arr)
    return _finish(_qkern.rolling_var(arr, w, mp, d, policy, target), target)


def rolling_std(
    x: ArrayIn,
    window: int,
    *,
    min_periods: int | None = None,
    ddof: int = 1,
    nan_policy: NanPolicy = "skip",
    out: F64 | None = None,
) -> F64:
    """Rolling standard deviation: the square root of :func:`rolling_var`.

    Example:
        >>> import numpy as np, qkern
        >>> qkern.rolling_std(np.array([1.0, 2.0, 3.0, 4.0]), 2)
        array([       nan, 0.70710678, 0.70710678, 0.70710678])
    """
    arr, w, mp, policy = _rolling_args(x, window, min_periods, nan_policy)
    d = check_int(ddof, "ddof")
    target = _out(out, arr)
    return _finish(_qkern.rolling_std(arr, w, mp, d, policy, target), target)


def rolling_moments(
    x: ArrayIn,
    window: int,
    *,
    min_periods: int | None = None,
    ddof: int = 1,
    nan_policy: NanPolicy = "skip",
) -> Moments:
    """Rolling mean, variance and valid count in one pass, as a :class:`Moments` tuple.

    ``count`` is the number of valid (non-NaN) values in each window, infinities included.
    Same rules as :func:`rolling_mean` and :func:`rolling_var`.

    Example:
        >>> import numpy as np, qkern
        >>> m = qkern.rolling_moments(np.array([1.0, np.nan, 3.0]), 2, min_periods=1)
        >>> m.count.tolist()
        [1, 1, 1]
    """
    arr, w, mp, policy = _rolling_args(x, window, min_periods, nan_policy)
    d = check_int(ddof, "ddof")
    mean, var, count = _qkern.rolling_moments(arr, w, mp, d, policy)
    return Moments(mean, var, count)


def rolling_zscore(
    x: ArrayIn,
    window: int,
    *,
    min_periods: int | None = None,
    ddof: int = 1,
    include_current: bool = True,
    nan_policy: NanPolicy = "skip",
    out: F64 | None = None,
) -> F64:
    """Rolling z-score ``(x_t - mean) / std``.

    With ``include_current=True`` the window is rows ``t - window + 1 .. t``; with
    ``include_current=False`` the mean and standard deviation come from the previous window
    (rows ``t - window .. t - 1``), so ``x_t`` is never part of its own baseline. NaN when
    ``x_t`` is NaN, when the standard deviation is NaN or 0. O(1) per element.

    Example:
        >>> import numpy as np, qkern
        >>> qkern.rolling_zscore(np.array([1.0, 2.0, 3.0, 6.0]), 3, include_current=False)
        array([nan, nan, nan,  4.])
    """
    arr, w, mp, policy = _rolling_args(x, window, min_periods, nan_policy)
    d = check_int(ddof, "ddof")
    target = _out(out, arr)
    return _finish(
        _qkern.rolling_zscore(arr, w, mp, d, bool(include_current), policy, target), target
    )


# ---------------------------------------------------------------------------------------
# Exponentially weighted
# ---------------------------------------------------------------------------------------


def _ewm_args(
    x: ArrayIn,
    alpha: float | None,
    span: float | None,
    halflife: float | None,
    min_periods: int,
    nan_policy: NanPolicy,
) -> tuple[npt.NDArray[np.generic], float, int, NanPolicy]:
    arr = as_input(x)
    a = _qkern._resolve_alpha(alpha, span, halflife)
    return arr, a, check_int(min_periods, "min_periods"), check_nan_policy(nan_policy)


def ewm_mean(
    x: ArrayIn,
    *,
    alpha: float | None = None,
    span: float | None = None,
    halflife: float | None = None,
    min_periods: int = 0,
    nan_policy: NanPolicy = "skip",
    out: F64 | None = None,
) -> F64:
    """Exponentially weighted mean ``m_t = (1 - alpha) m_{t-1} + alpha x_t``.

    Give exactly one of ``alpha`` in (0, 1], ``span >= 1`` (``alpha = 2 / (span + 1)``) or
    ``halflife > 0`` (``alpha = 1 - exp(-ln 2 / halflife)``). Starts at the first valid value
    (pandas ``adjust=False, ignore_na=True``). NaN inputs leave the state unchanged under
    ``nan_policy="skip"``; under ``"propagate"`` the first NaN makes every later output NaN.
    ``min_periods`` counts valid values seen so far (0 behaves like 1). O(1) per element.

    Example:
        >>> import numpy as np, qkern
        >>> qkern.ewm_mean(np.array([1.0, 2.0, 3.0]), alpha=0.5)
        array([1.  , 1.5 , 2.25])
    """
    arr, a, mp, policy = _ewm_args(x, alpha, span, halflife, min_periods, nan_policy)
    target = _out(out, arr)
    return _finish(_qkern.ewm_mean(arr, a, mp, policy, target), target)


def ewm_var(
    x: ArrayIn,
    *,
    alpha: float | None = None,
    span: float | None = None,
    halflife: float | None = None,
    bias: bool = False,
    min_periods: int = 0,
    nan_policy: NanPolicy = "skip",
    out: F64 | None = None,
) -> F64:
    """Exponentially weighted variance (pandas ``ewm(..., adjust=False).var(bias)``).

    The recurrence is pandas' ``ewmcov`` with ``x = y``. ``bias=False`` applies the
    unbiased weights correction ``(sum w)^2 / ((sum w)^2 - sum w^2)``, so the first output
    is NaN. Same smoothing and NaN rules as :func:`ewm_mean`. O(1) per element.

    Example:
        >>> import numpy as np, qkern
        >>> qkern.ewm_var(np.array([1.0, 2.0, 3.0]), alpha=0.5, bias=True)
        array([0.    , 0.25  , 0.6875])
    """
    arr, a, mp, policy = _ewm_args(x, alpha, span, halflife, min_periods, nan_policy)
    target = _out(out, arr)
    return _finish(_qkern.ewm_var(arr, a, bool(bias), mp, policy, target), target)


def ewm_std(
    x: ArrayIn,
    *,
    alpha: float | None = None,
    span: float | None = None,
    halflife: float | None = None,
    bias: bool = False,
    min_periods: int = 0,
    nan_policy: NanPolicy = "skip",
    out: F64 | None = None,
) -> F64:
    """Exponentially weighted standard deviation: the square root of :func:`ewm_var`.

    Example:
        >>> import numpy as np, qkern
        >>> qkern.ewm_std(np.array([1.0, 3.0]), alpha=0.5, bias=True)
        array([0., 1.])
    """
    arr, a, mp, policy = _ewm_args(x, alpha, span, halflife, min_periods, nan_policy)
    target = _out(out, arr)
    return _finish(_qkern.ewm_std(arr, a, bool(bias), mp, policy, target), target)


# ---------------------------------------------------------------------------------------
# Rolling extremes and regression
# ---------------------------------------------------------------------------------------


def rolling_min(
    x: ArrayIn,
    window: int,
    *,
    min_periods: int | None = None,
    nan_policy: NanPolicy = "skip",
    out: F64 | None = None,
) -> F64:
    """Rolling minimum of the valid values in each window.

    Algorithm: monotonic deque of (index, value) pairs (Lemire 2006): O(1) amortised and
    O(n) total per call, whatever the window. Infinities are ordered normally.

    Example:
        >>> import numpy as np, qkern
        >>> qkern.rolling_min(np.array([3.0, 1.0, 4.0, 1.0, 5.0]), 3)
        array([nan, nan,  1.,  1.,  1.])
    """
    arr, w, mp, policy = _rolling_args(x, window, min_periods, nan_policy)
    target = _out(out, arr)
    return _finish(_qkern.rolling_min(arr, w, mp, policy, target), target)


def rolling_max(
    x: ArrayIn,
    window: int,
    *,
    min_periods: int | None = None,
    nan_policy: NanPolicy = "skip",
    out: F64 | None = None,
) -> F64:
    """Rolling maximum of the valid values in each window (see :func:`rolling_min`).

    Example:
        >>> import numpy as np, qkern
        >>> qkern.rolling_max(np.array([3.0, 1.0, 4.0, 1.0, 5.0]), 3)
        array([nan, nan,  4.,  4.,  5.])
    """
    arr, w, mp, policy = _rolling_args(x, window, min_periods, nan_policy)
    target = _out(out, arr)
    return _finish(_qkern.rolling_max(arr, w, mp, policy, target), target)


def rolling_ols(
    y: ArrayIn,
    x: ArrayIn,
    window: int,
    *,
    min_periods: int | None = None,
    include_current: bool = True,
    nan_policy: NanPolicy = "skip",
) -> OLSResult:
    """Rolling regression of ``y`` on ``x`` with intercept, as an :class:`OLSResult`.

    ``beta = Sxy / Sxx``, ``alpha = mean_y - beta mean_x``, ``r2 = Sxy^2 / (Sxx Syy)`` and
    ``resid_t = y_t - alpha - beta x_t`` over the pairs in the window with no NaN on either
    side. With ``include_current=False`` all four outputs at row ``t`` use the previous
    window, so ``resid_t`` is out of sample. Needs ``min_periods`` valid pairs, at least two
    finite pairs and ``Sxx > 0``; NaN while an infinity is in the window. ``y`` may be 2-D
    ``(n, k)``; ``x`` then has the same shape or is 1-D of length ``n`` (every column of
    ``y`` regressed on the same ``x``). Sliding bivariate Welford, O(1) amortised.

    Example:
        >>> import numpy as np, qkern
        >>> x = np.array([0.0, 1.0, 2.0, 3.0])
        >>> qkern.rolling_ols(2.0 + 3.0 * x, x, 2).beta
        array([nan,  3.,  3.,  3.])
    """
    yarr = as_input(y, "y")
    xarr = as_input(x, "x")
    w = check_int(window, "window")
    mp = resolve_min_periods(min_periods, w)
    policy = check_nan_policy(nan_policy)
    alpha, beta, resid, r2 = _qkern.rolling_ols(yarr, xarr, w, mp, bool(include_current), policy)
    return OLSResult(alpha, beta, resid, r2)


# ---------------------------------------------------------------------------------------
# Build information
# ---------------------------------------------------------------------------------------


def build_info() -> dict[str, str | int | bool]:
    """Return build metadata: version, git sha, compiler, flags, ABI and GIL threshold."""
    info: dict[str, str | int | bool] = {}
    for key, value in _qkern._build_info().items():
        if not isinstance(value, str | int):  # bool is an int subclass
            raise TypeError(f"unexpected build_info value for {key!r}: {value!r}")
        info[key] = value
    return info


def show_config() -> None:
    """Print :func:`build_info` as aligned ``key : value`` lines."""
    info = build_info()
    width = max(len(k) for k in info)
    for key, value in info.items():
        print(f"{key:<{width}} : {value}")
