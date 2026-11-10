"""Slow pure-Python reference kernels with qkern's exact semantics.

These are written for clarity, not speed: each output is computed from scratch over its
window with ``math.fsum``. Downstream projects may use them in their own tests.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _window(values: Sequence[float], t: int, window: int) -> Sequence[float]:
    return values[max(0, t - window + 1) : t + 1]


def rolling_mean(
    values: Sequence[float],
    window: int,
    *,
    min_periods: int | None = None,
    nan_policy: str = "skip",
) -> list[float]:
    """Reference rolling mean (spec section 8, DQ-02 to DQ-04)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    mp = window if min_periods is None else min_periods
    if not 1 <= mp <= window:
        raise ValueError("min_periods must satisfy 1 <= min_periods <= window")
    if nan_policy not in ("skip", "propagate", "raise"):
        raise ValueError(f"bad nan_policy {nan_policy!r}")
    xs = [float(v) for v in values]
    if nan_policy == "raise" and any(math.isnan(v) for v in xs):
        raise ValueError("input contains NaN and nan_policy='raise'")
    out: list[float] = []
    for t in range(len(xs)):
        win = _window(xs, t, window)
        valid = [v for v in win if not math.isnan(v)]
        has_nan = len(valid) < len(win)
        if (nan_policy == "propagate" and has_nan) or len(valid) < mp:
            out.append(math.nan)
            continue
        pos = any(v == math.inf for v in valid)
        neg = any(v == -math.inf for v in valid)
        if pos and neg:
            out.append(math.nan)
        elif pos:
            out.append(math.inf)
        elif neg:
            out.append(-math.inf)
        else:
            out.append(math.fsum(valid) / len(valid))
    return out
