"""Slow pure-Python reference kernels with qkern's exact semantics.

Written for clarity, not speed: each output is computed from scratch over its window with
``math.fsum`` (or, for the exponentially weighted kernels, from the explicit weights). They
follow qkern's rules where pandas differs (infinities are valid values, see the docs'
Semantics page), so downstream projects can use them as test oracles.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

__all__ = [
    "ewm_mean",
    "ewm_var",
    "rolling_max",
    "rolling_mean",
    "rolling_min",
    "rolling_ols",
    "rolling_std",
    "rolling_var",
    "rolling_zscore",
]

_POLICIES = ("skip", "propagate", "raise")


def _check(window: int, min_periods: int | None, nan_policy: str) -> int:
    if window < 1:
        raise ValueError("window must be >= 1")
    mp = window if min_periods is None else min_periods
    if not 1 <= mp <= window:
        raise ValueError("min_periods must satisfy 1 <= min_periods <= window")
    if nan_policy not in _POLICIES:
        raise ValueError(f"bad nan_policy {nan_policy!r}")
    return mp


def _floats(values: Iterable[float], nan_policy: str) -> list[float]:
    xs = [float(v) for v in values]
    if nan_policy == "raise" and any(math.isnan(v) for v in xs):
        raise ValueError("input contains NaN and nan_policy='raise'")
    return xs


def _windows(xs: list[float], window: int, mp: int, nan_policy: str) -> list[list[float] | None]:
    """Valid values of each window, or None where the output must be NaN."""
    result: list[list[float] | None] = []
    for t in range(len(xs)):
        win = xs[max(0, t - window + 1) : t + 1]
        valid = [v for v in win if not math.isnan(v)]
        blocked = nan_policy == "propagate" and len(valid) < len(win)
        result.append(None if blocked or len(valid) < mp else valid)
    return result


def _mean(valid: list[float]) -> float:
    pos = math.inf in valid
    neg = -math.inf in valid
    if pos and neg:
        return math.nan
    if pos:
        return math.inf
    if neg:
        return -math.inf
    return math.fsum(valid) / len(valid)


def _var(valid: list[float], ddof: int) -> float:
    if any(math.isinf(v) for v in valid) or len(valid) - ddof < 1:
        return math.nan
    m = math.fsum(valid) / len(valid)
    return math.fsum((v - m) ** 2 for v in valid) / (len(valid) - ddof)


def rolling_mean(
    values: Iterable[float],
    window: int,
    *,
    min_periods: int | None = None,
    nan_policy: str = "skip",
) -> list[float]:
    """Reference rolling mean (spec section 8, DQ-02 to DQ-04)."""
    mp = _check(window, min_periods, nan_policy)
    xs = _floats(values, nan_policy)
    return [math.nan if w is None else _mean(w) for w in _windows(xs, window, mp, nan_policy)]


def rolling_var(
    values: Iterable[float],
    window: int,
    *,
    min_periods: int | None = None,
    ddof: int = 1,
    nan_policy: str = "skip",
) -> list[float]:
    """Reference rolling variance (two-pass with math.fsum)."""
    mp = _check(window, min_periods, nan_policy)
    xs = _floats(values, nan_policy)
    return [math.nan if w is None else _var(w, ddof) for w in _windows(xs, window, mp, nan_policy)]


def rolling_std(
    values: Iterable[float],
    window: int,
    *,
    min_periods: int | None = None,
    ddof: int = 1,
    nan_policy: str = "skip",
) -> list[float]:
    """Reference rolling standard deviation."""
    var = rolling_var(values, window, min_periods=min_periods, ddof=ddof, nan_policy=nan_policy)
    return [math.sqrt(v) for v in var]


def rolling_zscore(
    values: Iterable[float],
    window: int,
    *,
    min_periods: int | None = None,
    ddof: int = 1,
    include_current: bool = True,
    nan_policy: str = "skip",
) -> list[float]:
    """Reference rolling z-score; without include_current the baseline is the previous window."""
    mp = _check(window, min_periods, nan_policy)
    xs = _floats(values, nan_policy)
    wins = _windows(xs, window, mp, nan_policy)
    out: list[float] = []
    for t, x in enumerate(xs):
        w = wins[t] if include_current else (wins[t - 1] if t > 0 else None)
        if math.isnan(x) or w is None:
            out.append(math.nan)
            continue
        s = math.sqrt(_var(w, ddof))
        out.append(math.nan if math.isnan(s) or s == 0.0 else (x - _mean(w)) / s)
    return out


def rolling_min(
    values: Iterable[float],
    window: int,
    *,
    min_periods: int | None = None,
    nan_policy: str = "skip",
) -> list[float]:
    """Reference rolling minimum."""
    mp = _check(window, min_periods, nan_policy)
    xs = _floats(values, nan_policy)
    return [math.nan if w is None else min(w) for w in _windows(xs, window, mp, nan_policy)]


def rolling_max(
    values: Iterable[float],
    window: int,
    *,
    min_periods: int | None = None,
    nan_policy: str = "skip",
) -> list[float]:
    """Reference rolling maximum."""
    mp = _check(window, min_periods, nan_policy)
    xs = _floats(values, nan_policy)
    return [math.nan if w is None else max(w) for w in _windows(xs, window, mp, nan_policy)]


def _ewm_weights(n: int, alpha: float) -> list[float]:
    """Weights of n observations for adjust=False: (1-a)^(n-1), then a (1-a)^(n-1-k)."""
    return [(1.0 if k == 0 else alpha) * (1.0 - alpha) ** (n - 1 - k) for k in range(n)]


def _ewm_moments(
    values: Iterable[float], alpha: float, nan_policy: str
) -> list[tuple[int, float, float, float]]:
    """(count, mean, biased variance, sum of squared weights) after each row."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must satisfy 0 < alpha <= 1")
    if nan_policy not in _POLICIES:
        raise ValueError(f"bad nan_policy {nan_policy!r}")
    xs = _floats(values, nan_policy)
    seen: list[float] = []
    poisoned = False
    out: list[tuple[int, float, float, float]] = []
    for x in xs:
        if math.isnan(x):
            poisoned = poisoned or nan_policy == "propagate"
        else:
            seen.append(x)
        if poisoned or not seen:
            out.append((len(seen), math.nan, math.nan, math.nan))
            continue
        w = _ewm_weights(len(seen), alpha)
        if any(math.isinf(v) for v in seen):
            # fsum rejects inf + -inf; plain summation gives the IEEE result (NaN).
            m = sum(wi * v for wi, v in zip(w, seen, strict=True))
            out.append((len(seen), m, math.nan, math.nan))
            continue
        m = math.fsum(wi * v for wi, v in zip(w, seen, strict=True))
        biased = math.fsum(wi * (v - m) ** 2 for wi, v in zip(w, seen, strict=True))
        out.append((len(seen), m, biased, math.fsum(wi * wi for wi in w)))
    return out


def ewm_mean(
    values: Iterable[float],
    alpha: float,
    *,
    min_periods: int = 0,
    nan_policy: str = "skip",
) -> list[float]:
    """Reference EWM mean (adjust=False, ignore_na=True) from explicit weights."""
    need = max(min_periods, 1)
    return [m if n >= need else math.nan for n, m, _, _ in _ewm_moments(values, alpha, nan_policy)]


def ewm_var(
    values: Iterable[float],
    alpha: float,
    *,
    bias: bool = False,
    min_periods: int = 0,
    nan_policy: str = "skip",
) -> list[float]:
    """Reference EWM variance (adjust=False, ignore_na=True) from explicit weights."""
    need = max(min_periods, 1)
    out: list[float] = []
    for n, _, biased, sw2 in _ewm_moments(values, alpha, nan_policy):
        if n < need or math.isnan(biased):
            out.append(math.nan)
        elif bias:
            out.append(biased)
        else:
            out.append(biased / (1.0 - sw2) if sw2 < 1.0 else math.nan)
    return out


def rolling_ols(
    y: Iterable[float],
    x: Iterable[float],
    window: int,
    *,
    min_periods: int | None = None,
    include_current: bool = True,
    nan_policy: str = "skip",
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Reference rolling OLS of y on x with intercept: (alpha, beta, resid, r2)."""
    mp = _check(window, min_periods, nan_policy)
    ys = _floats(y, nan_policy)
    xs = _floats(x, nan_policy)
    if len(ys) != len(xs):
        raise ValueError("y and x must have the same length")
    n = len(ys)

    def fit(t: int) -> tuple[float, float, float]:
        if t < 0:
            return math.nan, math.nan, math.nan
        lo = max(0, t - window + 1)
        pairs = [(a, b) for a, b in zip(ys[lo : t + 1], xs[lo : t + 1], strict=True)]
        valid = [(a, b) for a, b in pairs if not (math.isnan(a) or math.isnan(b))]
        blocked = nan_policy == "propagate" and len(valid) < len(pairs)
        finite = [(a, b) for a, b in valid if math.isfinite(a) and math.isfinite(b)]
        if blocked or len(valid) < mp or len(finite) < len(valid) or len(finite) < 2:
            return math.nan, math.nan, math.nan
        my = math.fsum(a for a, _ in finite) / len(finite)
        mx = math.fsum(b for _, b in finite) / len(finite)
        sxx = math.fsum((b - mx) ** 2 for _, b in finite)
        syy = math.fsum((a - my) ** 2 for a, _ in finite)
        sxy = math.fsum((b - mx) * (a - my) for a, b in finite)
        if not sxx > 0.0:
            return math.nan, math.nan, math.nan
        beta = sxy / sxx
        r2 = min((sxy / sxx) * (sxy / syy), 1.0) if syy > 0.0 else math.nan
        return my - beta * mx, beta, r2

    alpha_out: list[float] = []
    beta_out: list[float] = []
    resid_out: list[float] = []
    r2_out: list[float] = []
    for t in range(n):
        a, b, r = fit(t if include_current else t - 1)
        alpha_out.append(a)
        beta_out.append(b)
        r2_out.append(r)
        if math.isnan(ys[t]) or math.isnan(xs[t]) or math.isnan(b):
            resid_out.append(math.nan)
        else:
            resid_out.append(ys[t] - a - b * xs[t])
    return alpha_out, beta_out, resid_out, r2_out
