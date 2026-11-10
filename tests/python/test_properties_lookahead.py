"""No look-ahead (DQ-01, AC-W-14): perturbing x[t+1:] never changes out[:t+1], for every
kernel and both include_current settings."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from strategies import F64, series, window_for

import qkern


def _perturb(x: F64, t: int, data: st.DataObject) -> F64:
    tail = data.draw(series(min_size=len(x) - t - 1, max_size=len(x) - t - 1, allow_inf=True))
    y = x.copy()
    y[t + 1 :] = tail
    return y


def _same_prefix(a: F64, b: F64, t: int) -> None:
    np.testing.assert_array_equal(a[: t + 1], b[: t + 1])


@given(data=st.data())
def test_univariate_kernels_have_no_lookahead(data: st.DataObject) -> None:
    x = data.draw(series(min_size=1, allow_inf=True))
    window, mp = data.draw(window_for(len(x)))
    t = data.draw(st.integers(0, len(x) - 1))
    y = _perturb(x, t, data)
    for include_current in (True, False):
        _same_prefix(
            qkern.rolling_zscore(x, window, min_periods=mp, include_current=include_current),
            qkern.rolling_zscore(y, window, min_periods=mp, include_current=include_current),
            t,
        )
    for fn in (
        qkern.rolling_mean,
        qkern.rolling_var,
        qkern.rolling_std,
        qkern.rolling_min,
        qkern.rolling_max,
    ):
        _same_prefix(fn(x, window, min_periods=mp), fn(y, window, min_periods=mp), t)
    for efn in (qkern.ewm_mean, qkern.ewm_var, qkern.ewm_std):
        _same_prefix(efn(x, span=float(window)), efn(y, span=float(window)), t)


@given(data=st.data())
def test_rolling_ols_has_no_lookahead(data: st.DataObject) -> None:
    n = data.draw(st.integers(1, 150))
    yv = data.draw(series(min_size=n, max_size=n))
    xv = data.draw(series(min_size=n, max_size=n))
    window, mp = data.draw(window_for(n))
    t = data.draw(st.integers(0, n - 1))
    y2 = _perturb(yv, t, data)
    x2 = _perturb(xv, t, data)
    for include_current in (True, False):
        a = qkern.rolling_ols(yv, xv, window, min_periods=mp, include_current=include_current)
        b = qkern.rolling_ols(y2, x2, window, min_periods=mp, include_current=include_current)
        for fa, fb in zip(a, b, strict=True):
            _same_prefix(fa, fb, t)


@given(data=st.data())
def test_excluding_current_ignores_row_t(data: st.DataObject) -> None:
    """With include_current=False the statistics at t ignore x_t itself (only z uses it)."""
    x = data.draw(series(min_size=2))
    window, mp = data.draw(window_for(len(x)))
    t = data.draw(st.integers(0, len(x) - 1))
    y = x.copy()
    y[t] = 12345.0
    a = qkern.rolling_ols(x, x[::-1].copy(), window, min_periods=mp, include_current=False)
    b = qkern.rolling_ols(y, x[::-1].copy(), window, min_periods=mp, include_current=False)
    np.testing.assert_array_equal(a.beta[: t + 1], b.beta[: t + 1])
