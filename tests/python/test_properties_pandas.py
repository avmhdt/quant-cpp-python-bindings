"""Kernels against pandas on bounded hypothesis data, |x| <= 1e6 (NFR-06 (d), AC-W-05).

pandas treats +-inf as missing, so these properties use finite data with NaN; infinity
semantics are covered against the reference and in test_semantics.py.

pandas' own sliding sums keep rounding residue after a large value leaves the window (for
example std 2.8e-7 for the window [0, 0, 0] after 46 left it). Where qkern and pandas differ
by more than the tolerance, qkern must then agree with the exact math.fsum reference: the
difference is pandas' error, listed on the docs' Semantics page.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st
from strategies import F64, assert_close, series, window_for

import qkern
from qkern.testing import reference


def assert_matches_pandas(got: F64, pandas: F64, exact: list[float], rtol: float = 1e-9) -> None:
    """Within rtol of pandas, or (where pandas is off) within rtol of the exact reference."""
    ex = np.asarray(exact, dtype=np.float64)
    tol = rtol * np.maximum(1.0, np.abs(pandas))
    near_pandas = (np.abs(got - pandas) <= tol) | (np.isnan(got) & np.isnan(pandas))
    if not np.all(near_pandas):
        idx = ~near_pandas
        assert_close(got[idx], ex[idx], rtol=rtol)
        # pandas must really be the one that is off.
        assert np.all(np.abs(pandas[idx] - ex[idx]) > np.abs(got[idx] - ex[idx]))


def _roll(x: F64, window: int, mp: int) -> Any:
    return pd.Series(x).rolling(window, min_periods=mp)


@given(data=st.data())
def test_rolling_moments_match_pandas(data: st.DataObject) -> None:
    x = data.draw(series())
    window, mp = data.draw(window_for(len(x)))
    ddof = data.draw(st.integers(0, 2))
    r = _roll(x, window, mp)
    assert_matches_pandas(
        qkern.rolling_mean(x, window, min_periods=mp),
        r.mean().to_numpy(),
        reference.rolling_mean(x, window, min_periods=mp),
    )
    assert_matches_pandas(
        qkern.rolling_var(x, window, min_periods=mp, ddof=ddof),
        r.var(ddof=ddof).to_numpy(),
        reference.rolling_var(x, window, min_periods=mp, ddof=ddof),
    )
    assert_matches_pandas(
        qkern.rolling_std(x, window, min_periods=mp, ddof=ddof),
        r.std(ddof=ddof).to_numpy(),
        reference.rolling_std(x, window, min_periods=mp, ddof=ddof),
    )


@given(data=st.data())
def test_rolling_min_max_match_pandas(data: st.DataObject) -> None:
    x = data.draw(series())
    window, mp = data.draw(window_for(len(x)))
    r = _roll(x, window, mp)
    np.testing.assert_array_equal(qkern.rolling_min(x, window, min_periods=mp), r.min())
    np.testing.assert_array_equal(qkern.rolling_max(x, window, min_periods=mp), r.max())


@given(data=st.data())
def test_ewm_matches_pandas(data: st.DataObject) -> None:
    x = data.draw(series(max_size=200))
    alpha = data.draw(st.floats(0.01, 1.0))
    mp = data.draw(st.integers(0, 10))
    bias = data.draw(st.booleans())
    e = pd.Series(x).ewm(alpha=alpha, adjust=False, ignore_na=True, min_periods=mp)
    var_exact = reference.ewm_var(x, alpha, min_periods=mp, bias=bias)
    assert_matches_pandas(
        qkern.ewm_mean(x, alpha=alpha, min_periods=mp),
        e.mean().to_numpy(),
        reference.ewm_mean(x, alpha, min_periods=mp),
    )
    assert_matches_pandas(
        qkern.ewm_var(x, alpha=alpha, min_periods=mp, bias=bias),
        e.var(bias=bias).to_numpy(),
        var_exact,
    )
    assert_matches_pandas(
        qkern.ewm_std(x, alpha=alpha, min_periods=mp, bias=bias),
        e.std(bias=bias).to_numpy(),
        list(np.sqrt(np.asarray(var_exact))),
    )


@given(data=st.data())
def test_zscore_matches_pandas_expression(data: st.DataObject) -> None:
    x = data.draw(series())
    window, mp = data.draw(window_for(len(x)))
    include_current = data.draw(st.booleans())
    r = _roll(x, window, mp)
    mean, sd = r.mean(), r.std()
    if not include_current:
        mean, sd = mean.shift(1), sd.shift(1)
    want = ((pd.Series(x) - mean) / sd).to_numpy()
    got = qkern.rolling_zscore(x, window, min_periods=mp, include_current=include_current)
    exact = np.asarray(
        reference.rolling_zscore(x, window, min_periods=mp, include_current=include_current)
    )
    # pandas gives +-inf (or noise) where the window is constant; qkern defines z as NaN
    # there. Compare where pandas' standard deviation is clearly non-zero.
    ok = (sd.to_numpy() > 1e-6 * max(1.0, float(np.nanmax(np.abs(x), initial=0.0)))) | np.isnan(
        sd.to_numpy()
    )
    assert_matches_pandas(got[ok], want[ok], list(exact[ok]), rtol=1e-8)


@given(data=st.data())
def test_rolling_ols_beta_matches_pandas(data: st.DataObject) -> None:
    n = data.draw(st.integers(0, 200))
    x = data.draw(series(min_size=n, max_size=n, max_abs=1e3))
    noise = data.draw(series(min_size=n, max_size=n, max_abs=1.0))
    y = 0.5 + 1.5 * x + noise
    window, mp = data.draw(window_for(n))
    window = max(window, 2)
    mp = min(max(mp, 2), window)
    got = qkern.rolling_ols(y, x, window, min_periods=mp)
    ys, xs = pd.Series(y), pd.Series(x)
    cov = ys.rolling(window, min_periods=mp).cov(xs)
    var = xs.where(~ys.isna()).rolling(window, min_periods=mp).var()
    want = (cov / var).to_numpy()
    # pandas computes cov as E[xy] - E[x]E[y]; compare where x varies enough for that
    # formula to be accurate.
    ok = (var.to_numpy() > 1e-3) | np.isnan(want)
    ok &= ~(np.isnan(got.beta) & ~np.isnan(want))  # Sxx > 0 is decided exactly by qkern
    assert_close(got.beta[ok], want[ok], rtol=1e-6)
