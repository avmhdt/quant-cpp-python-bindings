"""Invariants that hold for any input (spec section 11 property 7)."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from strategies import series, window_for

import qkern


@given(data=st.data())
def test_min_mean_max_ordering_and_nonnegative_variance(data: st.DataObject) -> None:
    x = data.draw(series())
    window, mp = data.draw(window_for(len(x)))
    lo = qkern.rolling_min(x, window, min_periods=mp)
    hi = qkern.rolling_max(x, window, min_periods=mp)
    mean = qkern.rolling_mean(x, window, min_periods=mp)
    var = qkern.rolling_var(x, window, min_periods=mp)
    std = qkern.rolling_std(x, window, min_periods=mp)
    ok = ~np.isnan(mean)
    scale = 1e-12 * np.maximum(1.0, np.abs(mean[ok]))
    assert np.all(lo[ok] <= mean[ok] + scale)
    assert np.all(mean[ok] <= hi[ok] + scale)
    assert np.all(var[~np.isnan(var)] >= 0.0)
    np.testing.assert_array_equal(std, np.sqrt(var))
    m = qkern.rolling_moments(x, window, min_periods=mp)
    np.testing.assert_array_equal(m.mean, mean)
    np.testing.assert_array_equal(m.var, var)
    assert np.all(m.count <= window)
    assert np.all(m.count[ok] >= mp)


@given(data=st.data())
def test_ewm_std_is_sqrt_of_var_and_var_nonnegative(data: st.DataObject) -> None:
    x = data.draw(series())
    alpha = data.draw(st.floats(0.01, 1.0))
    var = qkern.ewm_var(x, alpha=alpha)
    assert np.all(var[~np.isnan(var)] >= 0.0)
    np.testing.assert_array_equal(qkern.ewm_std(x, alpha=alpha), np.sqrt(var))


@given(data=st.data())
def test_ols_r2_in_unit_interval(data: st.DataObject) -> None:
    n = data.draw(st.integers(0, 150))
    y = data.draw(series(min_size=n, max_size=n))
    x = data.draw(series(min_size=n, max_size=n))
    window, mp = data.draw(window_for(n))
    r2 = qkern.rolling_ols(y, x, window, min_periods=mp).r2
    ok = ~np.isnan(r2)
    assert np.all((r2[ok] >= 0.0) & (r2[ok] <= 1.0))
