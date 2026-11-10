"""Every kernel against the exact pure-Python reference (AC-W-05), including NaN and inf."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from strategies import EWM, F64, ROLLING, Unary, assert_close, series, window_for

import qkern
from qkern.testing import reference

POLICIES = st.sampled_from(["skip", "propagate"])


@pytest.mark.parametrize("k", ROLLING, ids=lambda k: k.name)
@given(data=st.data())
def test_rolling_kernels_match_reference(k: Unary, data: st.DataObject) -> None:
    x = data.draw(series(allow_inf=True))
    window, mp = data.draw(window_for(len(x)))
    policy = data.draw(POLICIES)
    kw = {"min_periods": mp, "nan_policy": policy, **k.extra}
    assert_close(k.batch(x, window, **kw), k.ref(x, window, **kw))


@pytest.mark.parametrize("k", EWM, ids=lambda k: k.name)
@given(data=st.data())
def test_ewm_kernels_match_reference(k: Unary, data: st.DataObject) -> None:
    x = data.draw(series(max_size=200))
    alpha = data.draw(st.floats(0.01, 1.0))
    mp = data.draw(st.integers(0, 10))
    policy = data.draw(POLICIES)
    kw = {"min_periods": mp, "nan_policy": policy, **k.extra}
    assert_close(k.batch(x, alpha=alpha, **kw), k.ref(x, alpha=alpha, **kw))


def _well_conditioned(y: F64, x: F64, window: int, include_current: bool) -> F64:
    """Rows whose fitting window has x spread well above rounding (Sxx relative to the
    window's sum of x^2 above 1e-8). Where x values differ only in their last bits, Sxx is
    pure rounding noise and no two algorithms (the fsum reference included) agree."""
    ok = np.zeros(len(y), dtype=bool)
    for t in range(len(y)):
        end = t if include_current else t - 1
        if end < 0:
            continue
        lo = max(0, end - window + 1)
        xs, ys = x[lo : end + 1], y[lo : end + 1]
        keep = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[keep]
        if xs.size < 2:
            ok[t] = True
            continue
        sxx = float(np.sum((xs - xs.mean()) ** 2))
        ok[t] = sxx == 0.0 or sxx > 1e-8 * float(np.sum(xs * xs))
    return ok


@given(data=st.data())
def test_rolling_ols_matches_reference(data: st.DataObject) -> None:
    n = data.draw(st.integers(0, 200))
    y = data.draw(series(min_size=n, max_size=n, max_abs=1e3))
    x = data.draw(series(min_size=n, max_size=n, max_abs=1e3))
    window, mp = data.draw(window_for(n))
    include_current = data.draw(st.booleans())
    got = qkern.rolling_ols(y, x, window, min_periods=mp, include_current=include_current)
    want = reference.rolling_ols(y, x, window, min_periods=mp, include_current=include_current)
    ok = _well_conditioned(y, x, window, include_current)
    for got_arr, want_list in zip(got, want, strict=True):
        w = np.asarray(want_list)
        np.testing.assert_array_equal(np.isnan(got_arr[ok]), np.isnan(w[ok]))
    assert_close(got.beta[ok], np.asarray(want[1])[ok], rtol=1e-7)
    assert_close(got.r2[ok], np.asarray(want[3])[ok], rtol=1e-7)
    # alpha and resid scale with |beta| * |x|; compare relative to that scale.
    scale = 1.0 + np.abs(np.asarray(want[1])) * float(np.nanmax(np.abs(x), initial=1.0))
    for i in (0, 2):
        g, w = got[i][ok], np.asarray(want[i])[ok]
        both = ~np.isnan(w)
        assert np.all(np.abs(g[both] - w[both]) <= 1e-7 * scale[ok][both])
