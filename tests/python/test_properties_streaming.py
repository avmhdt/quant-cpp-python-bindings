"""Streaming objects fed any chunking give bit-identical results to the batch functions
(AC-W-06, FR-08), and a pickled object continues bit-identically (FR-09)."""

from __future__ import annotations

import copy
import itertools
import pickle
from typing import Any

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from strategies import F64, assert_identical, chunk_bounds, series, window_for

import qkern


def _feed(obj: Any, x: F64, bounds: list[int], use_scalar: bool) -> list[Any]:
    """Feed x in chunks through update_many (or update for 1-element chunks)."""
    outs: list[F64] = []
    for lo, hi in itertools.pairwise(bounds):
        if use_scalar and hi - lo == 1:
            obj.update(float(x[lo]))
            continue
        outs.append(obj.update_many(x[lo:hi]))
    return outs


@given(data=st.data())
def test_rolling_moments_chunked_equals_batch(data: st.DataObject) -> None:
    x = data.draw(series(allow_inf=True))
    window, mp = data.draw(window_for(len(x)))
    ddof = data.draw(st.integers(0, 2))
    bounds = data.draw(chunk_bounds(len(x)))
    obj = qkern.RollingMoments(window, min_periods=mp, ddof=ddof)
    parts = [obj.update_many(x[lo:hi]) for lo, hi in itertools.pairwise(bounds)]
    batch = qkern.rolling_moments(x, window, min_periods=mp, ddof=ddof)
    assert_identical(np.concatenate([p.mean for p in parts]), batch.mean)
    assert_identical(np.concatenate([p.var for p in parts]), batch.var)
    np.testing.assert_array_equal(np.concatenate([p.count for p in parts]), batch.count)
    if len(x):
        assert_identical(np.array([obj.var]), batch.var[-1:])


@given(data=st.data())
def test_rolling_moments_scalar_updates_equal_batch(data: st.DataObject) -> None:
    x = data.draw(series(max_size=100, allow_inf=True))
    window, mp = data.draw(window_for(len(x)))
    obj = qkern.RollingMoments(window, min_periods=mp)
    means, stds, zs = [], [], []
    zs_oos = []
    for v in x:
        zs_oos.append(obj.zscore(float(v)))
        obj.update(float(v))
        means.append(obj.mean)
        stds.append(obj.std)
        zs.append(obj.zscore(float(v)))
    assert_identical(np.array(means), qkern.rolling_mean(x, window, min_periods=mp))
    assert_identical(np.array(stds), qkern.rolling_std(x, window, min_periods=mp))
    assert_identical(np.array(zs), qkern.rolling_zscore(x, window, min_periods=mp))
    assert_identical(
        np.array(zs_oos),
        qkern.rolling_zscore(x, window, min_periods=mp, include_current=False),
    )


@given(data=st.data())
def test_ewm_chunked_equals_batch(data: st.DataObject) -> None:
    x = data.draw(series())
    alpha = data.draw(st.floats(0.01, 1.0))
    mp = data.draw(st.integers(0, 5))
    bias = data.draw(st.booleans())
    bounds = data.draw(chunk_bounds(len(x)))
    obj = qkern.EWM(alpha=alpha, min_periods=mp, bias=bias)
    parts = _feed(obj, x, bounds, use_scalar=False)
    means = np.concatenate([p[0] for p in parts]) if parts else np.empty(0)
    vars_ = np.concatenate([p[1] for p in parts]) if parts else np.empty(0)
    assert_identical(means, qkern.ewm_mean(x, alpha=alpha, min_periods=mp))
    assert_identical(vars_, qkern.ewm_var(x, alpha=alpha, min_periods=mp, bias=bias))


@given(data=st.data())
def test_minmax_chunked_equals_batch(data: st.DataObject) -> None:
    x = data.draw(series(allow_inf=True))
    window, mp = data.draw(window_for(len(x)))
    bounds = data.draw(chunk_bounds(len(x)))
    obj = qkern.RollingMinMax(window, min_periods=mp)
    parts = _feed(obj, x, bounds, use_scalar=False)
    lo = np.concatenate([p[0] for p in parts]) if parts else np.empty(0)
    hi = np.concatenate([p[1] for p in parts]) if parts else np.empty(0)
    assert_identical(lo, qkern.rolling_min(x, window, min_periods=mp))
    assert_identical(hi, qkern.rolling_max(x, window, min_periods=mp))


@given(data=st.data())
def test_ols_chunked_equals_batch(data: st.DataObject) -> None:
    n = data.draw(st.integers(0, 200))
    y = data.draw(series(min_size=n, max_size=n))
    x = data.draw(series(min_size=n, max_size=n))
    window, mp = data.draw(window_for(n))
    bounds = data.draw(chunk_bounds(n))
    obj = qkern.RollingOLS(window, min_periods=mp)
    parts = [obj.update_many(y[lo:hi], x[lo:hi]) for lo, hi in itertools.pairwise(bounds)]
    batch = qkern.rolling_ols(y, x, window, min_periods=mp)
    for field in ("alpha", "beta", "resid", "r2"):
        assert_identical(np.concatenate([getattr(p, field) for p in parts]), getattr(batch, field))
    if n:
        assert_identical(np.array([obj.resid]), batch.resid[-1:])


@given(data=st.data())
def test_pickled_objects_continue_identically(data: st.DataObject) -> None:
    x = data.draw(series(min_size=1, allow_inf=True))
    window, mp = data.draw(window_for(len(x)))
    cut = data.draw(st.integers(0, len(x)))
    objects: list[Any] = [
        qkern.RollingMoments(window, min_periods=mp),
        qkern.EWM(span=float(window), min_periods=mp),
        qkern.RollingMinMax(window, min_periods=mp),
        qkern.RollingOLS(window, min_periods=mp),
    ]
    for obj in objects:
        is_ols = isinstance(obj, qkern.RollingOLS)
        head = (x[:cut], x[::-1][:cut]) if is_ols else (x[:cut],)
        tail = (x[cut:], x[::-1][cut:]) if is_ols else (x[cut:],)
        obj.update_many(*head)
        restored = pickle.loads(pickle.dumps(obj))
        cloned = copy.deepcopy(obj)
        expected = obj.update_many(*tail)
        for other in (restored, cloned):
            got = other.update_many(*tail)
            for a, b in zip(got, expected, strict=True):
                np.testing.assert_array_equal(a, b)


@given(data=st.data())
def test_repeat_runs_are_bit_identical(data: st.DataObject) -> None:
    x = data.draw(series())
    window, mp = data.draw(window_for(len(x)))
    first = qkern.rolling_zscore(x, window, min_periods=mp)
    for _ in range(3):
        assert_identical(qkern.rolling_zscore(x, window, min_periods=mp), first)
