"""Streaming class API: constructors, properties, reset, repr, pickling and copies."""

from __future__ import annotations

import copy
import math
import pickle
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import qkern


def test_rolling_moments_properties_and_reset() -> None:
    r = qkern.RollingMoments(3, min_periods=2, ddof=0, nan_policy="propagate")
    assert (r.window, r.min_periods, r.ddof, r.nan_policy) == (3, 2, 0, "propagate")
    assert [r.ready, math.isnan(r.mean)] == [False, True]
    for v in (1.0, 2.0, 4.0):
        r.update(v)
    assert [r.ready] == [True]
    assert r.count == 3
    assert r.mean == pytest.approx(7 / 3)
    assert r.var == pytest.approx(14 / 9)
    assert r.std == pytest.approx(math.sqrt(14 / 9))
    assert r.zscore(r.mean) == pytest.approx(0.0)
    assert "RollingMoments(window=3, min_periods=2, ddof=0" in repr(r)
    r.reset()
    assert r.count == 0


def test_ewm_properties() -> None:
    e = qkern.EWM(span=9.0, bias=True, min_periods=2)
    assert e.alpha == pytest.approx(0.2)
    assert e.bias
    assert e.min_periods == 2
    assert e.nan_policy == "skip"
    e.update(1.0)
    assert [e.ready] == [False]
    e.update(2.0)
    assert [e.ready] == [True]
    assert e.count == 2
    assert e.mean == pytest.approx(1.2)
    assert e.std == pytest.approx(math.sqrt(e.var))
    assert "EWM(alpha=0.2" in repr(e)
    e.reset()
    assert e.count == 0
    assert qkern.EWM(halflife=1.0).alpha == pytest.approx(0.5)


def test_minmax_properties() -> None:
    m = qkern.RollingMinMax(2, min_periods=1)
    lo, hi = m.update_many(np.array([3.0, 1.0, 2.0]))
    np.testing.assert_array_equal(lo, [3.0, 1.0, 1.0])
    np.testing.assert_array_equal(hi, [3.0, 3.0, 2.0])
    assert (m.min, m.max, m.count, m.ready) == (1.0, 2.0, 2, True)
    assert (m.window, m.min_periods, m.nan_policy) == (2, 1, "skip")
    assert "RollingMinMax(window=2" in repr(m)
    m.reset()
    assert m.count == 0


def test_ols_properties_and_last_residual() -> None:
    o = qkern.RollingOLS(3, min_periods=2)
    o.update(1.0, 0.0)
    assert math.isnan(o.beta)
    o.update(3.0, 1.0)
    o.update(6.0, 2.0)
    res = qkern.rolling_ols(np.array([1.0, 3.0, 6.0]), np.array([0.0, 1.0, 2.0]), 3, min_periods=2)
    assert o.beta == res.beta[-1]
    assert o.alpha == res.alpha[-1]
    assert o.r2 == res.r2[-1]
    assert o.resid == res.resid[-1]
    assert (o.count, o.ready, o.window, o.min_periods, o.nan_policy) == (3, True, 3, 2, "skip")
    assert "RollingOLS(window=3" in repr(o)
    o.reset()
    assert math.isnan(o.resid)


def test_update_many_returns_named_tuples() -> None:
    m = qkern.RollingMoments(2).update_many(np.arange(3.0))
    assert isinstance(m, qkern.Moments)
    assert m.count.dtype == np.int64
    o = qkern.RollingOLS(2).update_many(np.arange(3.0), np.arange(3.0))
    assert isinstance(o, qkern.OLSResult)


FACTORIES: dict[str, Callable[[], Any]] = {
    "RollingMoments": lambda: qkern.RollingMoments(4, min_periods=2, ddof=0),
    "EWM": lambda: qkern.EWM(alpha=0.3, bias=True, min_periods=1, nan_policy="propagate"),
    "RollingMinMax": lambda: qkern.RollingMinMax(4, nan_policy="propagate"),
    "RollingOLS": lambda: qkern.RollingOLS(4, min_periods=3),
}


@pytest.mark.parametrize("name", list(FACTORIES))
def test_pickle_copy_and_deepcopy_round_trip(name: str) -> None:
    obj = FACTORIES[name]()
    state = obj.__getstate__()
    assert state["v"] == 1
    for clone in (pickle.loads(pickle.dumps(obj)), copy.copy(obj), copy.deepcopy(obj)):
        assert type(clone) is type(obj)
        assert repr(clone) == repr(obj)


def test_bad_state_is_rejected() -> None:
    r = qkern.RollingMoments(3)
    state = r.__getstate__()
    state["v"] = 99
    fresh = qkern.RollingMoments.__new__(qkern.RollingMoments)
    with pytest.raises(ValueError, match="version"):
        fresh.__setstate__(state)
    state = r.__getstate__()
    state["ints"] = [1, 2]
    with pytest.raises(ValueError, match="state"):
        qkern.RollingMoments.__new__(qkern.RollingMoments).__setstate__(state)
