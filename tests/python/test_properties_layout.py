"""Input layout never changes results: float32 equals its float64 upcast, strided equals a
contiguous copy, integers equal their float64 values, and 2-D equals column by column
(spec section 11 properties 4 and 5, FR-10, FR-13)."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from strategies import F64, UNARY, Unary, assert_identical, series, window_for

import qkern


def _call(k: Unary, x: object, window: int, mp: int) -> F64:
    if k.kind == "ewm":
        return k.batch(x, span=float(window), **k.extra)
    return k.batch(x, window, min_periods=mp, **k.extra)


@given(data=st.data())
def test_float32_equals_float64_upcast(data: st.DataObject) -> None:
    x32 = data.draw(series(max_abs=1e4)).astype(np.float32)
    window, mp = data.draw(window_for(len(x32)))
    for k in UNARY:
        assert_identical(_call(k, x32, window, mp), _call(k, x32.astype(np.float64), window, mp))


@given(data=st.data())
def test_strided_equals_contiguous(data: st.DataObject) -> None:
    x = data.draw(series(max_size=600))
    step = data.draw(st.sampled_from([2, 3, -1, -2]))
    view = x[::step]
    window, mp = data.draw(window_for(len(view)))
    for k in UNARY:
        assert_identical(_call(k, view, window, mp), _call(k, view.copy(), window, mp))


@given(data=st.data())
def test_integers_and_bools_equal_float64(data: st.DataObject) -> None:
    ints = np.array(data.draw(st.lists(st.integers(-1000, 1000), max_size=200)), dtype=np.int64)
    window, mp = data.draw(window_for(len(ints)))
    for dtype in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint32, np.uint64):
        arr = np.abs(ints).astype(dtype) if np.dtype(dtype).kind == "u" else ints.astype(dtype)
        if np.dtype(dtype).itemsize == 1:
            arr = (ints % 100).astype(dtype)
        for k in UNARY:
            assert_identical(
                _call(k, arr, window, mp), _call(k, arr.astype(np.float64), window, mp)
            )
    bools = ints > 0
    for k in UNARY:
        assert_identical(
            _call(k, bools, window, mp), _call(k, bools.astype(np.float64), window, mp)
        )


@given(data=st.data())
def test_two_d_equals_column_by_column(data: st.DataObject) -> None:
    n = data.draw(st.integers(0, 120))
    k_cols = data.draw(st.integers(1, 4))
    cols = [data.draw(series(min_size=n, max_size=n)) for _ in range(k_cols)]
    order = data.draw(st.sampled_from(["C", "F"]))
    stacked = np.stack(cols, axis=1)
    mat = np.asfortranarray(stacked) if order == "F" else np.ascontiguousarray(stacked)
    window, mp = data.draw(window_for(n))
    for k in UNARY:
        got = _call(k, mat, window, mp)
        assert got.shape == mat.shape
        for j in range(k_cols):
            assert_identical(got[:, j], _call(k, mat[:, j].copy(), window, mp))


@given(data=st.data())
def test_two_d_ols_with_shared_and_per_column_x(data: st.DataObject) -> None:
    n = data.draw(st.integers(0, 100))
    ys = np.stack([data.draw(series(min_size=n, max_size=n)) for _ in range(3)], axis=1)
    xs = np.stack([data.draw(series(min_size=n, max_size=n)) for _ in range(3)], axis=1)
    window, mp = data.draw(window_for(n))
    shared = qkern.rolling_ols(ys, xs[:, 0], window, min_periods=mp)
    per_col = qkern.rolling_ols(ys, xs, window, min_periods=mp)
    for j in range(3):
        one = qkern.rolling_ols(ys[:, j].copy(), xs[:, 0].copy(), window, min_periods=mp)
        own = qkern.rolling_ols(ys[:, j].copy(), xs[:, j].copy(), window, min_periods=mp)
        for f in range(4):
            assert_identical(shared[f][:, j], one[f])
            assert_identical(per_col[f][:, j], own[f])
