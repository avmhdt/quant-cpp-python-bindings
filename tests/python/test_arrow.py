"""Arrow input through the PyCapsule interface (FR-12, DQ-06, AC-H-10): zero-copy for
pyarrow arrays and single-chunk Polars series, nulls exactly as NaN."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from hypothesis import given
from hypothesis import strategies as st
from strategies import assert_identical, series, window_for

import qkern
from qkern import _qkern

pl = pytest.importorskip("polars")


@pytest.fixture(autouse=True)
def _reset() -> None:
    _qkern._reset_debug_counters()


def _to_arrow(x: np.ndarray, dtype: pa.DataType) -> pa.Array:
    return pa.array(x, type=dtype, from_pandas=True)  # NaN becomes null


@given(data=st.data())
def test_pyarrow_nulls_equal_numpy_nan(data: st.DataObject) -> None:
    x = data.draw(series())
    window, mp = data.draw(window_for(len(x)))
    arr = _to_arrow(x, pa.float64())
    assert arr.null_count == int(np.isnan(x).sum())
    for fn in (qkern.rolling_mean, qkern.rolling_std, qkern.rolling_max, qkern.rolling_median):
        assert_identical(fn(arr, window, min_periods=mp), fn(x, window, min_periods=mp))
    assert_identical(qkern.ewm_var(arr, alpha=0.3), qkern.ewm_var(x, alpha=0.3))


@given(data=st.data())
def test_sliced_arrow_arrays_use_the_offset(data: st.DataObject) -> None:
    x = data.draw(series(min_size=2))
    start = data.draw(st.integers(0, len(x) - 1))
    arr = _to_arrow(x, pa.float64()).slice(start)
    assert_identical(
        qkern.rolling_mean(arr, 3, min_periods=1), qkern.rolling_mean(x[start:], 3, min_periods=1)
    )


def test_float32_arrow_equals_float32_numpy() -> None:
    x = np.array([1.5, np.nan, 2.25, 4.0], dtype=np.float32)
    arr = pa.array(x, type=pa.float32(), from_pandas=True)
    assert_identical(
        qkern.rolling_mean(arr, 2, min_periods=1), qkern.rolling_mean(x, 2, min_periods=1)
    )


def test_arrow_inputs_are_not_copied() -> None:
    x = np.random.default_rng(0).standard_normal(5000)
    qkern.rolling_std(pa.array(x), 20)
    qkern.rolling_std(_to_arrow(np.where(x > 1, np.nan, x), pa.float64()), 20)
    qkern.rolling_std(pl.Series(x), 20)
    qkern.RollingMoments(20).update_many(pa.array(x))
    qkern.rolling_ols(pa.array(x), pl.Series(x[::-1].copy()), 20)
    assert _qkern._debug_counters() == {"input_conversions": 0, "output_copies": 0}


def test_polars_series_with_nulls() -> None:
    s = pl.Series([1.0, None, 3.0, 5.0])
    np.testing.assert_array_equal(qkern.rolling_mean(s, 2, min_periods=1), [1.0, 1.0, 3.0, 4.0])
    np.testing.assert_array_equal(qkern.rolling_mean(pl.Series([], dtype=pl.Float64), 2), [])


def test_chunked_inputs_raise_type_error_with_hint() -> None:
    with pytest.raises(TypeError, match="combine_chunks"):
        qkern.rolling_mean(pa.chunked_array([[1.0], [2.0]]), 1)
    chunked = pl.concat([pl.Series([1.0]), pl.Series([2.0])], rechunk=False)
    if chunked.n_chunks() > 1:
        with pytest.raises(TypeError, match="rechunk"):
            qkern.rolling_mean(chunked, 1)
    assert_identical(qkern.rolling_mean(pa.chunked_array([[1.0, 2.0]]), 1), np.array([1.0, 2.0]))


@pytest.mark.parametrize("bad", [pa.array([1, 2, 3]), pa.array(["a", "b"]), pa.array([True])])
def test_unsupported_arrow_types_raise_type_error(bad: pa.Array) -> None:
    with pytest.raises(TypeError, match="Arrow format"):
        qkern.rolling_mean(bad, 2)


def test_bad_capsules_raise_type_error() -> None:
    with pytest.raises(TypeError, match="PyCapsule"):
        _qkern._ArrowInput.from_array(object(), object())
    with pytest.raises(TypeError, match="PyCapsule"):
        _qkern._ArrowInput.from_stream(object())


def test_out_with_arrow_input_and_overlap_rules() -> None:
    arr = pa.array([1.0, 2.0, 3.0])
    out = np.empty(3)
    assert qkern.rolling_mean(arr, 2, min_periods=1, out=out) is out
    np.testing.assert_array_equal(out, [1.0, 1.5, 2.5])
