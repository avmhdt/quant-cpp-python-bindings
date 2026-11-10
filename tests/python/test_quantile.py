"""Rolling quantile and median (FR-06, AC-H-10): pandas for all five interpolation modes,
streaming parity, pickling and argument errors."""

from __future__ import annotations

import itertools
import pickle
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from strategies import assert_close, assert_identical, chunk_bounds, series, window_for

import qkern
from qkern.testing import reference

MODES = ["linear", "lower", "higher", "nearest", "midpoint"]


@pytest.mark.parametrize("interpolation", MODES)
@given(data=st.data())
def test_matches_pandas_for_every_interpolation(interpolation: str, data: st.DataObject) -> None:
    x = data.draw(series())
    window, mp = data.draw(window_for(len(x)))
    q = data.draw(st.sampled_from([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]) | st.floats(0.0, 1.0))
    got = qkern.rolling_quantile(x, window, q, min_periods=mp, interpolation=interpolation)  # type: ignore[arg-type]
    roll: Any = pd.Series(x).rolling(window, min_periods=mp)
    want = roll.quantile(q, interpolation=interpolation).to_numpy()
    # Linear interpolation lo + (hi - lo) * frac rounds once on the scale of the data. pandas'
    # compiled code may fuse that multiply-add (FMA, seen on macOS arm64) where qkern never
    # does (DQ-08), so allow one rounding relative to the data's magnitude.
    scale = max(1.0, float(np.nanmax(np.abs(x), initial=0.0)))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=4e-16 * scale)


@given(data=st.data())
def test_median_matches_pandas_and_reference(data: st.DataObject) -> None:
    x = data.draw(series(allow_inf=False))
    window, mp = data.draw(window_for(len(x)))
    got = qkern.rolling_median(x, window, min_periods=mp)
    assert_close(got, pd.Series(x).rolling(window, min_periods=mp).median().to_numpy())
    assert_close(got, reference.rolling_quantile(x, window, 0.5, min_periods=mp))


@given(data=st.data())
def test_streaming_chunks_equal_batch(data: st.DataObject) -> None:
    x = data.draw(series(allow_inf=True))
    window, mp = data.draw(window_for(len(x)))
    q = data.draw(st.floats(0.0, 1.0))
    mode = data.draw(st.sampled_from(MODES))
    bounds = data.draw(chunk_bounds(len(x)))
    obj = qkern.RollingQuantile(window, q, interpolation=mode, min_periods=mp)
    parts = [obj.update_many(x[lo:hi]) for lo, hi in itertools.pairwise(bounds)]
    batch = qkern.rolling_quantile(x, window, q, min_periods=mp, interpolation=mode)  # type: ignore[arg-type]
    assert_identical(np.concatenate(parts), batch)
    cut = data.draw(st.integers(0, len(x)))
    a = qkern.RollingQuantile(window, q, interpolation=mode, min_periods=mp)
    a.update_many(x[:cut])
    b = pickle.loads(pickle.dumps(a))
    assert_identical(b.update_many(x[cut:]), a.update_many(x[cut:]))


def test_streaming_properties() -> None:
    r = qkern.RollingQuantile(3, 0.25, interpolation="higher", min_periods=2)
    assert (r.window, r.q, r.interpolation, r.min_periods, r.nan_policy) == (
        3,
        0.25,
        "higher",
        2,
        "skip",
    )
    r.update(4.0)
    assert np.isnan(r.value)
    r.update(1.0)
    assert r.value == 4.0
    assert (r.count, r.ready) == (2, True)
    assert "RollingQuantile(window=3, q=0.25" in repr(r)
    r.reset()
    assert r.count == 0


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"q": 1.5}, ValueError),
        ({"q": -0.1}, ValueError),
        ({"q": 0.5, "interpolation": "cubic"}, ValueError),
        ({"q": 0.5, "window": 0}, ValueError),
    ],
)
def test_argument_errors(kwargs: dict[str, object], exc: type[Exception]) -> None:
    args: dict[str, object] = {"window": 3, **kwargs}
    with pytest.raises(exc):
        qkern.rolling_quantile(np.ones(5), **args)  # type: ignore[arg-type]
    with pytest.raises(exc):
        qkern.RollingQuantile(**args)  # type: ignore[arg-type]
