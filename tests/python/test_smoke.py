"""M1 smoke tests: the extension imports, rolling_mean works end to end, metadata is sane."""

from __future__ import annotations

import importlib.metadata
import pickle

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import qkern
from qkern import _qkern
from qkern.testing import reference


def _pandas_mean(
    x: npt.NDArray[np.float64], window: int, min_periods: int
) -> npt.NDArray[np.float64]:
    result: npt.NDArray[np.float64] = (
        pd.Series(x).rolling(window, min_periods=min_periods).mean().to_numpy()
    )
    return result


def test_version_matches_distribution_metadata() -> None:
    assert qkern.__version__ == importlib.metadata.version("qkern")
    assert qkern.build_info()["version"] == qkern.__version__


def test_build_info_fields() -> None:
    info = qkern.build_info()
    for key in (
        "version",
        "git_sha",
        "compiler",
        "compiler_version",
        "cxx_standard",
        "build_type",
        "fp_flags",
        "nanobind_version",
        "abi3",
        "python_build_version",
        "gil_release_min_n",
        "free_threaded",
    ):
        assert key in info
    assert info["gil_release_min_n"] == 1024
    assert isinstance(info["abi3"], bool)
    assert int(info["cxx_standard"]) >= 202002


def test_show_config_prints(capsys: pytest.CaptureFixture[str]) -> None:
    qkern.show_config()
    out = capsys.readouterr().out
    assert "version" in out
    assert "nanobind_version" in out


def test_rolling_mean_known_values() -> None:
    out = qkern.rolling_mean(np.array([1.0, 2.0, 3.0, 4.0]), 2)
    np.testing.assert_array_equal(out, [np.nan, 1.5, 2.5, 3.5])
    assert out.dtype == np.float64


def test_rolling_mean_matches_pandas_and_reference() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2_000)
    x[rng.choice(2_000, 100, replace=False)] = np.nan
    for window, mp in [(1, 1), (5, 3), (20, 20), (300, 50)]:
        got = qkern.rolling_mean(x, window, min_periods=mp)
        np.testing.assert_allclose(got, _pandas_mean(x, window, mp), rtol=1e-12, atol=1e-12)
        ref = np.array(reference.rolling_mean(x.tolist(), window, min_periods=mp))
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


@given(
    arrays(
        np.float64,
        st.integers(0, 300),
        elements=st.one_of(st.floats(-1e6, 1e6), st.just(np.nan)),
    ),
    st.integers(1, 40),
    st.data(),
)
def test_rolling_mean_property_vs_pandas(
    x: npt.NDArray[np.float64], window: int, data: st.DataObject
) -> None:
    mp = data.draw(st.integers(1, window))
    got = qkern.rolling_mean(x, window, min_periods=mp)
    want = _pandas_mean(x, window, mp)
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


def test_strided_and_float32_inputs_equal_contiguous_copies() -> None:
    x = np.random.default_rng(1).standard_normal(3_000)
    for view in (x[::3], x[::-1], x.reshape(1000, 3)[:, 1]):
        np.testing.assert_array_equal(
            qkern.rolling_mean(view, 10), qkern.rolling_mean(view.copy(), 10)
        )
    x32 = x.astype(np.float32)
    np.testing.assert_array_equal(
        qkern.rolling_mean(x32, 10), qkern.rolling_mean(x32.astype(np.float64), 10)
    )


def test_integer_and_bool_inputs_are_converted() -> None:
    np.testing.assert_array_equal(qkern.rolling_mean(np.arange(5), 2), [np.nan, 0.5, 1.5, 2.5, 3.5])
    np.testing.assert_array_equal(
        qkern.rolling_mean(np.array([True, False, True]), 1), [1.0, 0.0, 1.0]
    )


def test_out_parameter_is_filled_and_returned() -> None:
    x = np.arange(10, dtype=np.float64)
    out = np.empty(10)
    result = qkern.rolling_mean(x, 3, out=out)
    assert result is out
    np.testing.assert_array_equal(out, qkern.rolling_mean(x, 3))
    strided_out = np.zeros(20)[::2]
    qkern.rolling_mean(x, 3, out=strided_out)
    np.testing.assert_array_equal(strided_out, out)


def test_output_owns_a_capsule_buffer() -> None:
    x = np.ones(100)
    out = qkern.rolling_mean(x, 3)
    assert out.flags.writeable
    assert not np.shares_memory(out, x)
    assert out.base is not None


def test_nan_policies() -> None:
    x = np.array([1.0, np.nan, 3.0, 5.0, 7.0])
    np.testing.assert_array_equal(qkern.rolling_mean(x, 2, min_periods=1), [1, 1, 3, 4, 6])
    np.testing.assert_array_equal(
        qkern.rolling_mean(x, 2, min_periods=1, nan_policy="propagate"),
        [1, np.nan, np.nan, 4, 6],
    )
    with pytest.raises(ValueError, match="raise"):
        qkern.rolling_mean(x, 2, nan_policy="raise")


def test_infinities() -> None:
    x = np.array([1.0, np.inf, -np.inf, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(
        qkern.rolling_mean(x, 2, min_periods=1), [1, np.inf, np.nan, -np.inf, 2.5, 3.5]
    )


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"window": 0}, ValueError),
        ({"window": -3}, ValueError),
        ({"window": 2.5}, TypeError),
        ({"window": True}, TypeError),
        ({"window": 3, "min_periods": 0}, ValueError),
        ({"window": 3, "min_periods": 4}, ValueError),
        ({"window": 3, "min_periods": 1.0}, TypeError),
        ({"window": 3, "nan_policy": "ignore"}, ValueError),
    ],
)
def test_argument_errors(kwargs: dict[str, object], exc: type[Exception]) -> None:
    with pytest.raises(exc):
        qkern.rolling_mean(np.ones(5), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad",
    [
        np.ones(4, dtype=np.complex128),
        np.array(["a", "b"], dtype=object),
        np.array(["2026-01-01"], dtype="datetime64[D]"),
        np.ones(4, dtype=">f8"),
        np.ones((2, 2)),
    ],
)
def test_unsupported_inputs_raise_type_error(bad: npt.NDArray[np.generic]) -> None:
    with pytest.raises(TypeError):
        qkern.rolling_mean(bad, 2)  # type: ignore[arg-type]


def test_out_errors() -> None:
    x = np.arange(6, dtype=np.float64)
    with pytest.raises(TypeError):
        qkern.rolling_mean(x, 2, out=np.empty(6, dtype=np.float32))
    with pytest.raises(ValueError, match="length"):
        qkern.rolling_mean(x, 2, out=np.empty(5))
    ro = np.empty(6)
    ro.flags.writeable = False
    with pytest.raises(TypeError, match="writeable"):
        qkern.rolling_mean(x, 2, out=ro)
    with pytest.raises(ValueError, match="overlap"):
        qkern.rolling_mean(x, 2, out=x)
    buf = np.arange(12, dtype=np.float64)
    with pytest.raises(ValueError, match="overlap"):
        qkern.rolling_mean(buf[:6], 2, out=buf[5:11])
    qkern.rolling_mean(buf[:6], 2, out=buf[6:12])


def test_gil_threshold_hook_round_trips() -> None:
    assert _qkern._get_gil_release_min_n() == 1024
    try:
        _qkern._set_gil_release_min_n(2**62)
        assert _qkern._get_gil_release_min_n() == 2**62
        np.testing.assert_array_equal(qkern.rolling_mean(np.ones(5000), 3)[2:], 1.0)
    finally:
        _qkern._set_gil_release_min_n(1024)


def test_debug_counters_hook() -> None:
    _qkern._reset_debug_counters()
    assert _qkern._debug_counters() == {"input_conversions": 0, "output_copies": 0}


def test_reference_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="window"):
        reference.rolling_mean([1.0], 0)
    with pytest.raises(ValueError, match="min_periods"):
        reference.rolling_mean([1.0], 2, min_periods=3)
    with pytest.raises(ValueError, match="nan_policy"):
        reference.rolling_mean([1.0], 1, nan_policy="drop")
    with pytest.raises(ValueError, match="raise"):
        reference.rolling_mean([float("nan")], 1, nan_policy="raise")
    assert (
        reference.rolling_mean([1.0, float("nan")], 2, min_periods=1, nan_policy="propagate")[1]
        != reference.rolling_mean([1.0, float("nan")], 2, min_periods=1)[1]
    )


def test_results_are_picklable_arrays() -> None:
    out = qkern.rolling_mean(np.arange(4.0), 2)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(out)), out)
