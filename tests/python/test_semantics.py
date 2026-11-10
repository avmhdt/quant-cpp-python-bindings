"""One test per semantics rule (DQ-02 to DQ-06, DQ-11) and per error-table row (AC-W-16).

Test ids are cited on the docs' Semantics page next to each documented difference from
pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import qkern

NAN = np.nan
INF = np.inf


# --- DQ-02 NaN policy ------------------------------------------------------------------


def test_dq02_skip_uses_remaining_valid_values() -> None:
    x = np.array([1.0, NAN, 3.0, 5.0])
    np.testing.assert_array_equal(qkern.rolling_mean(x, 2, min_periods=1), [1, 1, 3, 4])
    # pandas-compatible for rolling kernels
    np.testing.assert_array_equal(
        qkern.rolling_mean(x, 2, min_periods=1), pd.Series(x).rolling(2, min_periods=1).mean()
    )


def test_dq02_propagate_while_nan_in_window() -> None:
    x = np.array([1.0, NAN, 3.0, 5.0, 7.0])
    np.testing.assert_array_equal(
        qkern.rolling_max(x, 2, min_periods=1, nan_policy="propagate"), [1, NAN, NAN, 5, 7]
    )


def test_dq02_raise_before_computing() -> None:
    out = np.full(3, 7.0)
    with pytest.raises(ValueError, match="nan_policy='raise'"):
        qkern.rolling_mean(np.array([1.0, 2.0, NAN]), 2, nan_policy="raise", out=out)
    np.testing.assert_array_equal(out, 7.0)  # nothing written


def test_dq02_ewm_skip_keeps_state_and_propagate_poisons() -> None:
    x = np.array([1.0, NAN, 3.0])
    np.testing.assert_array_equal(qkern.ewm_mean(x, alpha=0.5), [1.0, 1.0, 2.0])
    np.testing.assert_array_equal(
        qkern.ewm_mean(x, alpha=0.5, nan_policy="propagate"), [1.0, NAN, NAN]
    )


# --- DQ-03 min_periods -----------------------------------------------------------------


def test_dq03_default_min_periods_is_window() -> None:
    np.testing.assert_array_equal(qkern.rolling_mean(np.arange(4.0), 3), [NAN, NAN, 1, 2])


def test_dq03_ewm_default_min_periods_zero_behaves_as_one() -> None:
    np.testing.assert_array_equal(qkern.ewm_mean(np.array([NAN, 2.0]), alpha=0.5), [NAN, 2.0])


def test_dq03_variance_needs_n_minus_ddof_at_least_one() -> None:
    np.testing.assert_array_equal(qkern.rolling_var(np.array([5.0, NAN]), 2, min_periods=1), NAN)
    np.testing.assert_array_equal(
        qkern.rolling_var(np.array([5.0, NAN]), 2, min_periods=1, ddof=0), [0.0, 0.0]
    )


def test_dq03_ols_needs_two_pairs_and_positive_sxx() -> None:
    r = qkern.rolling_ols(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 2.0]), 2, min_periods=1)
    np.testing.assert_array_equal(r.beta, [NAN, NAN, 1.0])


# --- DQ-04 infinities (qkern differs from pandas, which treats inf as missing) -----------


def test_dq04_mean_with_one_or_both_infinity_signs() -> None:
    x = np.array([1.0, INF, -INF, 2.0, 3.0])
    np.testing.assert_array_equal(
        qkern.rolling_mean(x, 2, min_periods=1), [1.0, INF, NAN, -INF, 2.5]
    )


def test_dq04_variance_nan_while_inf_in_window_then_clean() -> None:
    x = np.array([1.0, 2.0, INF, 4.0, 5.0])
    np.testing.assert_array_equal(qkern.rolling_var(x, 2), [NAN, 0.5, NAN, NAN, 0.5])


def test_dq04_min_max_order_infinities() -> None:
    x = np.array([1.0, INF, -INF])
    np.testing.assert_array_equal(qkern.rolling_max(x, 2, min_periods=1), [1.0, INF, INF])
    np.testing.assert_array_equal(qkern.rolling_min(x, 2, min_periods=1), [1.0, 1.0, -INF])


def test_dq04_pandas_difference_is_documented() -> None:
    x = np.array([1.0, 2.0, INF, 4.0])
    assert np.isnan(pd.Series(x).rolling(2).max().iloc[2])  # pandas: missing
    assert qkern.rolling_max(x, 2)[2] == INF  # qkern: a valid value


# --- DQ-05 dtypes --------------------------------------------------------------------------


def test_dq05_float32_read_in_place_output_float64() -> None:
    out = qkern.rolling_mean(np.ones(4, dtype=np.float32), 2)
    assert out.dtype == np.float64


def test_dq05_integers_and_bools_converted() -> None:
    np.testing.assert_array_equal(qkern.rolling_mean(np.arange(3), 1), [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(qkern.rolling_mean(np.array([True, False]), 1), [1.0, 0.0])


def test_dq05_float16_and_longdouble_converted() -> None:
    np.testing.assert_array_equal(qkern.rolling_mean(np.ones(3, dtype=np.float16), 1), 1.0)
    np.testing.assert_array_equal(qkern.rolling_mean(np.ones(3, dtype=np.longdouble), 1), 1.0)


@pytest.mark.parametrize(
    "bad",
    [
        np.ones(3, dtype=np.complex128),
        np.array([1, 2], dtype=object),
        np.array(["a", "b"]),
        np.array(["2026-01-01"], dtype="datetime64[D]"),
        np.ones(3, dtype=">f8"),
    ],
    ids=["complex", "object", "str", "datetime", "big_endian"],
)
def test_dq05_unsupported_dtypes_raise_type_error(bad: np.ndarray) -> None:
    with pytest.raises(TypeError):
        qkern.rolling_mean(bad, 2)


# --- DQ-06 Arrow nulls: covered at M4 with the Arrow importer -----------------------------

# --- DQ-11 aliasing --------------------------------------------------------------------------


def test_dq11_out_overlapping_input_raises() -> None:
    buf = np.arange(12.0)
    with pytest.raises(ValueError, match="overlap"):
        qkern.rolling_mean(buf[:6], 2, out=buf[5:11])
    with pytest.raises(ValueError, match="overlap"):
        qkern.rolling_mean(buf[::2], 2, out=buf[1::2])  # interleaved spans overlap
    qkern.rolling_mean(buf[:6], 2, out=buf[6:12])


# --- Error table -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("call", "exc"),
    [
        (lambda: qkern.rolling_mean(np.ones(3), 0), ValueError),
        (lambda: qkern.rolling_mean(np.ones(3), 2, min_periods=3), ValueError),
        (lambda: qkern.rolling_mean(np.ones(3), 2, min_periods=0), ValueError),
        (lambda: qkern.rolling_var(np.ones(3), 2, ddof=-1), ValueError),
        (lambda: qkern.ewm_mean(np.ones(3)), ValueError),
        (lambda: qkern.ewm_mean(np.ones(3), alpha=0.5, span=3), ValueError),
        (lambda: qkern.ewm_mean(np.ones(3), alpha=1.5), ValueError),
        (lambda: qkern.ewm_mean(np.ones(3), span=0.5), ValueError),
        (lambda: qkern.ewm_mean(np.ones(3), halflife=0), ValueError),
        (lambda: qkern.ewm_mean(np.ones(3), alpha=0.5, min_periods=-1), ValueError),
        (lambda: qkern.rolling_ols(np.ones(3), np.ones(4), 2), ValueError),
        (lambda: qkern.rolling_mean(np.ones(3), 2, out=np.empty(4)), ValueError),
        (lambda: qkern.rolling_mean(np.array([NAN]), 1, nan_policy="raise"), ValueError),
        (lambda: qkern.rolling_mean(np.ones(3), 2, nan_policy="drop"), ValueError),  # type: ignore[arg-type]
        (lambda: qkern.rolling_mean(np.ones((2, 2, 2)), 2), TypeError),
        (lambda: qkern.rolling_mean(np.ones(3), 2.0), TypeError),  # type: ignore[arg-type]
        (lambda: qkern.rolling_mean(np.ones(3), True), TypeError),
        (lambda: qkern.rolling_mean(np.ones(3), 2, out=np.empty(3, np.float32)), TypeError),
        (lambda: qkern.rolling_mean(np.ones(3), 2, out=[0.0, 0.0, 0.0]), TypeError),  # type: ignore[arg-type]
        (lambda: qkern.RollingMoments(0), ValueError),
        (lambda: qkern.RollingMoments(3, min_periods=4), ValueError),
        (lambda: qkern.EWM(), ValueError),
        (lambda: qkern.EWM(alpha=0.2, halflife=3), ValueError),
        (lambda: qkern.RollingOLS(2, nan_policy="drop"), ValueError),
        (lambda: qkern.RollingMoments(2).update_many(np.ones((2, 2))), TypeError),
        (lambda: qkern.RollingOLS(2).update_many(np.ones(3), np.ones(2)), ValueError),
    ],
)
def test_error_table(call: object, exc: type[Exception]) -> None:
    with pytest.raises(exc):
        call()  # type: ignore[operator]


def test_read_only_out_raises_type_error() -> None:
    ro = np.empty(3)
    ro.flags.writeable = False
    with pytest.raises(TypeError, match="writeable"):
        qkern.rolling_mean(np.ones(3), 2, out=ro)
