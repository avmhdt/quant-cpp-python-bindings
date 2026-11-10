"""Validation against published worked examples (spec section 11)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import qkern


def test_welford_textbook_variance() -> None:
    # The classic example data set (Welford 1962 / Knuth TAOCP vol. 2, 4.2.2): sample
    # variance of 2, 4, 4, 4, 5, 5, 7, 9 is 32 / 7, population variance 4.
    x = np.array([2.0, 4, 4, 4, 5, 5, 7, 9])
    assert qkern.rolling_var(x, 8)[-1] == pytest.approx(32 / 7, rel=1e-15)
    assert qkern.rolling_var(x, 8, ddof=0)[-1] == pytest.approx(4.0, rel=1e-15)
    assert qkern.rolling_std(x, 8, ddof=0)[-1] == pytest.approx(2.0, rel=1e-15)
    assert qkern.rolling_mean(x, 8)[-1] == pytest.approx(5.0, rel=1e-15)


def test_lemire_style_worked_example() -> None:
    # A running max/min filter over w = 3 on a small sequence with ties and reversals.
    x = np.array([2.0, 1, 3, 3, 0, 5, 4, 4, 1])
    np.testing.assert_array_equal(qkern.rolling_max(x, 3)[2:], [3, 3, 3, 5, 5, 5, 4])
    np.testing.assert_array_equal(qkern.rolling_min(x, 3)[2:], [1, 1, 0, 0, 0, 4, 1])


def test_pandas_documentation_examples() -> None:
    # pandas window user guide: s = Series(range(5)); s.rolling(2).sum() -> mean = sum / 2.
    s = np.arange(5.0)
    np.testing.assert_array_equal(qkern.rolling_mean(s, 2), [np.nan, 0.5, 1.5, 2.5, 3.5])
    # pandas DataFrame.ewm docs, B = [0, 1, 2, NaN, 4], com=0.5 (alpha = 2/3), adjust=False,
    # with ignore_na=True (qkern's NaN rule).
    b = np.array([0.0, 1.0, 2.0, np.nan, 4.0])
    want = pd.Series(b).ewm(com=0.5, adjust=False, ignore_na=True).mean().to_numpy()
    np.testing.assert_allclose(qkern.ewm_mean(b, alpha=2 / 3), want, rtol=1e-15)


def test_rolling_ols_exact_line() -> None:
    x = np.arange(10.0)
    r = qkern.rolling_ols(1.0 + 2.0 * x, x, 4)
    np.testing.assert_allclose(r.beta[3:], 2.0, rtol=1e-14)
    np.testing.assert_allclose(r.alpha[3:], 1.0, rtol=1e-14)
    np.testing.assert_allclose(r.r2[3:], 1.0, rtol=1e-14)
    np.testing.assert_allclose(r.resid[3:], 0.0, atol=1e-13)
