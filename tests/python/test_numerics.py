"""Numerical accuracy, NFR-06 (a)-(c) (AC-W-15), and the naive-cumsum failure it avoids."""

from __future__ import annotations

import math

import numpy as np
import pytest
from strategies import F64

import qkern


def _exact(window: F64) -> tuple[float, float]:
    n = len(window)
    mean = math.fsum(window) / n
    return mean, math.fsum((v - mean) ** 2 for v in window) / (n - 1)


def _sample_points(n: int, w: int, count: int = 1000) -> F64:
    return np.linspace(w - 1, n - 1, count).astype(np.int64)


@pytest.mark.parametrize("w", [20, 252])
def test_gaussian_mean_and_variance(w: int) -> None:
    """(a) n = 1e6: mean error <= 1e-12 max(1, |exact|), variance relative error <= 1e-10."""
    x = np.random.default_rng(1).standard_normal(1_000_000)
    mean = qkern.rolling_mean(x, w)
    var = qkern.rolling_var(x, w)
    std = qkern.rolling_std(x, w)
    for t in _sample_points(len(x), w):
        m, v = _exact(x[t - w + 1 : t + 1])
        assert abs(mean[t] - m) <= 1e-12 * max(1.0, abs(m))
        assert abs(var[t] - v) <= 1e-10 * v
        assert abs(std[t] - math.sqrt(v)) <= 1e-10 * math.sqrt(v)


@pytest.mark.parametrize("w", [20, 252])
def test_offset_data_variance(w: int) -> None:
    """(b) 1e9 + N(0, 1): relative variance error <= 1e-6; the naive cumsum method fails."""
    x = 1e9 + np.random.default_rng(2).standard_normal(200_000)
    var = qkern.rolling_var(x, w)
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])
    worst = worst_naive = 0.0
    for t in _sample_points(len(x), w, 300):
        _, v = _exact(x[t - w + 1 : t + 1])
        s1 = c1[t + 1] - c1[t + 1 - w]
        s2 = c2[t + 1] - c2[t + 1 - w]
        naive = (s2 - s1 * s1 / w) / (w - 1)
        worst = max(worst, abs(var[t] - v) / v)
        worst_naive = max(worst_naive, abs(naive - v) / v)
    assert worst <= 1e-6
    assert worst_naive > 1e-6, "the naive cumulative-sum variance should fail on this data"


@pytest.mark.slow
def test_streaming_drift_over_1e8_updates() -> None:
    """(c) After 1e8 streaming updates with w = 100, relative variance error <= 1e-9."""
    rng = np.random.default_rng(3)
    obj = qkern.RollingMoments(100)
    last = np.empty(0)
    for _ in range(100):
        chunk = 3.0 + rng.standard_normal(1_000_000)
        obj.update_many(chunk)
        last = chunk[-100:]
    _, v = _exact(last)
    assert abs(obj.var - v) / v <= 1e-9


def test_large_value_leaving_window_keeps_small_variance_exact() -> None:
    x = np.array([1e8, 1.0, 2.0, 3.0, 0.5, 0.25])
    var = qkern.rolling_var(x, 3)
    assert var[3] == 1.0
    _, v = _exact(np.array([2.0, 3.0, 0.5]))
    assert abs(var[4] - v) <= 1e-15 * v
