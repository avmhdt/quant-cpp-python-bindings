"""Numba @njit loops; the runner times JIT compilation separately."""

from __future__ import annotations

import math

import numpy as np

from bench.py.baselines._types import F64, Impl


def numba_impls() -> dict[str, Impl]:
    import numba

    @numba.njit(cache=False)
    def mean(x: F64, w: int) -> F64:
        out = np.full(len(x), np.nan)
        s = 0.0
        for i in range(len(x)):
            s += x[i]
            if i >= w:
                s -= x[i - w]
            if i >= w - 1:
                out[i] = s / w
        return out

    @numba.njit(cache=False)
    def std(x: F64, w: int) -> F64:
        out = np.full(len(x), np.nan)
        n = 0
        m = 0.0
        m2 = 0.0
        for i in range(len(x)):
            v = x[i]
            n += 1
            d = v - m
            m += d / n
            m2 += d * (v - m)
            if i >= w:
                old = x[i - w]
                n -= 1
                mb = m
                m -= (old - mb) / n
                m2 -= (old - m) * (old - mb)
            if i >= w - 1:
                out[i] = math.sqrt(max(m2, 0.0) / (n - 1))
        return out

    @numba.njit(cache=False)
    def ewm_mean(x: F64, w: int) -> F64:
        a = 2.0 / (w + 1.0)
        out = np.empty(len(x))
        m = x[0]
        for i in range(len(x)):
            m = x[i] if i == 0 else (1.0 - a) * m + a * x[i]
            out[i] = m
        return out

    @numba.njit(cache=False)
    def rmax(x: F64, w: int) -> F64:
        n = len(x)
        out = np.full(n, np.nan)
        q = np.empty(n, dtype=np.int64)
        head = 0
        tail = 0
        for i in range(n):
            while tail > head and x[q[tail - 1]] <= x[i]:
                tail -= 1
            q[tail] = i
            tail += 1
            if q[head] <= i - w:
                head += 1
            if i >= w - 1:
                out[i] = x[q[head]]
        return out

    return {"rolling_mean": mean, "rolling_std": std, "ewm_mean": ewm_mean, "rolling_max": rmax}
