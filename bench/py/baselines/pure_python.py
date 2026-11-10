"""Pure-Python loops (the README's baseline): O(n) streaming algorithms."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from bench.py.baselines._types import F64, Impl


def _py_mean(x: F64, w: int) -> F64:
    out = np.full(len(x), np.nan)
    s = 0.0
    xs = x.tolist()
    for i, v in enumerate(xs):
        s += v
        if i >= w:
            s -= xs[i - w]
        if i >= w - 1:
            out[i] = s / w
    return out


def _py_std(x: F64, w: int) -> F64:
    out = np.full(len(x), np.nan)
    xs = x.tolist()
    n = 0
    mean = m2 = 0.0
    for i, v in enumerate(xs):
        n += 1
        d = v - mean
        mean += d / n
        m2 += d * (v - mean)
        if i >= w:
            old = xs[i - w]
            n -= 1
            mb = mean
            mean -= (old - mb) / n
            m2 -= (old - mean) * (old - mb)
        if i >= w - 1:
            out[i] = math.sqrt(max(m2, 0.0) / (n - 1))
    return out


def _py_max(x: F64, w: int) -> F64:
    out = np.full(len(x), np.nan)
    q: deque[int] = deque()
    xs = x.tolist()
    for i, v in enumerate(xs):
        while q and xs[q[-1]] <= v:
            q.pop()
        q.append(i)
        if q[0] <= i - w:
            q.popleft()
        if i >= w - 1:
            out[i] = xs[q[0]]
    return out


def _py_ewm_mean(x: F64, alpha: float) -> F64:
    out = np.empty(len(x))
    m = 0.0
    for i, v in enumerate(x.tolist()):
        m = v if i == 0 else (1 - alpha) * m + alpha * v
        out[i] = m
    return out


def pure_python_impls() -> dict[str, Impl]:
    return {
        "rolling_mean": _py_mean,
        "rolling_std": _py_std,
        "rolling_max": _py_max,
        "ewm_mean": lambda x, w: _py_ewm_mean(x, 2.0 / (w + 1.0)),
    }
