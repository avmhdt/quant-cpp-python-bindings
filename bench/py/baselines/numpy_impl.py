"""NumPy: cumulative sums and sliding_window_view."""

from __future__ import annotations

import numpy as np

from bench.py.baselines._types import F64, Impl


def _np_mean(x: F64, w: int) -> F64:
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(len(x), np.nan)
    out[w - 1 :] = (c[w:] - c[:-w]) / w
    return out


def _np_std(x: F64, w: int) -> F64:
    out = np.full(len(x), np.nan)
    out[w - 1 :] = np.lib.stride_tricks.sliding_window_view(x, w).std(axis=1, ddof=1)
    return out


def _np_max(x: F64, w: int) -> F64:
    out = np.full(len(x), np.nan)
    out[w - 1 :] = np.lib.stride_tricks.sliding_window_view(x, w).max(axis=1)
    return out


def _np_quantile(x: F64, w: int) -> F64:
    out = np.full(len(x), np.nan)
    out[w - 1 :] = np.quantile(np.lib.stride_tricks.sliding_window_view(x, w), 0.5, axis=1)
    return out


def numpy_impls() -> dict[str, Impl]:
    return {
        "rolling_mean": _np_mean,
        "rolling_std": _np_std,
        "rolling_max": _np_max,
        "rolling_quantile": _np_quantile,
    }
