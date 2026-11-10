"""Polars Series rolling expressions (POLARS_MAX_THREADS=1)."""

from __future__ import annotations

from typing import Any

import numpy as np

from bench.py.baselines._types import F64, Impl


def polars_impls() -> dict[str, Impl]:
    import polars as pl

    def s(x: F64) -> Any:
        return pl.Series(x)

    def zscore(x: F64, w: int) -> F64:
        ser = s(x)
        return np.asarray((ser - ser.rolling_mean(w)) / ser.rolling_std(w), dtype=np.float64)

    return {
        "rolling_mean": lambda x, w: s(x).rolling_mean(w).to_numpy(),
        "rolling_std": lambda x, w: s(x).rolling_std(w).to_numpy(),
        "rolling_zscore": zscore,
        "ewm_mean": lambda x, w: s(x).ewm_mean(span=w, adjust=False).to_numpy(),
        "ewm_var": lambda x, w: s(x).ewm_var(span=w, adjust=False).to_numpy(),
        "rolling_max": lambda x, w: s(x).rolling_max(w).to_numpy(),
        "rolling_quantile": lambda x, w: s(x).rolling_median(w).to_numpy(),
    }
