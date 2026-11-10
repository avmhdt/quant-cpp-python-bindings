"""pandas rolling / ewm (adjust=False)."""

from __future__ import annotations

from typing import Any

import numpy as np

from bench.py.baselines._types import F64, Impl


def pandas_impls() -> dict[str, Impl]:
    import pandas as pd

    def roll(x: F64, w: int) -> Any:
        return pd.Series(x).rolling(w)

    def zscore(x: F64, w: int) -> F64:
        r = roll(x, w)
        return np.asarray((pd.Series(x) - r.mean()) / r.std(), dtype=np.float64)

    def ols(y: F64, x: F64, w: int) -> F64:
        ys, xs = pd.Series(y), pd.Series(x)
        return np.asarray(ys.rolling(w).cov(xs) / xs.rolling(w).var(), dtype=np.float64)

    return {
        "rolling_mean": lambda x, w: roll(x, w).mean().to_numpy(),
        "rolling_std": lambda x, w: roll(x, w).std().to_numpy(),
        "rolling_zscore": zscore,
        "ewm_mean": lambda x, w: pd.Series(x).ewm(span=w, adjust=False).mean().to_numpy(),
        "ewm_var": lambda x, w: pd.Series(x).ewm(span=w, adjust=False).var().to_numpy(),
        "rolling_max": lambda x, w: roll(x, w).max().to_numpy(),
        "rolling_ols": ols,
        "rolling_quantile": lambda x, w: roll(x, w).median().to_numpy(),
    }
