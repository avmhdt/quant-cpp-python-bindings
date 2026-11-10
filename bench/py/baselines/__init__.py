"""Comparator implementations for the micro suite (FR-24).

Each comparator takes ``(x, window)`` (rolling_ols: ``(y, x, window)``) for a NaN-free
float64 series with ``min_periods = window`` and returns the kernel's output as float64.
The runner sets OMP_NUM_THREADS, NUMBA_NUM_THREADS and POLARS_MAX_THREADS to 1, checks every
output against qkern before timing, and skips a comparator that lacks a kernel.
"""

from __future__ import annotations

from collections.abc import Callable

from bench.py.baselines._types import F64, Impl
from bench.py.baselines.bottleneck_impl import bottleneck_impls
from bench.py.baselines.numba_impl import numba_impls
from bench.py.baselines.numpy_impl import numpy_impls
from bench.py.baselines.pandas_impl import pandas_impls
from bench.py.baselines.polars_impl import polars_impls
from bench.py.baselines.pure_python import pure_python_impls

LOADERS: dict[str, Callable[[], dict[str, Impl]]] = {
    "pure_python": pure_python_impls,
    "numpy": numpy_impls,
    "pandas": pandas_impls,
    "numba": numba_impls,
    "polars": polars_impls,
    "bottleneck": bottleneck_impls,
}

__all__ = ["F64", "LOADERS", "Impl"]
