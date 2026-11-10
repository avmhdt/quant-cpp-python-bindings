"""Hypothesis strategies and the kernel table shared by the property tests (spec section 11).

Series: lengths 0-2,000 (shorter by default for speed), finite floats with |x| <= 1e6, NaN
with probability 0-30%, optional infinities; windows 1 .. n + 5; random chunk boundaries.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from hypothesis import strategies as st

import qkern
from qkern.testing import reference

F64 = npt.NDArray[np.float64]

MAX_ABS = 1e6


@st.composite
def series(
    draw: st.DrawFn,
    *,
    min_size: int = 0,
    max_size: int = 300,
    allow_nan: bool = True,
    allow_inf: bool = False,
    max_abs: float = MAX_ABS,
) -> F64:
    """A float64 series with an optional share of NaN (0-30%) and infinities."""
    n = draw(st.integers(min_size, max_size))
    finite = st.floats(-max_abs, max_abs, allow_nan=False, allow_infinity=False)
    values = draw(st.lists(finite, min_size=n, max_size=n))
    x = np.array(values, dtype=np.float64)
    if n and allow_nan:
        p_nan = draw(st.floats(0.0, 0.3))
        mask = np.array(draw(st.lists(st.floats(0.0, 1.0), min_size=n, max_size=n))) < p_nan
        x[mask] = np.nan
    if n and allow_inf and draw(st.booleans()):
        idx = draw(st.lists(st.integers(0, n - 1), min_size=1, max_size=3))
        signs = draw(
            st.lists(st.sampled_from([np.inf, -np.inf]), min_size=len(idx), max_size=len(idx))
        )
        x[idx] = signs
    return x


@st.composite
def window_for(draw: st.DrawFn, n: int) -> tuple[int, int]:
    """(window, min_periods) with window in 1 .. n + 5."""
    window = draw(st.integers(1, max(1, n + 5)))
    return window, draw(st.integers(1, window))


@st.composite
def chunk_bounds(draw: st.DrawFn, n: int) -> list[int]:
    """Sorted cut points splitting range(n) into random chunks (possibly empty)."""
    cuts = draw(st.lists(st.integers(0, n), max_size=8))
    return [0, *sorted(cuts), n]


def assert_close(got: F64, want: F64 | list[float], rtol: float = 1e-9) -> None:
    """|got - want| <= rtol * max(1, |want|) elementwise, NaN where want is NaN."""
    w = np.asarray(want, dtype=np.float64)
    assert got.shape == w.shape, (got.shape, w.shape)
    nan_w = np.isnan(w)
    nan_g = np.isnan(got)
    bad = np.flatnonzero(nan_w != nan_g)
    assert bad.size == 0, f"NaN mismatch at {bad[:5]}: got {got[bad[:5]]}, want {w[bad[:5]]}"
    inf = np.isinf(w)
    np.testing.assert_array_equal(got[inf], w[inf])
    ok = ~nan_w & ~inf
    err = np.abs(got[ok] - w[ok])
    tol = rtol * np.maximum(1.0, np.abs(w[ok]))
    worst = int(np.argmax(err - tol)) if err.size else 0
    assert np.all(err <= tol), (
        f"max excess at {worst}: got {got[ok][worst]!r}, want {w[ok][worst]!r}"
    )


def assert_identical(a: F64, b: F64) -> None:
    """Bit-for-bit equality, NaN matching NaN."""
    np.testing.assert_array_equal(a, b)
    assert a.dtype == b.dtype


# ---------------------------------------------------------------------------------------
# Univariate kernels: qkern batch call, pure-Python reference, streaming equivalent
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Unary:
    name: str
    kind: str  # "rolling" or "ewm"
    batch: Callable[..., F64]
    ref: Callable[..., list[float]]
    extra: dict[str, Any] = field(default_factory=dict)


ROLLING: tuple[Unary, ...] = (
    Unary("rolling_mean", "rolling", qkern.rolling_mean, reference.rolling_mean),
    Unary("rolling_var", "rolling", qkern.rolling_var, reference.rolling_var),
    Unary("rolling_var_ddof0", "rolling", qkern.rolling_var, reference.rolling_var, {"ddof": 0}),
    Unary("rolling_std", "rolling", qkern.rolling_std, reference.rolling_std),
    Unary("rolling_zscore", "rolling", qkern.rolling_zscore, reference.rolling_zscore),
    Unary(
        "rolling_zscore_oos",
        "rolling",
        qkern.rolling_zscore,
        reference.rolling_zscore,
        {"include_current": False},
    ),
    Unary("rolling_min", "rolling", qkern.rolling_min, reference.rolling_min),
    Unary("rolling_max", "rolling", qkern.rolling_max, reference.rolling_max),
)


def _ref_ewm(fn: Callable[..., list[float]]) -> Callable[..., list[float]]:
    def call(x: F64, **kw: Any) -> list[float]:
        alpha = kw.pop("alpha")
        return fn(x, alpha, **kw)

    return call


def _ref_ewm_std(x: F64, **kw: Any) -> list[float]:
    alpha = kw.pop("alpha")
    return [math.sqrt(v) for v in reference.ewm_var(x, alpha, **kw)]


EWM: tuple[Unary, ...] = (
    Unary("ewm_mean", "ewm", qkern.ewm_mean, _ref_ewm(reference.ewm_mean)),
    Unary("ewm_var", "ewm", qkern.ewm_var, _ref_ewm(reference.ewm_var)),
    Unary("ewm_var_bias", "ewm", qkern.ewm_var, _ref_ewm(reference.ewm_var), {"bias": True}),
    Unary("ewm_std", "ewm", qkern.ewm_std, _ref_ewm_std),
)

UNARY = ROLLING + EWM
