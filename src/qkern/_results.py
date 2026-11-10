"""Named result tuples returned by multi-output kernels and streaming ``update_many``."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt


class Moments(NamedTuple):
    """Rolling mean, variance and number of valid observations, one value per input row."""

    mean: npt.NDArray[np.float64]
    var: npt.NDArray[np.float64]
    # The spec names this field count; it shadows tuple.count on Moments instances.
    count: npt.NDArray[np.int64]  # type: ignore[assignment]


class OLSResult(NamedTuple):
    """Rolling regression of y on x with intercept: ``y = alpha + beta x + resid``."""

    alpha: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]
    resid: npt.NDArray[np.float64]
    r2: npt.NDArray[np.float64]


class BBO(NamedTuple):
    """Top of book after each order-book update (order-flow helpers, FR-07)."""

    ts: npt.NDArray[np.int64]
    bid_px: npt.NDArray[np.float64]
    bid_qty: npt.NDArray[np.float64]
    ask_px: npt.NDArray[np.float64]
    ask_qty: npt.NDArray[np.float64]
    gap: npt.NDArray[np.bool_]
    crossed: npt.NDArray[np.bool_]
