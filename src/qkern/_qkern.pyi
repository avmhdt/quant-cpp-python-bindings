"""qkern C++ extension: streaming and rolling time-series kernels."""

from typing import Annotated, Any

import numpy
import numpy.typing
from numpy.typing import NDArray

import qkern._results


def _build_info() -> dict[str, object]:
    """Build metadata used by qkern.build_info()."""

def _set_gil_release_min_n(n: int) -> None:
    """Private test/benchmark hook: GIL release threshold in elements."""

def _get_gil_release_min_n() -> int:
    """Private test/benchmark hook: current GIL release threshold."""

def _debug_counters() -> dict[str, int]:
    """
    Private test hook: input conversions and output copies made by the binding layer.
    """

def _reset_debug_counters() -> None:
    """Private test hook: zero the debug counters."""

def rolling_mean(x: object, window: int, min_periods: int | None, nan_policy: str, out: Annotated[NDArray[numpy.float64], dict(device='cpu')] | None) -> NDArray[numpy.float64] | None:
    """Rolling mean (see qkern.rolling_mean). Returns None when out is given."""

def rolling_var(x: object, window: int, min_periods: int | None, ddof: int, nan_policy: str, out: Annotated[NDArray[numpy.float64], dict(device='cpu')] | None) -> NDArray[numpy.float64] | None:
    """
    Rolling variance (see qkern.rolling_var). Returns None when out is given.
    """

def rolling_std(x: object, window: int, min_periods: int | None, ddof: int, nan_policy: str, out: Annotated[NDArray[numpy.float64], dict(device='cpu')] | None) -> NDArray[numpy.float64] | None:
    """
    Rolling standard deviation (see qkern.rolling_std). Returns None when out is given.
    """

def rolling_zscore(x: object, window: int, min_periods: int | None, ddof: int, include_current: bool, nan_policy: str, out: Annotated[NDArray[numpy.float64], dict(device='cpu')] | None) -> NDArray[numpy.float64] | None:
    """
    Rolling z-score (see qkern.rolling_zscore). Returns None when out is given.
    """

def rolling_moments(x: object, window: int, min_periods: int | None, ddof: int, nan_policy: str) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64], NDArray[numpy.int64]]:
    """
    Rolling mean, variance and valid count in one pass (see qkern.rolling_moments).
    """

def _resolve_alpha(alpha: float | None, span: float | None, halflife: float | None) -> float:
    """
    Smoothing factor from exactly one of alpha, span (2 / (span + 1)) or halflife (1 - exp(-ln 2 / halflife)).
    """

def ewm_mean(x: object, alpha: float, min_periods: int, nan_policy: str, out: Annotated[NDArray[numpy.float64], dict(device='cpu')] | None) -> NDArray[numpy.float64] | None:
    """EWM mean (see qkern.ewm_mean). Returns None when out is given."""

def ewm_var(x: object, alpha: float, bias: bool, min_periods: int, nan_policy: str, out: Annotated[NDArray[numpy.float64], dict(device='cpu')] | None) -> NDArray[numpy.float64] | None:
    """EWM variance (see qkern.ewm_var). Returns None when out is given."""

def ewm_std(x: object, alpha: float, bias: bool, min_periods: int, nan_policy: str, out: Annotated[NDArray[numpy.float64], dict(device='cpu')] | None) -> NDArray[numpy.float64] | None:
    """
    EWM standard deviation (see qkern.ewm_std). Returns None when out is given.
    """

def rolling_min(x: object, window: int, min_periods: int | None, nan_policy: str, out: Annotated[NDArray[numpy.float64], dict(device='cpu')] | None) -> NDArray[numpy.float64] | None:
    """
    Rolling minimum (see qkern.rolling_min). Returns None when out is given.
    """

def rolling_max(x: object, window: int, min_periods: int | None, nan_policy: str, out: Annotated[NDArray[numpy.float64], dict(device='cpu')] | None) -> NDArray[numpy.float64] | None:
    """
    Rolling maximum (see qkern.rolling_max). Returns None when out is given.
    """

def rolling_ols(y: object, x: object, window: int, min_periods: int | None, include_current: bool, nan_policy: str) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64], NDArray[numpy.float64], NDArray[numpy.float64]]:
    """
    Rolling OLS of y on x with intercept: (alpha, beta, resid, r2) (see qkern.rolling_ols).
    """

class RollingMoments:
    """
    Streaming rolling mean, variance, std and z-score over the last `window` observations (same semantics as qkern.rolling_moments).
    """

    def __init__(self, window: int, *, min_periods: int | None = None, ddof: int = 1, nan_policy: str = 'skip') -> None: ...

    def update(self, x: float) -> None:
        """Push one observation."""

    def update_many(self, x: numpy.typing.NDArray[Any]) -> qkern._results.Moments:
        """Push every element of a 1-D array; returns the per-element Moments."""

    def zscore(self, x: float) -> float:
        """z-score of x against the current window (x is not pushed)."""

    def reset(self) -> None:
        """Forget every observation."""

    @property
    def mean(self) -> float: ...

    @property
    def var(self) -> float: ...

    @property
    def std(self) -> float: ...

    @property
    def count(self) -> int: ...

    @property
    def ready(self) -> bool: ...

    @property
    def window(self) -> int: ...

    @property
    def min_periods(self) -> int: ...

    @property
    def ddof(self) -> int: ...

    @property
    def nan_policy(self) -> str: ...

    def __getstate__(self) -> dict[str, object]: ...

    def __setstate__(self, arg: dict[str, object], /) -> None: ...

    def __repr__(self) -> str: ...

class EWM:
    """
    Streaming exponentially weighted mean, variance and std (same semantics as qkern.ewm_mean / ewm_var).
    """

    def __init__(self, *, alpha: float | None = None, span: float | None = None, halflife: float | None = None, bias: bool = False, min_periods: int = 0, nan_policy: str = 'skip') -> None: ...

    def update(self, x: float) -> None:
        """Push one observation."""

    def update_many(self, x: numpy.typing.NDArray[Any]) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]:
        """Push every element of a 1-D array; returns per-element (mean, var)."""

    def reset(self) -> None:
        """Forget every observation."""

    @property
    def mean(self) -> float: ...

    @property
    def var(self) -> float: ...

    @property
    def std(self) -> float: ...

    @property
    def count(self) -> int: ...

    @property
    def ready(self) -> bool: ...

    @property
    def alpha(self) -> float: ...

    @property
    def bias(self) -> bool: ...

    @property
    def min_periods(self) -> int: ...

    @property
    def nan_policy(self) -> str: ...

    def __getstate__(self) -> dict[str, object]: ...

    def __setstate__(self, arg: dict[str, object], /) -> None: ...

    def __repr__(self) -> str: ...

class RollingMinMax:
    """
    Streaming rolling minimum and maximum (monotonic deques; same semantics as qkern.rolling_min / rolling_max).
    """

    def __init__(self, window: int, *, min_periods: int | None = None, nan_policy: str = 'skip') -> None: ...

    def update(self, x: float) -> None:
        """Push one observation."""

    def update_many(self, x: numpy.typing.NDArray[Any]) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]:
        """Push every element of a 1-D array; returns per-element (min, max)."""

    def reset(self) -> None:
        """Forget every observation."""

    @property
    def min(self) -> float: ...

    @property
    def max(self) -> float: ...

    @property
    def count(self) -> int: ...

    @property
    def ready(self) -> bool: ...

    @property
    def window(self) -> int: ...

    @property
    def min_periods(self) -> int: ...

    @property
    def nan_policy(self) -> str: ...

    def __getstate__(self) -> dict[str, object]: ...

    def __setstate__(self, arg: dict[str, object], /) -> None: ...

    def __repr__(self) -> str: ...

class RollingOLS:
    """
    Streaming rolling OLS of y on x with intercept (same semantics as qkern.rolling_ols with include_current=True).
    """

    def __init__(self, window: int, *, min_periods: int | None = None, nan_policy: str = 'skip') -> None: ...

    def update(self, y: float, x: float) -> None:
        """Push one (y, x) pair."""

    def update_many(self, y: numpy.typing.NDArray[Any], x: numpy.typing.NDArray[Any]) -> qkern._results.OLSResult:
        """Push every pair of two 1-D arrays; returns the per-element OLSResult."""

    def reset(self) -> None:
        """Forget every observation."""

    @property
    def alpha(self) -> float: ...

    @property
    def beta(self) -> float: ...

    @property
    def r2(self) -> float: ...

    @property
    def resid(self) -> float: ...

    @property
    def count(self) -> int: ...

    @property
    def ready(self) -> bool: ...

    @property
    def window(self) -> int: ...

    @property
    def min_periods(self) -> int: ...

    @property
    def nan_policy(self) -> str: ...

    def __getstate__(self) -> dict[str, object]: ...

    def __setstate__(self, arg: dict[str, object], /) -> None: ...

    def __repr__(self) -> str: ...

class _ArrowInput:
    """Zero-copy view of a float64/float32 Arrow array (PyCapsule interface)."""

    @staticmethod
    def from_array(schema: object, array: object) -> _ArrowInput:
        """Wrap the capsules returned by __arrow_c_array__()."""

    @staticmethod
    def from_stream(stream: object) -> _ArrowInput:
        """Wrap the single chunk of the stream returned by __arrow_c_stream__()."""

    def __len__(self) -> int: ...

def rolling_quantile(x: object, window: int, min_periods: int | None, q: float, interpolation: str, nan_policy: str, out: Annotated[NDArray[numpy.float64], dict(device='cpu')] | None) -> NDArray[numpy.float64] | None:
    """
    Rolling quantile (see qkern.rolling_quantile). Returns None when out is given.
    """

class RollingQuantile:
    """
    Streaming rolling quantile (two heaps with lazy deletion; same semantics as qkern.rolling_quantile).
    """

    def __init__(self, window: int, q: float, *, interpolation: str = 'linear', min_periods: int | None = None, nan_policy: str = 'skip') -> None: ...

    def update(self, x: float) -> None:
        """Push one observation."""

    def update_many(self, x: numpy.typing.NDArray[Any]) -> numpy.typing.NDArray[numpy.float64]:
        """Push every element of a 1-D array; returns the per-element quantile."""

    def reset(self) -> None:
        """Forget every observation."""

    @property
    def value(self) -> float: ...

    @property
    def count(self) -> int: ...

    @property
    def ready(self) -> bool: ...

    @property
    def q(self) -> float: ...

    @property
    def interpolation(self) -> str: ...

    @property
    def window(self) -> int: ...

    @property
    def min_periods(self) -> int: ...

    @property
    def nan_policy(self) -> str: ...

    def __getstate__(self) -> dict[str, object]: ...

    def __setstate__(self, arg: dict[str, object], /) -> None: ...

    def __repr__(self) -> str: ...
