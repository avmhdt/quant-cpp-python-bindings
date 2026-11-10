"""qkern: fast streaming and rolling time-series kernels (C++20 core, nanobind bindings).

Batch functions take NumPy arrays (1-D, or 2-D along axis 0) and return float64 arrays;
streaming classes update one observation or one chunk at a time and give bit-identical
results to the batch functions. Streaming objects are not thread-safe.
"""

from qkern._api import (
    ArrayIn,
    ArrowArrayExportable,
    ArrowStreamExportable,
    build_info,
    ewm_mean,
    ewm_std,
    ewm_var,
    rolling_max,
    rolling_mean,
    rolling_median,
    rolling_min,
    rolling_moments,
    rolling_ols,
    rolling_quantile,
    rolling_std,
    rolling_var,
    rolling_zscore,
    show_config,
)
from qkern._qkern import EWM, RollingMinMax, RollingMoments, RollingOLS, RollingQuantile
from qkern._results import BBO, Moments, OLSResult
from qkern._validate import Interp, NanPolicy

#: Package version, compiled into the extension from pyproject.toml (FR-22).
__version__: str = str(build_info()["version"])

__all__ = [
    "BBO",
    "EWM",
    "ArrayIn",
    "ArrowArrayExportable",
    "ArrowStreamExportable",
    "Interp",
    "Moments",
    "NanPolicy",
    "OLSResult",
    "RollingMinMax",
    "RollingMoments",
    "RollingOLS",
    "RollingQuantile",
    "__version__",
    "build_info",
    "ewm_mean",
    "ewm_std",
    "ewm_var",
    "rolling_max",
    "rolling_mean",
    "rolling_median",
    "rolling_min",
    "rolling_moments",
    "rolling_ols",
    "rolling_quantile",
    "rolling_std",
    "rolling_var",
    "rolling_zscore",
    "show_config",
]
