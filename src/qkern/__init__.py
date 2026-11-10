"""qkern: fast streaming and rolling time-series kernels (C++20 core, nanobind bindings)."""

from qkern._api import ArrayIn, ArrowArrayExportable, build_info, rolling_mean, show_config
from qkern._validate import NanPolicy

#: Package version, compiled into the extension from pyproject.toml (FR-22).
__version__: str = str(build_info()["version"])

__all__ = [
    "ArrayIn",
    "ArrowArrayExportable",
    "NanPolicy",
    "__version__",
    "build_info",
    "rolling_mean",
    "show_config",
]
