"""Shared type aliases for the baselines."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

F64: TypeAlias = npt.NDArray[np.float64]
Impl: TypeAlias = Callable[..., F64]
