"""bottleneck move_* functions."""

from __future__ import annotations

from bench.py.baselines._types import Impl


def bottleneck_impls() -> dict[str, Impl]:
    import bottleneck as bn

    return {
        "rolling_mean": lambda x, w: bn.move_mean(x, w),
        "rolling_std": lambda x, w: bn.move_std(x, w, ddof=1),
        "rolling_max": lambda x, w: bn.move_max(x, w),
        "rolling_quantile": lambda x, w: bn.move_median(x, w),
    }
