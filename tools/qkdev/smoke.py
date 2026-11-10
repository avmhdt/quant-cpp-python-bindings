"""Installed-package smoke test (FR-23, AC-W-01): run from outside the repository against an
installed qkern. Uses only qkern and NumPy, so it works in a bare environment.

    python tools/qkdev/smoke.py
"""

from __future__ import annotations

import pickle
import sys

import numpy as np


def main() -> int:
    import qkern

    qkern.show_config()
    info = qkern.build_info()
    x = np.random.default_rng(0).standard_normal(10_000)
    x[::97] = np.nan
    checks = {
        "rolling_mean": qkern.rolling_mean(x, 20, min_periods=5),
        "rolling_std": qkern.rolling_std(x, 20, min_periods=5),
        "rolling_zscore": qkern.rolling_zscore(x, 20, min_periods=5),
        "ewm_var": qkern.ewm_var(x, span=20),
        "rolling_max": qkern.rolling_max(x, 20, min_periods=5),
        "rolling_median": qkern.rolling_median(x, 20, min_periods=5),
        "rolling_ols": qkern.rolling_ols(x, x[::-1].copy(), 20, min_periods=5).beta,
    }
    for name, out in checks.items():
        assert out.shape == x.shape, name
        assert out.dtype == np.float64, name
        assert np.isfinite(out[200:]).any(), name
    stream = qkern.RollingMoments(20, min_periods=5)
    parts = [
        stream.update_many(x[:5000]).var,
        pickle.loads(pickle.dumps(stream)).update_many(x[5000:]).var,
    ]
    batch = qkern.rolling_var(x, 20, min_periods=5)
    np.testing.assert_array_equal(np.concatenate(parts), batch)
    want_abi3 = sys.version_info >= (3, 12) and not info["free_threaded"]
    assert info["abi3"] == want_abi3, f"abi3={info['abi3']} on Python {sys.version_info[:2]}"
    print(f"qkern {qkern.__version__} smoke test passed on Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
