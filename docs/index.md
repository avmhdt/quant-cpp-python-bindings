# qkern

Fast rolling and streaming time-series kernels for Python: a header-only C++20 core bound
with nanobind, shipped as typed wheels for Linux, macOS and Windows.

- **Kernels**: rolling mean, variance, standard deviation and z-score; exponentially weighted
  mean, variance and standard deviation; rolling minimum and maximum; rolling OLS (alpha,
  beta, residual, r²); rolling quantile and median.
- **Batch and streaming**: every kernel is a C++ state machine. The batch functions and the
  streaming classes drive the same code, so a live system and its backtest agree bit for bit.
- **No copies**: float64 and float32 NumPy arrays of any stride and Arrow arrays (pyarrow,
  Polars) are read in place; results are returned without a copy.
- **No GIL contention**: calls on 1,024 or more elements release the GIL, so threads run
  kernels in parallel.
- **Explicit semantics**: NaN policy, `min_periods`, infinities and every difference from
  pandas are documented and tested ([Semantics](semantics.md)).

```python
import numpy as np
import qkern

x = np.random.default_rng(0).standard_normal(10_000_000)
vol = qkern.rolling_std(x, 252)
z = qkern.rolling_zscore(x, 252, include_current=False)
```

Licence: MIT. Source: [github.com/avmhdt/quant-cpp-python-bindings](https://github.com/avmhdt/quant-cpp-python-bindings).
