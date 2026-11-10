# Batch vs streaming

Each kernel is a C++ state machine. A batch function creates one, pushes every element and
records the output; a streaming object keeps it between calls. Feeding a series in any
chunking gives results identical to the batch function, bit for bit (tested with hypothesis
for random chunkings in `tests/python/test_properties_streaming.py`).

```python
>>> import numpy as np, qkern
>>> x = np.arange(10.0)
>>> r = qkern.RollingMoments(4)
>>> first = r.update_many(x[:6])
>>> second = r.update_many(x[6:])
>>> np.array_equal(np.concatenate([first.var, second.var]), qkern.rolling_var(x, 4), equal_nan=True)
True
>>> r.update(10.0)
>>> float(r.mean)
8.5

```

| Batch | Streaming class | `update_many` returns |
|---|---|---|
| `rolling_mean`, `rolling_var`, `rolling_std`, `rolling_moments`, `rolling_zscore` | `RollingMoments` (`zscore(x)` scores against the current window) | `Moments(mean, var, count)` |
| `ewm_mean`, `ewm_var`, `ewm_std` | `EWM` | `(mean, var)` |
| `rolling_min`, `rolling_max` | `RollingMinMax` | `(min, max)` |
| `rolling_ols` (`include_current=True`) | `RollingOLS` (`resid` is the last pair's residual) | `OLSResult` |
| `rolling_quantile`, `rolling_median` | `RollingQuantile` | quantiles |

Streaming objects pickle and copy; a restored object continues identically, so a live
process can checkpoint its state:

```python
>>> import pickle
>>> e = qkern.EWM(span=20)
>>> _ = e.update_many(x)
>>> restored = pickle.loads(pickle.dumps(e))
>>> restored.update(11.0); e.update(11.0)
>>> restored.mean == e.mean
True

```

`update(x)` keeps the GIL (it is a single, very short call); `update_many` releases it like a
batch call. A streaming object is not thread-safe: give each thread its own.
