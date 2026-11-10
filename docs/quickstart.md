# Quickstart

Batch functions take a NumPy array (1-D, or 2-D processed column by column) and return a
float64 array of the same shape.

```python
>>> import numpy as np, qkern
>>> x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
>>> qkern.rolling_mean(x, 3)
array([       nan,        nan, 2.33333333, 4.66666667, 9.33333333])
>>> qkern.rolling_std(x, 3, min_periods=2)
array([       nan, 0.70710678, 1.52752523, 3.05505046, 6.11010093])
>>> qkern.ewm_mean(x, alpha=0.5)
array([ 1.    ,  1.5   ,  2.75  ,  5.375 , 10.6875])

```

NaN values are skipped by default; `nan_policy="propagate"` and `"raise"` are the
alternatives. `min_periods` defaults to the window.

```python
>>> y = np.array([1.0, np.nan, 3.0, 5.0])
>>> qkern.rolling_mean(y, 2, min_periods=1)
array([1., 1., 3., 4.])
>>> qkern.rolling_mean(y, 2, min_periods=1, nan_policy="propagate")
array([ 1., nan, nan,  4.])

```

A z-score that never uses the current value in its own baseline:

```python
>>> qkern.rolling_zscore(np.array([1.0, 2.0, 3.0, 6.0]), 3, include_current=False)
array([nan, nan, nan,  4.])

```

Rolling regression, for example a market beta:

```python
>>> m = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
>>> r = qkern.rolling_ols(1.0 + 2.0 * m, m, 3)
>>> r.beta
array([nan, nan,  2.,  2.,  2.])

```

Write into an existing array with `out=` (it is returned):

```python
>>> out = np.empty(5)
>>> qkern.rolling_max(x, 2, out=out) is out
True

```
