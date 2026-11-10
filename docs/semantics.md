# Semantics

Every qkern kernel follows the rules on this page. Each rule names the test that checks it
(`tests/python/...` unless marked C++). The last section lists every place qkern
deliberately differs from pandas.

## Windows and look-ahead

Windows count observations, not time. At row `t` a window of size `w` holds rows
`t - w + 1 .. t` (clipped at 0). No output at row `t` depends on rows after `t` (DQ-01). With
`include_current=False` (`rolling_zscore`, `rolling_ols`) the statistics come from rows
`t - w .. t - 1`, so row `t` never takes part in its own baseline.

Tests: `test_properties_lookahead.py::test_univariate_kernels_have_no_lookahead`,
`::test_rolling_ols_has_no_lookahead`, `::test_excluding_current_ignores_row_t`.

## NaN policy (DQ-02)

| `nan_policy` | Rolling kernels | EWM kernels | Test |
|---|---|---|---|
| `"skip"` (default) | NaN is left out; statistics use the valid values among the window's rows | A NaN leaves the state unchanged; the output is the current value | `test_semantics.py::test_dq02_skip_uses_remaining_valid_values`, `::test_dq02_ewm_skip_keeps_state_and_propagate_poisons` |
| `"propagate"` | NaN while any NaN is in the window | The window is unbounded, so the first NaN makes every later output NaN (until `reset()`) | `::test_dq02_propagate_while_nan_in_window`, `::test_dq02_ewm_skip_keeps_state_and_propagate_poisons` |
| `"raise"` | ValueError before any output is written | same | `::test_dq02_raise_before_computing` |

A NaN `x_t` always gives a NaN z-score. In `rolling_ols` a pair with NaN on either side is
not an observation (pairwise complete).

## `min_periods` (DQ-03)

The output is NaN while the window holds fewer than `min_periods` valid values. Rolling
kernels default to `min_periods = window` and need `1 <= min_periods <= window`. EWM kernels
default to 0, which behaves as 1. The variance also needs `n_valid - ddof >= 1`, and OLS needs
at least two finite pairs with `Sxx > 0`.

Tests: `test_semantics.py::test_dq03_default_min_periods_is_window`,
`::test_dq03_ewm_default_min_periods_zero_behaves_as_one`,
`::test_dq03_variance_needs_n_minus_ddof_at_least_one`,
`::test_dq03_ols_needs_two_pairs_and_positive_sxx`.

## Infinities (DQ-04)

`+inf` and `-inf` are valid values, never missing.

| Kernel | While an infinity is in the window | Test |
|---|---|---|
| mean | `+inf` or `-inf` if one sign is present, NaN if both are | `test_semantics.py::test_dq04_mean_with_one_or_both_infinity_signs` |
| var, std, z-score, OLS | NaN; once the infinity leaves, results are clean again (it is counted apart from the running sums) | `::test_dq04_variance_nan_while_inf_in_window_then_clean` |
| min, max | ordered normally | `::test_dq04_min_max_order_infinities` |
| EWM | follows the recurrence: the mean becomes ±inf (NaN once both signs were seen), the variance NaN from then on | `test_properties_reference.py::test_ewm_kernels_match_reference` |

## dtypes (DQ-05)

float64 and float32 arrays of any stride (including negative strides and columns of 2-D
arrays) are read in place; all arithmetic is in double precision and results are float64.
Integer and bool arrays are converted once to float64 (counted by
`qkern._qkern._debug_counters()`); float16 and long double are converted by the Python layer.
Complex, object, string, datetime and non-native byte-order arrays raise TypeError.

Tests: `test_semantics.py::test_dq05_*`, `test_zero_copy.py`, and
`test_properties_layout.py` (float32 equals its float64 upcast exactly, strided equals
contiguous exactly, integers equal their float64 values exactly).

## 2-D inputs

A `(n, k)` array, C or Fortran order, is processed column by column along axis 0 and gives
`(n, k)` results, which qkern returns in column-major (Fortran) order because each column is
computed contiguously. The GIL is released once for the whole call. In `rolling_ols`, `y` may
be `(n, k)` with `x` either `(n, k)` or `(n,)`.

Tests: `test_properties_layout.py::test_two_d_equals_column_by_column`,
`::test_two_d_ols_with_shared_and_per_column_x`.

## Arrow nulls (DQ-06)

Arrow input arrives in M4; null slots will be treated exactly as NaN.

## `out=` and aliasing (DQ-11)

`out` must be a writeable native float64 array of the input's shape (any stride). It must not
overlap the input: qkern compares the two arrays' byte ranges and raises ValueError on
overlap. Do not mutate an input from another thread while a call that released the GIL is
running; this is documented, not detected.

Tests: `test_semantics.py::test_dq11_out_overlapping_input_raises`,
`::test_read_only_out_raises_type_error`, `test_zero_copy.py::test_out_is_the_same_object_and_written_in_place`.

## Errors

| Condition | Python exception | Test |
|---|---|---|
| `window < 1`, `min_periods` out of range, `ddof < 0`, bad `alpha`/`span`/`halflife`, not exactly one decay parameter | ValueError | `test_semantics.py::test_error_table` |
| Length or shape mismatch (OLS, `out`), overlapping `out`, NaN with `nan_policy="raise"` | ValueError | same |
| Unsupported dtype or ndim, read-only or non-float64 `out`, non-integer `window` | TypeError | same, `::test_read_only_out_raises_type_error` |

## Differences from pandas

Each difference is deliberate and tested.

| # | Topic | pandas | qkern | Test |
|---|---|---|---|---|
| 1 | Infinities | Treated as missing in rolling windows | Valid values (DQ-04) | `test_semantics.py::test_dq04_pandas_difference_is_documented` |
| 2 | Constant windows | Variance can carry rounding residue; z-scores become ±inf or noise | Variance exactly 0, z-score NaN | `test_properties_pandas.py::test_zscore_matches_pandas_expression`; C++ `RollingVar.ConstantWindowIsExactlyZero` |
| 3 | After a large value leaves the window | Sliding sums keep residue (for example std 2.8e-7 for the window `[0, 0, 0]` after 46 left it) | Exact recomputation when the sum of squares collapses | `test_properties_pandas.py::test_rolling_moments_match_pandas`, `test_numerics.py::test_large_value_leaving_window_keeps_small_variance_exact` |
| 4 | NaN handling options | `ignore_na`, `adjust` | EWM is pandas' `adjust=False, ignore_na=True`; `nan_policy` adds `"propagate"` and `"raise"` | `test_properties_pandas.py::test_ewm_matches_pandas`, `test_semantics.py::test_dq02_*` |
| 5 | Rolling covariance | `E[xy] - E[x]E[y]`, inaccurate when x varies little relative to its level | Sliding co-moments with exact resync | `test_properties_pandas.py::test_rolling_ols_beta_matches_pandas` |
| 6 | 2-D result layout | Same layout as the input frame | Column-major array | `test_properties_layout.py::test_two_d_equals_column_by_column` |

Elsewhere qkern matches pandas within `1e-9 * max(1, |pandas|)` on bounded data
(`|x| <= 1e6`, NFR-06 (d)); where pandas is further from the exact answer than that, qkern
must match the exact `math.fsum` reference instead (`assert_matches_pandas` in
`test_properties_pandas.py`).

## Numerical accuracy

| Target (NFR-06) | Test |
|---|---|
| (a) Gaussian data, n = 1e6: mean error ≤ 1e-12 · max(1, \|exact\|), variance ≤ 1e-10 relative | `test_numerics.py::test_gaussian_mean_and_variance`; C++ `Numerics.GaussianMeanAndVarianceAgainstExactTwoPass` |
| (b) `1e9 + N(0, 1)`: variance ≤ 1e-6 relative, where the naive cumulative-sum method fails | `test_numerics.py::test_offset_data_variance` |
| (c) After 1e8 streaming updates (w = 100): variance ≤ 1e-9 relative | `test_numerics.py::test_streaming_drift_over_1e8_updates` (slow, nightly) |
| (d) Against pandas on `|x| <= 1e6` | `test_properties_pandas.py` |
