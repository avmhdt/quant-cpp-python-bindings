# Using qkern in other repos

qkern was written for the owner's other research projects. The table lists which kernels
each one uses; integration pull requests replace their own rolling code with qkern and keep
a test showing equality with the code they replace.

| Repository | Kernels | Use |
|---|---|---|
| equity-factor-pipeline | `rolling_zscore`, `rolling_std`, `rolling_ols` (2-D), `rolling_quantile` | Factor standardisation, rolling market beta and residuals, winsorisation bounds |
| crypto-lob-alpha | `ewm_mean`, `rolling_zscore` (order-flow helpers later) | Smoothed and standardised order-book signals |
| stat-arb-pairs | `rolling_ols` (`include_current=False`), `rolling_zscore`, `RollingOLS` | Rolling hedge ratio and out-of-sample spread z-score |
| regime-portfolio-construction | `ewm_var`, `ewm_std`, `rolling_std`, `rolling_max` | Volatility targeting, regime features, running peak for drawdown |
| market-making-sim | `EWM`, `RollingMinMax` | Per-event volatility and short-horizon range |

A market beta for every stock at once:

```python
import qkern

fit = qkern.rolling_ols(stock_returns, market_returns, 252)  # (n, k) on (n,)
beta, idio = fit.beta, fit.resid
```

Status of each integration is tracked on the [Adoption](adoption.md) page.
