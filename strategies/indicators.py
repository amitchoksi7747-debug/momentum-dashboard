"""Pure, vectorized indicator functions. No I/O.

All functions take/return a wide DataFrame: index = Date (sorted ascending), columns = Symbol.
This lets the same function serve every signal date at once instead of looping per-date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def ema(closes: pd.DataFrame, span: int) -> pd.DataFrame:
    return closes.ewm(span=span, adjust=False, min_periods=span).mean()


def sma(closes: pd.DataFrame, period: int) -> pd.DataFrame:
    return closes.rolling(window=period, min_periods=period).mean()


def dema(closes: pd.DataFrame, period: int) -> pd.DataFrame:
    """True Double EMA: 2*EMA(n) - EMA(EMA(n), n)."""
    ema1 = ema(closes, period)
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    return 2 * ema1 - ema2


def window_return(closes: pd.DataFrame, window: int) -> pd.DataFrame:
    """Total return over the trailing `window` trading days: close[t] / close[t-window] - 1."""
    return closes.pct_change(periods=window, fill_method=None)


def window_vol(closes: pd.DataFrame, window: int, annualize: bool = True) -> pd.DataFrame:
    """Rolling std-dev of daily returns over the trailing `window` trading days."""
    daily_returns = closes.pct_change(fill_method=None)
    vol = daily_returns.rolling(window=window, min_periods=window).std(ddof=1)
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    return vol


def window_sharpe(closes: pd.DataFrame, window: int, style: str = "vol_adjusted_return", risk_free_annual: float = 0.0) -> pd.DataFrame:
    """Sharpe-style momentum score over a trailing window.

    style='vol_adjusted_return' (default, matches NSE's own momentum-index methodology):
        window total return / annualized daily volatility over the window.
    style='classic':
        (annualized window return - risk_free) / annualized volatility.
    """
    ret = window_return(closes, window)
    vol = window_vol(closes, window, annualize=True)

    if style == "vol_adjusted_return":
        score = ret / vol
    elif style == "classic":
        years = window / TRADING_DAYS_PER_YEAR
        ann_ret = (1 + ret) ** (1 / years) - 1
        score = (ann_ret - risk_free_annual) / vol
    else:
        raise ValueError(f"Unknown sharpe_style {style!r}")

    return score.replace([np.inf, -np.inf], np.nan)
