from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def drawdown_series(nav: pd.Series) -> pd.Series:
    """Drawdown at each point = nav / running-max(nav) - 1."""
    return nav / nav.cummax() - 1


def sortino(nav: pd.Series, target_return: float = 0.0) -> float | None:
    """Annualized return / annualized downside deviation (only return periods below target)."""
    rets = nav.pct_change().dropna()
    if rets.empty:
        return None
    downside = rets[rets < target_return]
    if downside.empty:
        return None
    downside_dev = float(np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS_PER_YEAR))
    if downside_dev == 0:
        return None
    ann_ret = float(rets.mean() * TRADING_DAYS_PER_YEAR)
    return ann_ret / downside_dev


def calmar(nav: pd.Series) -> float | None:
    """CAGR / |max drawdown|."""
    if nav.empty or nav.iloc[0] == 0:
        return None
    years = max(len(nav) - 1, 1) / TRADING_DAYS_PER_YEAR
    cagr = float(nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1 if years > 0 else None
    dd = drawdown_series(nav)
    max_dd = abs(float(dd.min())) if not dd.empty else 0
    if cagr is None or max_dd == 0:
        return None
    return cagr / max_dd


def rolling_returns(nav: pd.Series, window_trading_days: int) -> pd.Series:
    """Rolling total return over a trailing window (e.g. 252 for rolling-1Y)."""
    return nav.pct_change(periods=window_trading_days, fill_method=None)


def monthly_return_table(nav: pd.Series) -> pd.DataFrame:
    """Pivot table: rows=Year, columns=Month(1-12), values=monthly return %. Tolerant of
    short series (returns whatever months exist so far)."""
    if nav.empty:
        return pd.DataFrame()
    s = nav.copy()
    s.index = pd.to_datetime(s.index)
    # pandas >= 2.2 REMOVED the "M" (month-end) alias and requires "ME"; pandas < 2.2 only knows
    # "M". Try the new alias first so it works on GitHub Actions / Streamlit Cloud (newer pandas),
    # falling back to "M" on older local installs.
    try:
        monthly_nav = s.resample("ME").last()
    except ValueError:
        monthly_nav = s.resample("M").last()
    monthly_ret = monthly_nav.pct_change(fill_method=None)
    monthly_ret.iloc[0] = monthly_nav.iloc[0] / s.iloc[0] - 1  # first partial month vs series start
    df = monthly_ret.to_frame("Return")
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    return df.pivot(index="Year", columns="Month", values="Return")


def alpha_beta(nav: pd.Series, benchmark_nav: pd.Series) -> dict:
    """Simple daily-returns regression beta and annualized alpha vs a benchmark NAV series.
    Returns {'alpha': None, 'beta': None} if there's insufficient overlapping history."""
    aligned = pd.concat([nav.rename("s"), benchmark_nav.rename("b")], axis=1).dropna()
    if len(aligned) < 20:
        return {"alpha": None, "beta": None}
    s_ret = aligned["s"].pct_change(fill_method=None).dropna()
    b_ret = aligned["b"].pct_change(fill_method=None).dropna()
    aligned_ret = pd.concat([s_ret, b_ret], axis=1).dropna()
    if len(aligned_ret) < 20 or aligned_ret.iloc[:, 1].var() == 0:
        return {"alpha": None, "beta": None}
    cov = aligned_ret.cov()
    beta = float(cov.iloc[0, 1] / cov.iloc[1, 1])
    ann_s = float(aligned_ret.iloc[:, 0].mean() * TRADING_DAYS_PER_YEAR)
    ann_b = float(aligned_ret.iloc[:, 1].mean() * TRADING_DAYS_PER_YEAR)
    alpha = ann_s - beta * ann_b
    return {"alpha": alpha, "beta": beta}


def compute_risk_stats(timeseries: pd.DataFrame, start_date: str | None = None) -> dict:
    """
    Compute risk statistics for the portfolio timeseries.

    Parameters
    ----------
    timeseries : DataFrame with columns Date, PortfolioValue, PortfolioReturn, PortfolioNAV
    start_date : Optional ISO date string (e.g. '2026-01-01'). If given, all stats are
                 computed from that date onwards so they stay consistent with the equity
                 curve displayed on the dashboard.
    """
    _empty = {
        "annualized_return": None,
        "annualized_vol": None,
        "sharpe": None,
        "max_drawdown": None,
        "cagr": None,
        "period_return": None,
        "risk_start_date": start_date,
    }

    if timeseries.empty:
        return _empty

    ts = timeseries.copy()
    ts["Date"] = pd.to_datetime(ts["Date"], errors="coerce")

    if start_date:
        ts = ts[ts["Date"] >= pd.Timestamp(start_date)].copy()

    if ts.empty:
        return _empty

    # Re-base NAV to 1.0 at the start of the chosen window so all metrics
    # are self-consistent with that window.
    nav_raw = pd.to_numeric(ts["PortfolioNAV"], errors="coerce").dropna()
    if nav_raw.empty or nav_raw.iloc[0] == 0:
        return _empty

    nav = nav_raw / nav_raw.iloc[0]                    # rebased to 1.0
    rets = nav.pct_change().dropna()

    if rets.empty:
        return _empty

    trading_days = max(len(rets), 1)
    years = trading_days / 252

    ann_vol = float(rets.std(ddof=1) * np.sqrt(252)) if len(rets) > 1 else 0.0

    # CAGR: (end/start)^(1/years) - 1
    cagr = float(nav.iloc[-1] ** (1 / years) - 1) if years > 0 else None

    # Simple period return (no annualization — more honest for short windows)
    period_return = float(nav.iloc[-1] - 1)

    ann_ret = float(rets.mean() * 252)
    sharpe = ann_ret / ann_vol if ann_vol not in (0, None) else None

    # Max drawdown over the chosen window
    drawdown = nav / nav.cummax() - 1
    max_dd = float(drawdown.min())

    return {
        "annualized_return": ann_ret,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "cagr": cagr,
        "period_return": period_return,
        "risk_start_date": start_date,
    }
