"""Eligibility gates applied BEFORE ranking, so ranks 1..N always refer to eligible names.

Per the confirmed spec, these same gates are re-checked for every holding at every monthly
rebalance (not just at entry) — a held stock that fails any gate becomes ineligible and is
therefore excluded from that rebalance's ranking, which triggers its exit.
"""
from __future__ import annotations

import pandas as pd

from strategies.indicators import dema, ema, sma


def _trend_ma(closes: pd.DataFrame, ma_type: str, period: int) -> pd.DataFrame:
    if ma_type == "dema":
        return dema(closes, period)
    if ma_type == "ema":
        return ema(closes, period)
    if ma_type == "sma":
        return sma(closes, period)
    raise ValueError(f"Unknown ma_type {ma_type!r}")


def compute_eligibility(
    closes: pd.DataFrame,
    circuit_flags: pd.DataFrame,
    signal_date: pd.Timestamp,
    rules: dict,
) -> pd.DataFrame:
    """Return a DataFrame indexed by Symbol with columns:
    Close, TrendMA, HistoryOK, TrendOK, CircuitDaysTrailing, CircuitOK, Eligible.

    `closes` and `circuit_flags` are wide (Date index, Symbol columns); circuit_flags may be
    missing entirely for a symbol/date (treated as 0 = no circuit that day).
    """
    if signal_date not in closes.index:
        raise ValueError(f"signal_date {signal_date} not present in closes index")

    symbols = closes.columns
    history_count = closes.loc[:signal_date].count()
    history_ok = history_count >= rules["min_history_trading_days"]

    close_today = closes.loc[signal_date]

    trend_cfg = rules.get("trend_filter", {})
    if trend_cfg.get("enabled", True):
        trend_series = _trend_ma(closes, trend_cfg.get("ma_type", "dema"), trend_cfg.get("period", 200))
        trend_ma_today = trend_series.loc[signal_date]
        trend_ok = close_today > trend_ma_today
    else:
        trend_ma_today = pd.Series(index=symbols, dtype=float)
        trend_ok = pd.Series(True, index=symbols)

    circuit_cfg = rules.get("circuit_filter", {})
    if circuit_cfg.get("enabled", True):
        lookback = circuit_cfg.get("lookback_trading_days", 252)
        cf = circuit_flags.reindex(columns=symbols).fillna(0)
        window = cf.loc[:signal_date].tail(lookback)
        circuit_days_trailing = window.sum()
        circuit_ok = circuit_days_trailing <= circuit_cfg.get("max_circuit_days", 25)
    else:
        circuit_days_trailing = pd.Series(0, index=symbols)
        circuit_ok = pd.Series(True, index=symbols)

    out = pd.DataFrame(
        {
            "Close": close_today,
            "TrendMA": trend_ma_today,
            "HistoryOK": history_ok,
            "TrendOK": trend_ok.fillna(False),
            "CircuitDaysTrailing": circuit_days_trailing,
            "CircuitOK": circuit_ok,
        }
    )
    out["Eligible"] = out["HistoryOK"] & out["TrendOK"] & out["CircuitOK"]
    return out
