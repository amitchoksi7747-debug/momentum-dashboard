"""Momentum ranking — pure function: closes DataFrame -> ranks DataFrame.

CONFIRMED spec (docs/momentum_strategies_plan.md section 0): rank the universe separately on
each of the 3/6/9/12-month (63/126/189/252 trading day) Sharpe-style momentum windows, then
score = the average of the four per-window ranks (lower = better). This average is compared
directly against `worst_rank_held` (60) for the hold/exit rule — it is not re-ranked.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from strategies.eligibility import compute_eligibility
from strategies.indicators import window_sharpe


def compute_ranks(
    closes: pd.DataFrame,
    circuit_flags: pd.DataFrame,
    signal_date: pd.Timestamp,
    universe_symbols: Iterable[str],
    rules: dict,
) -> pd.DataFrame:
    """Returns a DataFrame indexed by Symbol (only eligible universe members) with columns:
    Sharpe_<window> / Rank_<window> per window, AvgRank, Close, TrendMA, CircuitDaysTrailing.
    Sorted ascending by AvgRank (best first).
    """
    universe_symbols = [s for s in universe_symbols if s in closes.columns]
    universe_closes = closes[universe_symbols]

    elig = compute_eligibility(universe_closes, circuit_flags, signal_date, rules)
    eligible_symbols = elig.index[elig["Eligible"]]

    windows = rules["windows_trading_days"]
    style = rules.get("sharpe_style", "vol_adjusted_return")

    rank_cols = []
    result = elig.loc[eligible_symbols, ["Close", "TrendMA", "CircuitDaysTrailing"]].copy()

    for window in windows:
        sharpe_today = window_sharpe(universe_closes, window, style=style).loc[signal_date]
        sharpe_today = sharpe_today.loc[eligible_symbols]
        rank_today = sharpe_today.rank(ascending=False, method="min")  # rank 1 = best (highest Sharpe)
        result[f"Sharpe_{window}"] = sharpe_today
        result[f"Rank_{window}"] = rank_today
        rank_cols.append(f"Rank_{window}")

    result["AvgRank"] = result[rank_cols].mean(axis=1, skipna=False)
    result = result.dropna(subset=["AvgRank"]).sort_values("AvgRank")
    return result


def select_top_n(ranks: pd.DataFrame, top_n: int) -> pd.DataFrame:
    return ranks.head(top_n)
