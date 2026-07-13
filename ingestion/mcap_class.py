"""Large/Mid/Small cap classification using NSE's RANK-based definition (CONFIRMED
2026-07-08): rank all universe symbols by market cap, Large = rank 1-100, Mid = rank
101-250, Small = rank 251+. This matches how Nifty Largecap/Midcap/Smallcap indices are
actually defined (relative ranking, not a fixed rupee cutoff) — replacing an earlier
absolute-Rs-Cr-threshold attempt that produced counter-intuitive results (well-known
"midcap" names like Federal Bank/Polycab/MRF are now well above any reasonable fixed
Large-cap cutoff in absolute rupee terms, because the whole market has grown).

Simplification vs NSE's official process: NSE/AMFI rank by *average* market cap over a
trailing 6-month window (semi-annual reclassification) to avoid day-to-day noise flipping a
stock's bucket. We rank by a single latest snapshot instead (no rolling-average market cap
series is maintained here) — acceptable for this system's purpose (a live dashboard pie
chart), but a stock near a rank boundary could occasionally flip bucket day to day where
NSE's own smoothed ranking would not.

Also: this ranks within our ~750-symbol universe (Nifty Total Market: N500 + Microcap250),
not all ~2000+ NSE-listed stocks. Since our universe is already NSE's own market-cap-ranked
top ~750, and we only care about the Large/Mid boundary at rank 250, everything below our
universe's boundary would rank beyond 750 in a full-market ranking anyway — so this doesn't
affect the Large (1-100) / Mid (101-250) split, only pushes what NSE would call very deep
small-cap into our own "Small (251+)" bucket, which is the correct side anyway.

Market cap comes directly from yfinance's per-symbol `.info['marketCap']` field (one network
call per symbol, no bulk endpoint — refresh periodically via scripts/refresh_mcap.py).
"""
from __future__ import annotations

import sqlite3
import time
from typing import Iterable

import pandas as pd
import yfinance as yf

from ingestion.prices import _bse_fallback, normalize_symbol

CRORE = 1e7


def classify_by_rank(market_caps: pd.Series, large_rank_max: int = 100, mid_rank_max: int = 250) -> pd.Series:
    """market_caps: Series indexed by Symbol, values = MarketCap_Cr. Returns a Series of the
    same index with 'Large'/'Mid'/'Small' based on descending-market-cap rank."""
    rank = market_caps.rank(ascending=False, method="first")
    return pd.cut(
        rank,
        bins=[0, large_rank_max, mid_rank_max, float("inf")],
        labels=["Large", "Mid", "Small"],
    ).astype(str)


def fetch_market_cap(symbols: Iterable[str], retries: int = 2) -> pd.DataFrame:
    """Returns Symbol, MarketCap_Cr (None where unavailable)."""
    rows = []
    for original_symbol in symbols:
        yf_symbol = normalize_symbol(original_symbol)
        market_cap = None
        for attempt in range(retries):
            try:
                info = yf.Ticker(yf_symbol).info
                market_cap = info.get("marketCap")
                if market_cap is None:
                    fallback = _bse_fallback(yf_symbol)
                    if fallback:
                        market_cap = yf.Ticker(fallback).info.get("marketCap")
                break
            except Exception:  # noqa: BLE001
                if attempt < retries - 1:
                    time.sleep(1)
        rows.append({
            "Symbol": original_symbol,
            "MarketCap_Cr": (market_cap / CRORE) if market_cap else None,
        })
    return pd.DataFrame(rows)


def refresh_mcap_class(
    conn: sqlite3.Connection, symbols: Iterable[str], rank_thresholds: dict, as_of: str,
    market_caps: pd.DataFrame | None = None,
) -> int:
    """Fetch (or reuse already-fetched) market cap for `symbols`, classify by rank, persist to
    mcap_class keyed by AsOfDate. Pass `market_caps` (Symbol, MarketCap_Cr) to skip re-fetching
    from yfinance (e.g. when only the classification rule changed, not the underlying data).
    Returns rows written."""
    mc = market_caps if market_caps is not None else fetch_market_cap(symbols)
    mc = mc.dropna(subset=["MarketCap_Cr"]).copy()
    mc["Category"] = classify_by_rank(
        mc.set_index("Symbol")["MarketCap_Cr"],
        large_rank_max=rank_thresholds["large_rank_max"],
        mid_rank_max=rank_thresholds["mid_rank_max"],
    ).values
    mc["AsOfDate"] = as_of

    conn.execute("DELETE FROM mcap_class WHERE AsOfDate = ?", (as_of,))
    if not mc.empty:
        conn.executemany(
            "INSERT INTO mcap_class (Symbol, Category, MarketCap_Cr, AsOfDate) VALUES (?, ?, ?, ?)",
            mc[["Symbol", "Category", "MarketCap_Cr", "AsOfDate"]].itertuples(index=False, name=None),
        )
    conn.commit()
    return len(mc)


def get_latest_mcap_class(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT Symbol, Category, MarketCap_Cr, AsOfDate FROM mcap_class
        WHERE AsOfDate = (SELECT MAX(AsOfDate) FROM mcap_class)
        """,
        conn,
    )
