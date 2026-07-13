from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import List, Optional

import pandas as pd

STALENESS_DAYS = 190  # niftyindices reconstitutes semi-annually; flag if older than ~6.3 months


def get_universe(conn: sqlite3.Connection, index_name: str) -> List[str]:
    """Current constituents only (v1 — no point-in-time membership; see docs/momentum_strategies_plan.md).

    Returns a list of canonical symbols (NSE:X form).
    """
    df = pd.read_sql_query(
        "SELECT Symbol FROM universe_membership WHERE IndexName = ?", conn, params=(index_name,)
    )
    return df["Symbol"].tolist()


def get_universe_frame(conn: sqlite3.Connection, index_name: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM universe_membership WHERE IndexName = ?", conn, params=(index_name,)
    )


def get_sector_map(conn: sqlite3.Connection) -> pd.DataFrame:
    """Symbol -> Industry, deduplicated (a symbol may appear in multiple index lists; Industry
    should be identical across lists — take the first non-null)."""
    df = pd.read_sql_query("SELECT Symbol, Industry FROM universe_membership", conn)
    return df.dropna(subset=["Industry"]).drop_duplicates(subset=["Symbol"], keep="first").reset_index(drop=True)


def membership_staleness_days(conn: sqlite3.Connection, index_name: str) -> Optional[int]:
    df = pd.read_sql_query(
        "SELECT MAX(AsOfDate) as latest FROM universe_membership WHERE IndexName = ?",
        conn,
        params=(index_name,),
    )
    latest = df["latest"].iloc[0] if not df.empty else None
    if not latest:
        return None
    latest_date = datetime.fromisoformat(str(latest)[:10])
    return (datetime.utcnow() - latest_date).days


def is_stale(conn: sqlite3.Connection, index_name: str) -> bool:
    days = membership_staleness_days(conn, index_name)
    return days is None or days > STALENESS_DAYS
