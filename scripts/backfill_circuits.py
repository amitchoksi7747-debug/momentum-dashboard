"""Backfill trailing ~252 trading days of circuit-lock flags from NSE bhavcopy into circuit_days.

This is a separate, slower backfill (one bhavcopy download per trading day) from the price
bootstrap, since circuit/price-band behavior isn't carried in yfinance data at all. Safe to
re-run (upsert keyed by Symbol+Date). Weekends/holidays are skipped automatically (bhavcopy
404s on non-trading days).

Usage: python scripts/backfill_circuits.py [--trading-days 252] [--max-calendar-lookback 400]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd

from analytics.db import ensure_schema, get_connection
from ingestion.providers.bhavcopy import _get_session, fetch_bhavcopy, flag_circuit_days


def upsert_circuit_days(conn, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    conn.executemany(
        """
        INSERT INTO circuit_days (Symbol, Date, IsCircuit) VALUES (?, ?, ?)
        ON CONFLICT(Symbol, Date) DO UPDATE SET IsCircuit=excluded.IsCircuit
        """,
        df[["Symbol", "Date", "IsCircuit"]].itertuples(index=False, name=None),
    )
    conn.commit()
    return len(df)


def backfill(trading_days: int = 252, max_calendar_lookback: int = 400) -> None:
    db_path = BASE_DIR / "data" / "processed" / "portfolio_dashboard.db"
    conn = get_connection(db_path)
    ensure_schema(conn)

    session = _get_session()
    collected = 0
    total_circuit_rows = 0
    d = date.today()
    checked = 0

    while collected < trading_days and checked < max_calendar_lookback:
        checked += 1
        d = d - timedelta(days=1)
        if d.weekday() >= 5:  # Sat/Sun — bhavcopy won't exist, skip without counting as a failed attempt
            continue
        try:
            bhav = fetch_bhavcopy(d, session=session)
        except Exception as exc:  # noqa: BLE001
            print(f"  {d.isoformat()}: ERROR {exc}")
            time.sleep(1)
            continue
        if bhav.empty:
            continue  # holiday
        circuits = flag_circuit_days(bhav)
        n = upsert_circuit_days(conn, circuits)
        total_circuit_rows += int(circuits["IsCircuit"].sum())
        collected += 1
        if collected % 25 == 0:
            print(f"  ...{collected}/{trading_days} trading days collected (as of {d.isoformat()})")
        time.sleep(0.3)  # be polite to NSE's archive server

    print(f"Backfilled {collected} trading days ({checked} calendar days scanned). "
          f"Total circuit-lock symbol-days flagged: {total_circuit_rows}.")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trading-days", type=int, default=252)
    parser.add_argument("--max-calendar-lookback", type=int, default=400)
    args = parser.parse_args()
    backfill(trading_days=args.trading_days, max_calendar_lookback=args.max_calendar_lookback)
