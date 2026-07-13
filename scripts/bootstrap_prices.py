"""One-shot bootstrap: pull ~450 trading days of adjusted daily OHLCV for every symbol across
the 4 momentum universes into prices_eod. Safe to re-run (upsert keyed by Symbol+Date).

Usage: python scripts/bootstrap_prices.py [--calendar-days 700] [--skip-universe-refresh]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import yaml

from analytics.db import ensure_schema, get_connection, replace_universe_membership, upsert_prices_eod
from ingestion.providers.yf_provider import YFinanceProvider
from universe.constituents import INDEX_CSV_MAP, refresh_all_constituents

MOVE_QUARANTINE_THRESHOLD = 0.25  # |day-over-day return| beyond this, absent a known corporate
                                    # action (not yet wired), gets flagged rather than trusted.


def apply_sanity_gate(prices: pd.DataFrame) -> pd.DataFrame:
    """Flag Quarantined=1 on bars with an implausible day-over-day move. First bar per symbol
    (no prior close to compare) is never quarantined."""
    out = prices.sort_values(["Symbol", "Date"]).copy()
    out["Quarantined"] = 0
    prev_close = out.groupby("Symbol")["Close"].shift(1)
    move = (out["Close"] - prev_close).abs() / prev_close
    flagged = move > MOVE_QUARANTINE_THRESHOLD
    out.loc[flagged.fillna(False), "Quarantined"] = 1
    return out


def bootstrap(calendar_days: int = 700, skip_universe_refresh: bool = False) -> None:
    with open(BASE_DIR / "config" / "strategies.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    db_path = BASE_DIR / "data" / "processed" / "portfolio_dashboard.db"
    conn = get_connection(db_path)
    ensure_schema(conn)

    if not skip_universe_refresh:
        print("Refreshing universe constituent lists from niftyindices.com ...")
        constituents = refresh_all_constituents(raw_cache_dir=BASE_DIR / "data" / "raw" / "universe")
        as_of = date.today().isoformat()
        for index_key, df in constituents.items():
            df2 = df.copy()
            df2["IndexName"] = index_key
            df2["AsOfDate"] = as_of
            df2 = df2[["IndexName", "Symbol", "CompanyName", "Industry", "ISIN", "AsOfDate"]]
            n = replace_universe_membership(conn, index_key, df2)
            print(f"  {index_key}: {n} constituents")

    universe_symbols = pd.read_sql_query(
        "SELECT DISTINCT Symbol FROM universe_membership", conn
    )["Symbol"].tolist()
    print(f"Bootstrapping prices for {len(universe_symbols)} unique symbols across {len(INDEX_CSV_MAP)} universes")

    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=calendar_days)

    provider = YFinanceProvider()
    raw = provider.get_daily(universe_symbols, start=start.isoformat(), end=end.isoformat())
    print(f"Fetched {len(raw)} raw bars for {raw['Symbol'].nunique()} symbols")

    missing = set(universe_symbols) - set(raw["Symbol"].unique())
    if missing:
        print(f"WARNING: {len(missing)} symbols returned no data (delisted/renamed/illiquid?): "
              f"{sorted(missing)[:15]}{'...' if len(missing) > 15 else ''}")

    gated = apply_sanity_gate(raw)
    n_quarantined = int(gated["Quarantined"].sum())
    if n_quarantined:
        print(f"WARNING: {n_quarantined} bars quarantined (|move| > {MOVE_QUARANTINE_THRESHOLD:.0%}, "
              f"no corporate-action check wired yet — review before trusting these dates).")

    written = upsert_prices_eod(conn, gated)
    print(f"Wrote {written} rows to prices_eod.")

    coverage = gated[gated["Quarantined"] == 0].groupby("Symbol")["Date"].count()
    thin = coverage[coverage < 200]
    if not thin.empty:
        print(f"NOTE: {len(thin)} symbols have < 200 trading days of history (recent IPOs or "
              f"data gaps) — they will fail the 252-day min-history eligibility gate until more "
              f"history accumulates.")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calendar-days", type=int, default=700)
    parser.add_argument("--skip-universe-refresh", action="store_true")
    args = parser.parse_args()
    bootstrap(calendar_days=args.calendar_days, skip_universe_refresh=args.skip_universe_refresh)
