"""Refresh Large/Mid/Small market-cap classification for every symbol across the 4 momentum
universes, using NSE's rank-based definition in config/strategies.yaml (mcap_rank_thresholds):
rank 1-100 by market cap = Large, 101-250 = Mid, 251+ = Small.

One yfinance .info call per symbol — no bulk endpoint exists, so fetching market cap is slow
(~743 symbols takes several minutes); run periodically (e.g. weekly), not every day.

Use --reclassify-only to re-run classification against the most recently fetched market caps
without hitting yfinance again (useful when only the rank thresholds changed).

Usage: python scripts/refresh_mcap.py [--reclassify-only]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import yaml

from analytics.db import ensure_schema, get_connection
from ingestion.mcap_class import get_latest_mcap_class, refresh_mcap_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reclassify-only", action="store_true",
                         help="Reuse the most recently fetched market caps; only re-apply the rank thresholds")
    args = parser.parse_args()

    with open(BASE_DIR / "config" / "strategies.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rank_thresholds = cfg["mcap_rank_thresholds"]

    db_path = BASE_DIR / "data" / "processed" / "portfolio_dashboard.db"
    conn = get_connection(db_path)
    ensure_schema(conn)

    symbols = pd.read_sql_query("SELECT DISTINCT Symbol FROM universe_membership", conn)["Symbol"].tolist()
    print(f"Classifying {len(symbols)} symbols by market-cap rank "
          f"(Large = rank 1-{rank_thresholds['large_rank_max']}, "
          f"Mid = rank {rank_thresholds['large_rank_max']+1}-{rank_thresholds['mid_rank_max']}, "
          f"Small = rank {rank_thresholds['mid_rank_max']+1}+)...")

    market_caps = None
    if args.reclassify_only:
        prior = get_latest_mcap_class(conn)
        if prior.empty:
            print("No previously-fetched market caps found; fetching fresh instead.")
        else:
            market_caps = prior[["Symbol", "MarketCap_Cr"]]
            print(f"Reusing {len(market_caps)} previously-fetched market caps (as of {prior['AsOfDate'].iloc[0]}) — no new yfinance calls.")

    as_of = date.today().isoformat()
    n = refresh_mcap_class(conn, symbols, rank_thresholds, as_of, market_caps=market_caps)
    print(f"Classified {n} of {len(symbols)} symbols (rest had no market cap available).")

    latest = get_latest_mcap_class(conn)
    print(latest["Category"].value_counts())

    conn.close()


if __name__ == "__main__":
    main()
