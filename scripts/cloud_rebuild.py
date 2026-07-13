"""Deterministic cloud rebuild — the entry point GitHub Actions runs on a schedule.

The 151MB price DB is far too big for GitHub, so nothing stateful is committed. Instead this
script rebuilds everything from scratch each run, which is safe because the strategies are
deterministic given (prices, inception date, rules):

  1. refresh the 4 index constituent lists live from niftyindices.com (keyless)
  2. load the two things too slow to regenerate — circuit flags & mcap classes — from the
     small committed CSVs in data/state/
  3. fetch ~720 calendar days of prices fresh (enough for 252d Sharpe + 200d DEMA warm-up)
  4. replay the live engine from the fixed inception date to today (same code as golive_backdated)
  5. write the dashboard's metric files under data/processed/strategies/

The Actions workflow then commits data/processed/strategies/ so Streamlit Cloud serves the
fresh numbers. Prices/DB stay ephemeral (gitignored).

Usage: python scripts/cloud_rebuild.py [--inception 2026-01-01]
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

from analytics.db import (ensure_schema, get_connection, load_circuit_flags_wide, load_field_wide,
                          load_prices_wide, replace_universe_membership, upsert_prices_eod)
from analytics.strategy_metrics import compute_all_strategy_metrics
from ingestion.providers.yf_provider import YFinanceProvider
from scripts.bootstrap_prices import apply_sanity_gate
from scripts.golive_backdated import rollforward_one
from universe.constituents import refresh_all_constituents

STATE_DIR = BASE_DIR / "data" / "state"


def _load_state_csv(conn, table: str, csv_path: Path, columns: list[str]) -> int:
    if not csv_path.exists():
        print(f"  WARNING: {csv_path.name} missing — {table} will be empty this run.")
        return 0
    df = pd.read_csv(csv_path)
    df = df[[c for c in columns if c in df.columns]]
    conn.execute(f"DELETE FROM {table}")
    conn.executemany(
        f"INSERT INTO {table} ({','.join(df.columns)}) VALUES ({','.join('?' for _ in df.columns)})",
        df.itertuples(index=False, name=None),
    )
    conn.commit()
    return len(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inception", default="2026-01-01")
    # ~1000 calendar days ≈ back to late 2023: enough trading history for the 200-DEMA to
    # fully converge BEFORE the Jan-2026 inception in a FRESH environment (GitHub Actions /
    # a clean clone start with an empty price DB). 720 was too short there → 0 eligible stocks.
    parser.add_argument("--history-days", type=int, default=1000)
    args = parser.parse_args()

    with open(BASE_DIR / "config" / "strategies.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    notional = cfg["portfolio_engine"]["notional_capital"]

    conn = get_connection(BASE_DIR / "data" / "processed" / "portfolio_dashboard.db")
    ensure_schema(conn)

    # 1. universe (live) ------------------------------------------------------
    print("Refreshing universe constituents from niftyindices.com ...")
    constituents = refresh_all_constituents(raw_cache_dir=BASE_DIR / "data" / "raw" / "universe")
    as_of = date.today().isoformat()
    for index_key, df in constituents.items():
        df2 = df.copy(); df2["IndexName"] = index_key; df2["AsOfDate"] = as_of
        replace_universe_membership(conn, index_key, df2[["IndexName", "Symbol", "CompanyName", "Industry", "ISIN", "AsOfDate"]])
    symbols = pd.read_sql_query("SELECT DISTINCT Symbol FROM universe_membership", conn)["Symbol"].tolist()
    print(f"  {len(symbols)} symbols across {len(constituents)} universes")

    # 2. committed state (circuits + mcap) ------------------------------------
    n_circ = _load_state_csv(conn, "circuit_days", STATE_DIR / "circuit_days.csv", ["Symbol", "Date", "IsCircuit"])
    n_mcap = _load_state_csv(conn, "mcap_class", STATE_DIR / "mcap_class.csv", ["Symbol", "Category", "MarketCap_Cr", "AsOfDate"])
    print(f"  loaded {n_circ} circuit flags, {n_mcap} mcap classes from data/state/")

    # 3. prices (fresh) -------------------------------------------------------
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=args.history_days)
    print(f"Fetching {args.history_days}d of prices for {len(symbols)} symbols ...")
    raw = YFinanceProvider().get_daily(symbols, start=start.isoformat(), end=end.isoformat())
    upsert_prices_eod(conn, apply_sanity_gate(raw))

    closes = load_prices_wide(conn)
    opens = load_field_wide(conn, "Open")
    circuit_flags = load_circuit_flags_wide(conn)
    circuit_flags = circuit_flags.reindex(index=closes.index, columns=closes.columns).fillna(0) if not circuit_flags.empty else pd.DataFrame(0, index=closes.index, columns=closes.columns)
    trading_days = sorted(closes.index)
    inception_idx = max(i for i, d in enumerate(trading_days) if d <= pd.Timestamp(args.inception))
    print(f"Prices to {trading_days[-1].date()}; inception {trading_days[inception_idx].date()}")

    # 4. replay engine --------------------------------------------------------
    for t in ["strategy_meta", "strategy_state", "strategy_nav", "strategy_rebalances", "strategy_trades"]:
        conn.execute(f"DELETE FROM {t}")
    conn.commit()
    for key in cfg["strategies"]:
        rollforward_one(conn, key, cfg["strategies"][key], notional, trading_days, closes, opens, circuit_flags, inception_idx)

    # 5. dashboard metric files ----------------------------------------------
    compute_all_strategy_metrics(conn, cfg["strategies"], BASE_DIR / "data" / "processed", prices_wide=closes)
    conn.close()

    # 6. mutual-fund comparison (independent of the strategy engine) ----------
    try:
        from analytics.mf_comparison import build_fund_comparison
        build_fund_comparison(BASE_DIR)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: fund comparison update failed (dashboard keeps last good data): {exc}")

    print(f"\nRebuilt & wrote dashboard metrics to {trading_days[-1].date()}.")


if __name__ == "__main__":
    main()
