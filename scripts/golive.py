"""Go-live: seed strategy_meta with starting notional capital and generate the FIRST pending
rebalance sheet for one or all of the 4 momentum strategies.

This does NOT immediately buy anything — it follows the same two-phase flow as every monthly
rebalance: this script decides and saves a PENDING plan; the next time the daily orchestrator
runs (orchestrator/daily_task.py) and finds a trading day after the signal date with real
prices, it settles the plan at that day's open and the NAV curve starts from there at 100.

Deliberately requires an explicit --signal-date (the most recent trading day's data actually
in prices_eod) so this can't be triggered by accident. Run scripts/bootstrap_prices.py and
scripts/backfill_circuits.py first so signals are computed on real, current data.

Usage:
    python scripts/golive.py --strategy S2 --signal-date 2026-07-07 --notional 10000000
    python scripts/golive.py --strategy all --signal-date 2026-07-07 --notional 10000000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import yaml

from analytics.db import ensure_schema, get_connection, load_circuit_flags_wide, load_prices_wide
from strategies.portfolio_engine import build_rebalance_plan, get_strategy_meta, initialize_strategy, save_pending_rebalance
from universe.membership import get_universe, is_stale


def golive_one(conn, strategy_key: str, strategy_cfg: dict, signal_date: pd.Timestamp, notional_capital: float,
                closes: pd.DataFrame, circuit_flags: pd.DataFrame) -> None:
    print(f"\n=== {strategy_key}: {strategy_cfg['name']} ===")

    if is_stale(conn, strategy_cfg["universe"]):
        print(f"  WARNING: {strategy_cfg['universe']} constituent list is stale (>190 days) — "
              f"re-run scripts/bootstrap_prices.py to refresh before going live.")

    existing_meta = get_strategy_meta(conn, strategy_key)
    if existing_meta is not None:
        print(f"  Already initialized (inception={existing_meta['InceptionDate']}). Skipping re-init.")
    else:
        initialize_strategy(conn, strategy_key, notional_capital)
        print(f"  Initialized with notional capital {notional_capital:,.0f}.")

    universe_symbols = get_universe(conn, strategy_cfg["universe"])
    if signal_date not in closes.index:
        raise ValueError(f"signal_date {signal_date.date()} has no price data in prices_eod — "
                          f"re-run scripts/bootstrap_prices.py to fetch the latest bars first.")

    plan = build_rebalance_plan(conn, strategy_key, universe_symbols, strategy_cfg["rules"], closes, circuit_flags, signal_date)
    save_pending_rebalance(conn, plan, force=True)

    print(f"  Signal date: {plan.signal_date}")
    print(f"  Buys ({len(plan.buys)}): {[b['Symbol'] for b in plan.buys][:10]}{'...' if len(plan.buys) > 10 else ''}")
    print(f"  This PENDING plan will be filled automatically at the next trading day's open "
          f"the next time the daily pipeline runs (orchestrator/daily_task.py).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True, help="S1 | S2 | S3 | S4 | all")
    parser.add_argument("--signal-date", required=True, help="ISO date, must exist in prices_eod")
    parser.add_argument("--notional", type=float, default=None, help="Starting notional capital (defaults to config/strategies.yaml portfolio_engine.notional_capital)")
    args = parser.parse_args()

    with open(BASE_DIR / "config" / "strategies.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    notional = args.notional or cfg["portfolio_engine"]["notional_capital"]
    signal_date = pd.Timestamp(args.signal_date)

    db_path = BASE_DIR / "data" / "processed" / "portfolio_dashboard.db"
    conn = get_connection(db_path)
    ensure_schema(conn)

    closes = load_prices_wide(conn)
    circuit_flags = load_circuit_flags_wide(conn)
    if circuit_flags.empty:
        circuit_flags = pd.DataFrame(0, index=closes.index, columns=closes.columns)
    else:
        circuit_flags = circuit_flags.reindex(index=closes.index, columns=closes.columns).fillna(0)

    targets = list(cfg["strategies"].keys()) if args.strategy == "all" else [args.strategy]
    for key in targets:
        golive_one(conn, key, cfg["strategies"][key], signal_date, notional, closes, circuit_flags)

    conn.close()


if __name__ == "__main__":
    main()
