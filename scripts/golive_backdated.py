"""Backdated go-live: initialize all 4 strategies with inception signal = last trading day
before a target start date, then replay the SAME live engine (build_rebalance_plan / settle /
mark_to_market) forward through every real historical trading day up to today.

This is NOT a separate backtester — it is the live engine (strategies/portfolio_engine.py)
called once per historical day instead of once per real calendar day, using only prices known
up to each day (no look-ahead). Every monthly rebalance in between fires exactly as it would
have live, using real bootstrapped price/circuit history.

Usage:
    python scripts/golive_backdated.py --target-start 2026-01-01 --notional 10000000
    python scripts/golive_backdated.py --strategy S2 --target-start 2026-01-01
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

from analytics.db import ensure_schema, get_connection, load_circuit_flags_wide, load_field_wide, load_prices_wide
from analytics.strategy_metrics import compute_all_strategy_metrics
from strategies.portfolio_engine import (
    build_rebalance_plan,
    get_strategy_meta,
    initialize_strategy,
    mark_to_market,
    save_pending_rebalance,
    settle_pending_rebalance,
)
from universe.membership import get_universe


def rollforward_one(conn, strategy_key, strategy_cfg, notional_capital, trading_days, closes, opens, circuit_flags, inception_idx):
    print(f"\n=== {strategy_key}: {strategy_cfg['name']} ===")

    if get_strategy_meta(conn, strategy_key) is not None:
        print(f"  Already initialized — skipping (use a fresh DB or manual reset to redo).")
        return
    initialize_strategy(conn, strategy_key, notional_capital)

    universe_symbols = get_universe(conn, strategy_cfg["universe"])
    signal_date = trading_days[inception_idx]
    plan = build_rebalance_plan(conn, strategy_key, universe_symbols, strategy_cfg["rules"], closes, circuit_flags, signal_date)
    save_pending_rebalance(conn, plan)
    print(f"  Inception signal {signal_date.date()}: {len(plan.buys)} buys")

    n_settled, n_marked = 0, 0
    for i in range(inception_idx + 1, len(trading_days)):
        d = trading_days[i]
        result = settle_pending_rebalance(conn, strategy_key, opens, closes, d)
        if result and result.get("status") == "settled":
            n_settled += 1
        mark_to_market(conn, strategy_key, closes, d)
        n_marked += 1
        # EXACT month-end check: we already know the full historical trading-day list here
        # (unlike the live daily case), so use it directly instead of a business-day guess —
        # a plain "next business day" heuristic misses real NSE holidays (e.g. every year the
        # day after March's last trading day is a holiday, which silently skipped the entire
        # March rebalance in an earlier run of this script until this was fixed).
        is_real_month_end = i + 1 < len(trading_days) and trading_days[i + 1].month != d.month
        if is_real_month_end:
            universe_symbols = get_universe(conn, strategy_cfg["universe"])
            new_plan = build_rebalance_plan(conn, strategy_key, universe_symbols, strategy_cfg["rules"], closes, circuit_flags, d)
            try:
                save_pending_rebalance(conn, new_plan)
            except ValueError:
                pass

    print(f"  Replayed {n_marked} trading days, settled {n_settled} rebalances (incl. inception).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="all", help="S1 | S2 | S3 | S4 | all")
    parser.add_argument("--target-start", required=True, help="ISO date; inception uses the last trading day before/at this date")
    parser.add_argument("--notional", type=float, default=None)
    args = parser.parse_args()

    with open(BASE_DIR / "config" / "strategies.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    notional = args.notional or cfg["portfolio_engine"]["notional_capital"]

    db_path = BASE_DIR / "data" / "processed" / "portfolio_dashboard.db"
    conn = get_connection(db_path)
    ensure_schema(conn)

    closes = load_prices_wide(conn)
    opens = load_field_wide(conn, "Open")
    circuit_flags = load_circuit_flags_wide(conn)
    circuit_flags = circuit_flags.reindex(index=closes.index, columns=closes.columns).fillna(0) if not circuit_flags.empty else pd.DataFrame(0, index=closes.index, columns=closes.columns)

    trading_days = sorted(closes.index)
    target = pd.Timestamp(args.target_start)
    inception_idx = max(i for i, d in enumerate(trading_days) if d <= target)
    print(f"Inception signal date resolved to {trading_days[inception_idx].date()} "
          f"(execution/NAV=100 on {trading_days[inception_idx+1].date()})")

    targets = list(cfg["strategies"].keys()) if args.strategy == "all" else [args.strategy]
    for key in targets:
        rollforward_one(conn, key, cfg["strategies"][key], notional, trading_days, closes, opens, circuit_flags, inception_idx)

    compute_all_strategy_metrics(conn, cfg["strategies"], BASE_DIR / "data" / "processed", prices_wide=closes)
    conn.close()
    print("\nDone. Strategy metrics files refreshed.")


if __name__ == "__main__":
    main()
