"""One-command catch-up: bring the momentum dashboard current to the latest trading day.

Pulls recent prices, rolls every live strategy's NAV forward across ALL trading days
missing since its last NAV row (so the equity curve stays continuous, not a jump),
settles any pending rebalance, fires a month-end rebalance if the rollover happened,
and recomputes the dashboard metric files.

Run this whenever the dashboard's "as of" date is behind today (there is no scheduler
running the daily pipeline automatically — see docs/momentum_strategies_plan.md Phase 7).

Usage: python scripts/update_dashboard.py [--lookback-days 15]
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

from analytics.db import ensure_schema, get_connection, load_circuit_flags_wide, load_field_wide, load_prices_wide, upsert_prices_eod
from analytics.strategy_metrics import compute_all_strategy_metrics
from ingestion.providers.yf_provider import YFinanceProvider
from scripts.bootstrap_prices import apply_sanity_gate
from strategies.portfolio_engine import (
    build_rebalance_plan, get_strategy_meta, mark_to_market,
    save_pending_rebalance, settle_pending_rebalance,
)
from universe.membership import get_universe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=15,
                         help="Calendar days of prices to re-fetch (must cover the gap since last update)")
    args = parser.parse_args()

    with open(BASE_DIR / "config" / "strategies.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    conn = get_connection(BASE_DIR / "data" / "processed" / "portfolio_dashboard.db")
    ensure_schema(conn)

    symbols = pd.read_sql_query("SELECT DISTINCT Symbol FROM universe_membership", conn)["Symbol"].tolist()
    if not symbols:
        print("universe_membership is empty — run scripts/bootstrap_prices.py first.")
        return

    # 1. refresh recent prices
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=args.lookback_days)
    print(f"Fetching prices {start} → {date.today()} for {len(symbols)} symbols...")
    raw = YFinanceProvider().get_daily(symbols, start=start.isoformat(), end=end.isoformat())
    gated = apply_sanity_gate(raw)
    upsert_prices_eod(conn, gated)
    if int(gated["Quarantined"].sum()):
        print(f"  WARNING: {int(gated['Quarantined'].sum())} bars quarantined (implausible move) — review before trusting.")

    closes = load_prices_wide(conn)
    opens = load_field_wide(conn, "Open")
    circuit_flags = load_circuit_flags_wide(conn)
    circuit_flags = circuit_flags.reindex(index=closes.index, columns=closes.columns).fillna(0) if not circuit_flags.empty else pd.DataFrame(0, index=closes.index, columns=closes.columns)
    trading_days = sorted(closes.index)
    latest = trading_days[-1]
    print(f"Latest trading day in data: {latest.date()}")

    # 2. roll each live strategy forward across every missing trading day
    for key, scfg in cfg["strategies"].items():
        meta = get_strategy_meta(conn, key)
        if meta is None or meta.get("InceptionDate") is None:
            continue
        last_nav = conn.execute("SELECT MAX(Date) FROM strategy_nav WHERE Strategy=?", (key,)).fetchone()[0]
        last_nav = pd.Timestamp(last_nav) if last_nav else None
        new_days = [d for d in trading_days if last_nav is None or d > last_nav]

        settled = 0
        for i, d in enumerate(new_days):
            # settle any pending rebalance whose exec day has arrived
            res = settle_pending_rebalance(conn, key, opens, closes, d)
            if res and res.get("status") == "settled":
                settled += 1
            mark_to_market(conn, key, closes, d)
            # month rollover: the previous trading day was the last of its month
            idx = trading_days.index(d)
            if idx > 0 and trading_days[idx - 1].month != d.month:
                universe = get_universe(conn, scfg["universe"])
                plan = build_rebalance_plan(conn, key, universe, scfg["rules"], closes, circuit_flags, trading_days[idx - 1])
                try:
                    save_pending_rebalance(conn, plan)
                    r2 = settle_pending_rebalance(conn, key, opens, closes, d)
                    if r2 and r2.get("status") == "settled":
                        settled += 1
                except ValueError:
                    pass
        print(f"  {key}: +{len(new_days)} trading days to {latest.date()}"
              + (f", {settled} rebalance(s) settled" if settled else ""))

    # 3. refresh dashboard metric files
    compute_all_strategy_metrics(conn, cfg["strategies"], BASE_DIR / "data" / "processed", prices_wide=closes)
    conn.close()

    # 4. mutual-fund comparison (MFAPI)
    try:
        from analytics.mf_comparison import build_fund_comparison
        build_fund_comparison(BASE_DIR)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: fund comparison update failed: {exc}")

    print("Done — dashboard metrics refreshed to", latest.date())


if __name__ == "__main__":
    main()
