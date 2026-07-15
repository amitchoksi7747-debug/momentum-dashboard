"""Robust cloud rebuild — the entry point GitHub Actions runs on a schedule.

DESIGN FOR RELIABILITY IN CI: external services (Yahoo Finance, niftyindices.com) frequently
throttle or block requests from datacenter IPs like GitHub Actions'. The earlier version
re-downloaded ~1000 days × 750 symbols from Yahoo AND fetched the universe live every run, and
both are exactly the calls that get blocked — so the job failed daily.

This version commits everything the rebuild needs as small files under data/state/ (universe,
circuit flags, mcap classes, and a slim Open/Close price history) and each run only:
  1. loads that committed state (no large external fetch),
  2. does a SMALL incremental price fetch for just the recent days — and if that fails, it
     logs a warning and proceeds with the committed prices instead of crashing,
  3. replays the strategies + rebuilds the fund comparison,
  4. re-exports the rolled-forward slim prices so the committed history moves with time.

Net effect: the job SUCCEEDS every run (never crashes on a blocked fetch) and updates whenever
the small fetch gets through — which a ~10-day request almost always does, unlike a 1000-day one.

Usage: python scripts/cloud_rebuild.py [--inception 2026-01-01] [--incremental-days 12]
       python scripts/cloud_rebuild.py --full-refetch   # (run locally) re-pull all history
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
                          load_prices_wide, upsert_prices_eod)
from analytics.strategy_metrics import compute_all_strategy_metrics
from scripts.bootstrap_prices import apply_sanity_gate
from scripts.golive_backdated import rollforward_one

STATE_DIR = BASE_DIR / "data" / "state"
PRICES_GZ = STATE_DIR / "prices.csv.gz"


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


def _load_committed_prices(conn) -> None:
    """Load the slim committed Open/Close history into prices_eod."""
    if not PRICES_GZ.exists():
        print(f"  WARNING: {PRICES_GZ.name} missing — no committed price history to load.")
        return
    df = pd.read_csv(PRICES_GZ, compression="gzip")
    df["Source"] = "committed"
    df["Quarantined"] = 0
    n = upsert_prices_eod(conn, df)
    print(f"  loaded {n} committed price rows ({df['Date'].min()} -> {df['Date'].max()})")


def _incremental_price_update(conn, symbols: list[str], incremental_days: int) -> None:
    """Fetch only the recent window and upsert. NON-FATAL: on any failure or empty result,
    warn and keep the committed prices (dashboard shows last-good data, job still succeeds)."""
    try:
        from ingestion.providers.yf_provider import YFinanceProvider
        end = date.today() + timedelta(days=1)
        start = end - timedelta(days=incremental_days)
        print(f"  incremental price fetch {start} -> {date.today()} ({len(symbols)} symbols) ...")
        raw = YFinanceProvider().get_daily(symbols, start=start.isoformat(), end=end.isoformat())
        if raw.empty:
            print("  WARNING: incremental fetch returned no data — using committed prices as-is.")
            return
        upsert_prices_eod(conn, apply_sanity_gate(raw))
        print(f"  incremental fetch added/updated {len(raw)} rows for {raw['Symbol'].nunique()} symbols.")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: incremental price fetch failed ({exc}) — using committed prices as-is.")


def _full_refetch(conn, symbols: list[str], history_days: int = 1000) -> None:
    """Re-pull the full history from Yahoo (run locally where Yahoo isn't blocked)."""
    from ingestion.providers.yf_provider import YFinanceProvider
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=history_days)
    print(f"  FULL refetch {start} -> {date.today()} ...")
    raw = YFinanceProvider().get_daily(symbols, start=start.isoformat(), end=end.isoformat())
    upsert_prices_eod(conn, apply_sanity_gate(raw))


def _export_slim_prices(conn) -> None:
    df = pd.read_sql_query("SELECT Symbol, Date, Open, Close FROM prices_eod WHERE Quarantined=0", conn)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PRICES_GZ, index=False, compression="gzip")
    print(f"  re-exported slim prices: {len(df)} rows -> {PRICES_GZ.name} (through {df['Date'].max()})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inception", default="2026-01-01")
    parser.add_argument("--incremental-days", type=int, default=12)
    parser.add_argument("--full-refetch", action="store_true", help="re-pull all history (run locally)")
    args = parser.parse_args()

    with open(BASE_DIR / "config" / "strategies.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    notional = cfg["portfolio_engine"]["notional_capital"]

    conn = get_connection(BASE_DIR / "data" / "processed" / "portfolio_dashboard.db")
    ensure_schema(conn)

    # 1. committed state — no large external fetch -----------------------------
    n_uni = _load_state_csv(conn, "universe_membership", STATE_DIR / "universe_membership.csv",
                            ["IndexName", "Symbol", "CompanyName", "Industry", "ISIN", "AsOfDate"])
    n_circ = _load_state_csv(conn, "circuit_days", STATE_DIR / "circuit_days.csv", ["Symbol", "Date", "IsCircuit"])
    n_mcap = _load_state_csv(conn, "mcap_class", STATE_DIR / "mcap_class.csv", ["Symbol", "Category", "MarketCap_Cr", "AsOfDate"])
    _load_committed_prices(conn)
    print(f"  state loaded: {n_uni} universe rows, {n_circ} circuit flags, {n_mcap} mcap classes")

    symbols = pd.read_sql_query("SELECT DISTINCT Symbol FROM universe_membership", conn)["Symbol"].tolist()

    # 2. small, graceful price update -----------------------------------------
    if args.full_refetch:
        _full_refetch(conn, symbols)
    else:
        _incremental_price_update(conn, symbols, args.incremental_days)

    closes = load_prices_wide(conn)
    opens = load_field_wide(conn, "Open")
    circuit_flags = load_circuit_flags_wide(conn)
    circuit_flags = circuit_flags.reindex(index=closes.index, columns=closes.columns).fillna(0) if not circuit_flags.empty else pd.DataFrame(0, index=closes.index, columns=closes.columns)
    trading_days = sorted(closes.index)
    inception_idx = max(i for i, d in enumerate(trading_days) if d <= pd.Timestamp(args.inception))
    print(f"Prices to {trading_days[-1].date()}; inception {trading_days[inception_idx].date()}")

    # 3. replay strategies -----------------------------------------------------
    for t in ["strategy_meta", "strategy_state", "strategy_nav", "strategy_rebalances", "strategy_trades"]:
        conn.execute(f"DELETE FROM {t}")
    conn.commit()
    for key in cfg["strategies"]:
        rollforward_one(conn, key, cfg["strategies"][key], notional, trading_days, closes, opens, circuit_flags, inception_idx)

    compute_all_strategy_metrics(conn, cfg["strategies"], BASE_DIR / "data" / "processed", prices_wide=closes)

    # 4. advance the committed price base ONLY on a full local refetch. The daily CI job leaves
    #    prices.csv.gz untouched (it just incremental-fetches the gap from the fixed base each
    #    run) so we don't commit a 7MB file every day and bloat the repo. Refresh the base
    #    periodically by running this locally with --full-refetch.
    if args.full_refetch:
        _export_slim_prices(conn)

    conn.close()

    # 5. fund comparison (independent; non-fatal) ------------------------------
    try:
        from analytics.mf_comparison import build_fund_comparison
        build_fund_comparison(BASE_DIR)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: fund comparison update failed (keeps last good data): {exc}")

    print(f"\nRebuilt dashboard to {trading_days[-1].date()}.")


if __name__ == "__main__":
    main()
