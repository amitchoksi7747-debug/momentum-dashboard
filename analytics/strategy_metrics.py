"""Per-strategy live-tracking outputs, written to data/processed/strategies/{sid}/ — the same
'pipeline writes files, Streamlit reads files' contract as the rest of this repo.

Every function here is tolerant of a strategy that hasn't gone live yet (no strategy_meta /
strategy_nav rows): it writes empty-but-present files with status='not_live' rather than
crashing, so the dashboard can render a clean "not live yet" state instead of erroring.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.risk import calmar, drawdown_series, monthly_return_table, sortino, alpha_beta
from analytics.utils import ensure_dir, utc_now_iso, write_json
from strategies.portfolio_engine import get_strategy_meta, rebalance_sheet_from_dict

TRADING_DAYS_PER_YEAR = 252
MIN_DAYS_FOR_ANNUALIZATION = 60


def _load_nav(conn: sqlite3.Connection, strategy: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT Date, NAV, Cash, NHoldings FROM strategy_nav WHERE Strategy = ? ORDER BY Date",
        conn, params=(strategy,),
    )
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
    return df


def _compute_core_metrics(nav_series: pd.Series, inception_date: str) -> dict:
    n = len(nav_series)
    period_return = float(nav_series.iloc[-1] / 100.0 - 1)
    dd = drawdown_series(nav_series)
    metrics = {
        "status": "live",
        "inception_date": inception_date,
        "as_of_date": nav_series.index.max().date().isoformat(),
        "days_live": n,
        "nav_latest": float(nav_series.iloc[-1]),
        "period_return": period_return,
        "max_drawdown": float(dd.min()),
        "current_drawdown": float(dd.iloc[-1]),
        "cagr": None,
        "annualized_vol": None,
        "sharpe": None,
        "sortino": None,
        "calmar": None,
        "annualization_note": None,
    }
    if n < MIN_DAYS_FOR_ANNUALIZATION:
        metrics["annualization_note"] = (
            f"Only {n} live trading days so far — annualized figures (CAGR/vol/Sharpe) are "
            f"suppressed until at least {MIN_DAYS_FOR_ANNUALIZATION} days of history exist; "
            f"use period_return instead."
        )
        return metrics

    rets = nav_series.pct_change().dropna()
    years = n / TRADING_DAYS_PER_YEAR
    metrics["cagr"] = float((nav_series.iloc[-1] / nav_series.iloc[0]) ** (1 / years) - 1)
    ann_vol = float(rets.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    metrics["annualized_vol"] = ann_vol
    ann_ret = float(rets.mean() * TRADING_DAYS_PER_YEAR)
    metrics["sharpe"] = ann_ret / ann_vol if ann_vol else None
    metrics["sortino"] = sortino(nav_series)
    metrics["calmar"] = calmar(nav_series)
    return metrics


def compute_strategy_metrics(
    conn: sqlite3.Connection,
    strategy_key: str,
    strategy_cfg: dict,
    processed_dir: Path,
    prices_wide: pd.DataFrame | None = None,
) -> dict:
    out_dir = ensure_dir(Path(processed_dir) / "strategies" / strategy_key)
    meta = get_strategy_meta(conn, strategy_key)
    nav_df = _load_nav(conn, strategy_key)

    if meta is None or meta.get("InceptionDate") is None or nav_df.empty:
        metrics = {"status": "not_live", "strategy_name": strategy_cfg.get("name"), "run_timestamp": utc_now_iso()}
        write_json(out_dir / "metrics.json", metrics)
        pd.DataFrame(columns=["Date", "NAV", "Benchmark"]).to_csv(out_dir / "nav_vs_benchmark.csv", index=False)
        pd.DataFrame(columns=["Date", "Drawdown"]).to_csv(out_dir / "drawdown.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "monthly_returns.csv")
        pd.DataFrame(columns=["Symbol", "Units", "EntryDate", "EntryRank", "CurrentPrice", "MarketValue", "Weight", "DaysHeld", "Sector", "MCapCategory"]).to_csv(out_dir / "current_holdings.csv", index=False)
        pd.DataFrame(columns=["SectorName", "Weight"]).to_csv(out_dir / "sector_breakdown.csv", index=False)
        pd.DataFrame(columns=["Category", "Weight"]).to_csv(out_dir / "mcap_breakdown.csv", index=False)
        pd.DataFrame(columns=["Date", "Symbol", "Side", "Units", "Price", "Reason"]).to_csv(out_dir / "trades.csv", index=False)
        pd.DataFrame(columns=["Action", "Symbol", "AvgRank", "Reason"]).to_csv(out_dir / "latest_rebalance_sheet.csv", index=False)
        return metrics

    nav_series = nav_df.set_index("Date")["NAV"].dropna()
    metrics = _compute_core_metrics(nav_series, meta["InceptionDate"])
    metrics["strategy_name"] = strategy_cfg.get("name")
    metrics["benchmark_status"] = strategy_cfg.get("benchmark_status")
    metrics["current_cash"] = float(meta["CurrentCash"])
    metrics["notional_capital"] = float(meta["NotionalCapital"])
    metrics["run_timestamp"] = utc_now_iso()

    # --- NAV vs benchmark ---
    bench_col = pd.Series(dtype=float, name="Benchmark")
    if strategy_cfg.get("benchmark_symbol") and prices_wide is not None and strategy_cfg["benchmark_symbol"] in prices_wide.columns:
        bench_raw = prices_wide[strategy_cfg["benchmark_symbol"]].reindex(nav_series.index).ffill()
        if bench_raw.notna().any():
            first_valid = bench_raw.first_valid_index()
            bench_col = (bench_raw / bench_raw.loc[first_valid] * 100.0).rename("Benchmark")
            ab = alpha_beta(nav_series, bench_col)
            metrics["alpha"] = ab["alpha"]
            metrics["beta"] = ab["beta"]
    nav_vs_bench = pd.concat([nav_series.rename("NAV"), bench_col], axis=1).reset_index().rename(columns={"index": "Date"})
    nav_vs_bench.to_csv(out_dir / "nav_vs_benchmark.csv", index=False)

    drawdown_series(nav_series).reset_index().rename(columns={"index": "Date", "NAV": "Drawdown", 0: "Drawdown"}).set_axis(["Date", "Drawdown"], axis=1).to_csv(out_dir / "drawdown.csv", index=False)

    monthly_return_table(nav_series).to_csv(out_dir / "monthly_returns.csv")

    # --- Current holdings + sector/mcap breakdown ---
    holdings = pd.read_sql_query(
        "SELECT Symbol, Units, EntryDate, EntryRank FROM strategy_state WHERE Strategy = ? AND Units > 0",
        conn, params=(strategy_key,),
    )
    if not holdings.empty and prices_wide is not None:
        as_of = nav_series.index.max()
        last_prices = prices_wide.reindex(columns=holdings["Symbol"]).loc[:as_of].ffill().iloc[-1]
        holdings["CurrentPrice"] = holdings["Symbol"].map(last_prices)
        holdings["MarketValue"] = holdings["Units"] * holdings["CurrentPrice"]
        total_value = holdings["MarketValue"].sum() + metrics["current_cash"]
        holdings["Weight"] = holdings["MarketValue"] / total_value if total_value else np.nan
        holdings["DaysHeld"] = (as_of - pd.to_datetime(holdings["EntryDate"])).dt.days

        sector_map = pd.read_sql_query("SELECT DISTINCT Symbol, Industry FROM universe_membership", conn)
        holdings = holdings.merge(sector_map, on="Symbol", how="left").rename(columns={"Industry": "Sector"})

        mcap = pd.read_sql_query(
            "SELECT Symbol, Category FROM mcap_class WHERE AsOfDate = (SELECT MAX(AsOfDate) FROM mcap_class)", conn
        )
        holdings = holdings.merge(mcap, on="Symbol", how="left").rename(columns={"Category": "MCapCategory"})

    holdings.to_csv(out_dir / "current_holdings.csv", index=False)

    if not holdings.empty and "Sector" in holdings.columns:
        sector_bd = holdings.groupby(holdings["Sector"].fillna("Unknown"))["Weight"].sum().reset_index().rename(columns={"Sector": "SectorName"}).sort_values("Weight", ascending=False)
    else:
        sector_bd = pd.DataFrame(columns=["SectorName", "Weight"])
    sector_bd.to_csv(out_dir / "sector_breakdown.csv", index=False)

    if not holdings.empty and "MCapCategory" in holdings.columns:
        mcap_bd = holdings.groupby(holdings["MCapCategory"].fillna("Unclassified"))["Weight"].sum().reset_index().rename(columns={"MCapCategory": "Category"}).sort_values("Weight", ascending=False)
        cash_weight = metrics["current_cash"] / (holdings["MarketValue"].sum() + metrics["current_cash"]) if (holdings["MarketValue"].sum() + metrics["current_cash"]) else 0
        if cash_weight > 1e-6:
            mcap_bd = pd.concat([mcap_bd, pd.DataFrame([{"Category": "Cash", "Weight": cash_weight}])], ignore_index=True)
    else:
        mcap_bd = pd.DataFrame(columns=["Category", "Weight"])
    mcap_bd.to_csv(out_dir / "mcap_breakdown.csv", index=False)

    # --- Trades & latest rebalance sheet ---
    trades = pd.read_sql_query(
        "SELECT Date, Symbol, Side, Units, Price, Reason FROM strategy_trades WHERE Strategy = ? ORDER BY Date DESC",
        conn, params=(strategy_key,),
    )
    trades.to_csv(out_dir / "trades.csv", index=False)

    latest_rebal = pd.read_sql_query(
        "SELECT SignalDate, ExecDate, SheetJson FROM strategy_rebalances WHERE Strategy = ? ORDER BY SignalDate DESC LIMIT 1",
        conn, params=(strategy_key,),
    )
    if not latest_rebal.empty:
        import json
        sheet_dict = json.loads(latest_rebal.iloc[0]["SheetJson"])
        sheet_df = rebalance_sheet_from_dict(sheet_dict)
        metrics["latest_rebalance_signal_date"] = latest_rebal.iloc[0]["SignalDate"]
        metrics["latest_rebalance_exec_date"] = latest_rebal.iloc[0]["ExecDate"]
    else:
        sheet_df = pd.DataFrame(columns=["Action", "Symbol", "AvgRank", "Reason"])
    sheet_df.to_csv(out_dir / "latest_rebalance_sheet.csv", index=False)

    write_json(out_dir / "metrics.json", metrics)
    return metrics


def compute_all_strategy_metrics(conn: sqlite3.Connection, strategies_cfg: dict, processed_dir: Path, prices_wide: pd.DataFrame | None = None) -> dict:
    return {
        key: compute_strategy_metrics(conn, key, cfg, processed_dir, prices_wide=prices_wide)
        for key, cfg in strategies_cfg.items()
    }
