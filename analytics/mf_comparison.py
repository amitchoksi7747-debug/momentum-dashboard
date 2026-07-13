"""Build the Fund Comparison dataset: fetch the tracked funds' NAVs from MFAPI, compute a
fair common-window rebase, and write the files the dashboard page reads.

The funds have very different inception dates, so a single chart rebased to each fund's own
start would not be comparable. The builder therefore also computes a COMMON window — starting
at the latest inception among the tracked funds — where every fund is rebased to 0% on the
same day, making the lines directly comparable. Per-fund since-inception figures are kept too.

Outputs (data/processed/mf_comparison/):
  navs.csv          — long: ShortName, Category, Date, NAV (raw)
  rebased.csv       — wide: Date + one % -return column per fund over the common window
  metrics.json      — per-fund: inception, latest NAV, since-inception %, common-window %, CAGR
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from analytics.utils import ensure_dir, utc_now_iso, write_json
from ingestion.mf_navs import fetch_many

TRADING_DAYS_PER_YEAR = 252


def _cagr(nav_start: float, nav_end: float, days: int) -> float | None:
    if nav_start <= 0 or days <= 0:
        return None
    years = days / 365.25
    if years <= 0:
        return None
    return float((nav_end / nav_start) ** (1 / years) - 1)


def _period_return(nav_by_date: pd.Series, latest_nav: float, latest_date, months: int | None = None,
                   ytd: bool = False) -> float | None:
    """Trailing-period return. `nav_by_date` is a NAV Series indexed by Date (sorted ascending).
    Base = the last NAV on/before the target date. Returns None if the fund has no history back
    to the target (too new) — except YTD, where a fund launched during the current year falls
    back to since-inception (its first NAV), which is what 'year to date' means for such a fund.
    """
    if ytd:
        target = pd.Timestamp(year=latest_date.year, month=1, day=1)
    else:
        target = latest_date - pd.DateOffset(months=months)
    prior = nav_by_date[nav_by_date.index <= target]
    if prior.empty:
        if ytd:
            base = float(nav_by_date.iloc[0])  # fund born this year → YTD = since inception
        else:
            return None
    else:
        base = float(prior.iloc[-1])
    if base <= 0:
        return None
    return latest_nav / base - 1.0


def build_fund_comparison(base_dir: Path) -> dict:
    with open(base_dir / "config" / "momentum_funds.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    funds = cfg["funds"]

    out_dir = ensure_dir(base_dir / "data" / "processed" / "mf_comparison")
    print(f"Fetching NAVs for {len(funds)} funds from api.mfapi.in ...")
    navs = fetch_many(funds)

    if navs.empty:
        write_json(out_dir / "metrics.json", {"status": "no_data", "run_timestamp": utc_now_iso()})
        pd.DataFrame(columns=["ShortName", "Category", "Date", "NAV"]).to_csv(out_dir / "navs.csv", index=False)
        pd.DataFrame(columns=["Date"]).to_csv(out_dir / "rebased.csv", index=False)
        print("  no NAV data returned — wrote empty files.")
        return {"status": "no_data"}

    navs.to_csv(out_dir / "navs.csv", index=False)

    # common window = latest inception among the funds (so all lines start together)
    inceptions = navs.groupby("ShortName")["Date"].min()
    common_start = inceptions.max()

    per_fund = {}
    rebased_series = {}
    for name, g in navs.groupby("ShortName"):
        g = g.sort_values("Date")
        inception = g["Date"].min()
        latest_date = g["Date"].max()
        latest_nav = float(g["NAV"].iloc[-1])
        first_nav = float(g["NAV"].iloc[0])

        # common-window rebase: NAV relative to the NAV on/after common_start, as % return
        gw = g[g["Date"] >= common_start]
        if not gw.empty:
            base_nav = float(gw["NAV"].iloc[0])
            rebased_series[name] = (gw.set_index("Date")["NAV"] / base_nav - 1.0) * 100.0
            common_window_return = latest_nav / base_nav - 1.0
        else:
            common_window_return = None

        nav_by_date = g.set_index("Date")["NAV"]
        per_fund[name] = {
            "category": g["Category"].iloc[0],
            "official_name": g["OfficialName"].iloc[0],
            "inception": inception.date().isoformat(),
            "latest_date": latest_date.date().isoformat(),
            "latest_nav": latest_nav,
            "return_1m": _period_return(nav_by_date, latest_nav, latest_date, months=1),
            "return_3m": _period_return(nav_by_date, latest_nav, latest_date, months=3),
            "return_6m": _period_return(nav_by_date, latest_nav, latest_date, months=6),
            "return_ytd": _period_return(nav_by_date, latest_nav, latest_date, ytd=True),
            "since_inception_return": latest_nav / first_nav - 1.0,
            "since_inception_cagr": _cagr(first_nav, latest_nav, (latest_date - inception).days),
            "common_window_return": common_window_return,
        }

    rebased = pd.DataFrame(rebased_series).sort_index()
    rebased.index.name = "Date"
    rebased.to_csv(out_dir / "rebased.csv")

    metrics = {
        "status": "ok",
        "run_timestamp": utc_now_iso(),
        "common_start": common_start.date().isoformat(),
        "as_of_date": navs["Date"].max().date().isoformat(),
        "funds": per_fund,
    }
    write_json(out_dir / "metrics.json", metrics)
    print(f"  built comparison for {len(per_fund)} funds; common window from {common_start.date()} "
          f"to {navs['Date'].max().date()}.")
    return metrics
