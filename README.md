# Momentum Portfolio Dashboard

A live Streamlit dashboard tracking four rules-based momentum equity strategies and a
comparison of momentum mutual funds. Prices from Yahoo Finance, index constituents from
niftyindices.com, fund NAVs from api.mfapi.in — all free and keyless.

**Live app:** _(add your Streamlit Cloud URL here after deploying)_

## What it shows

- **Strategies (S1–S4):** long-only, monthly-rebalanced, top-30 by average-of-4-window
  Sharpe momentum, with a 200-DEMA trend filter and a circuit-day quality filter, across four
  universes (Nifty LargeMidcap 250, Nifty 500, Nifty MidSmallcap 400, Nifty Total Market).
  Equity curve, drawdown, metrics, sector and market-cap (NSE rank-based) breakdowns, holdings,
  and the latest rebalance sheet.
- **Strategy Comparison:** all four rebased to a common start.
- **Fund Comparison:** six momentum mutual funds (active + index) rebased to a common window.

All portfolio values are **notional** (a cosmetic base for computing % returns) — this repo
contains no personal holdings or account data.

## How it runs

- The Streamlit app (`momentum_dashboard/app.py`) only **reads** the committed files under
  `data/processed/` — no live API calls at page load.
- `scripts/cloud_rebuild.py` regenerates those files from fresh prices + the small committed
  state in `data/state/` (circuit flags, market-cap classes). The price database is rebuilt
  each run and never committed (too large for git).
- A daily GitHub Actions job (`.github/workflows/daily_update.yml`) runs that rebuild every
  weekday evening and commits the fresh numbers, so the app stays current automatically.

## Deploy

1. Push this repo to GitHub (public, so the Streamlit app is publicly viewable).
2. On [share.streamlit.io](https://share.streamlit.io), create an app from this repo with
   **main file path** `momentum_dashboard/app.py`.
3. In the repo: Settings → Actions → General → Workflow permissions → **Read and write**, so the
   daily job can commit updates.

## Run locally

```bash
pip install -r requirements.txt
python scripts/cloud_rebuild.py         # build data
streamlit run momentum_dashboard/app.py # serve
```

Methodology and known limitations are documented in the code comments (see
`strategies/portfolio_engine.py` and `analytics/mf_comparison.py`). Past performance is not
indicative of future results; this dashboard is for information, not investment advice.
