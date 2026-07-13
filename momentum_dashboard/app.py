from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yaml

import theme

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED = BASE_DIR / "data" / "processed"
STRATEGIES_DIR = PROCESSED / "strategies"

st.set_page_config(page_title="Momentum Portfolio Dashboard", layout="wide", initial_sidebar_state="collapsed")

st.markdown(theme.PAGE_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=60)
def load_strategies_cfg():
    with open(BASE_DIR / "config" / "strategies.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_json_safe(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_pct(x, decimals=2):
    return f"{x*100:.{decimals}f}%" if x is not None and pd.notna(x) else "N/A"


def kpi_card(label, value, cls="kpi-neu"):
    st.markdown(
        f"""<div class="kpi-card"><div class="kpi-label">{label}</div>
        <div class="kpi-value {cls}">{value}</div></div>""",
        unsafe_allow_html=True,
    )


cfg = load_strategies_cfg()
strategies = cfg["strategies"]

st.markdown(
    """<div class="dash-header"><h1>Momentum Portfolio Dashboard</h1>
    <p>4 long-only, monthly-rebalanced momentum model portfolios (top-30, avg-of-4-rank momentum, 200DEMA + circuit filters)</p></div>""",
    unsafe_allow_html=True,
)

strategy_keys = list(strategies.keys())
labels = [f"{k} — {strategies[k]['name']}" for k in strategy_keys]
selected_label = st.selectbox("Strategy", labels, index=0)
sid = strategy_keys[labels.index(selected_label)]
scfg = strategies[sid]

sdir = STRATEGIES_DIR / sid
metrics = read_json_safe(sdir / "metrics.json")

if not metrics or metrics.get("status") == "not_live":
    st.info(
        f"**{scfg['name']}** is not live yet. Once go-live is run (scripts/golive.py) with a "
        f"chosen inception date, this page will show the live equity curve, drawdown, holdings "
        f"and sector/market-cap breakdown for this strategy.",
        icon="ℹ️",
    )
    st.stop()

run_ts = metrics.get("run_timestamp")
st.markdown(
    f"""<div class="stale-banner">LIVE since {metrics.get('inception_date')} · as of {metrics.get('as_of_date')} · last computed {run_ts}</div>""",
    unsafe_allow_html=True,
)

# ── KPI row ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    kpi_card("NAV (base 100)", f"{metrics['nav_latest']:.2f}")
with c2:
    ret = metrics.get("period_return")
    kpi_card("Since-inception return", fmt_pct(ret), "kpi-pos" if (ret or 0) >= 0 else "kpi-neg")
with c3:
    cagr = metrics.get("cagr")
    kpi_card("CAGR (annualized)", fmt_pct(cagr) if cagr is not None else "N/A (< 60d live)")
with c4:
    sharpe = metrics.get("sharpe")
    kpi_card("Sharpe", f"{sharpe:.2f}" if sharpe is not None else "N/A")
with c5:
    mdd = metrics.get("max_drawdown")
    kpi_card("Max drawdown", fmt_pct(mdd), "kpi-neg")

if metrics.get("annualization_note"):
    st.caption(metrics["annualization_note"])

# ── Equity curve + Drawdown ──────────────────────────────────────────────────
CHART_START_DATE = pd.Timestamp("2026-01-01")  # matches the main portfolio page's convention

st.markdown('<div class="section-title">Equity Curve vs Benchmark (% Return)</div>', unsafe_allow_html=True)
nav_bench = read_csv_safe(sdir / "nav_vs_benchmark.csv", parse_dates=["Date"])
if not nav_bench.empty:
    # NAV is already base-100 at inception, so NAV-100 is simply the % return since inception.
    nav_pct = nav_bench["NAV"] - 100.0
    has_bench = "Benchmark" in nav_bench.columns and nav_bench["Benchmark"].notna().any()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nav_bench["Date"], y=nav_pct, name=scfg["name"], mode="lines",
        line=dict(color=theme.STRATEGY_COLORS.get(sid, theme.PRIMARY_HUE), width=2, shape="linear"),
        hovertemplate="%{x|%d %b %Y}<br>%{y:.2f}%<extra></extra>",
    ))
    # end-dot marker so the latest value reads at a glance
    fig.add_trace(go.Scatter(
        x=[nav_bench["Date"].iloc[-1]], y=[nav_pct.iloc[-1]], mode="markers",
        marker=dict(color=theme.STRATEGY_COLORS.get(sid, theme.PRIMARY_HUE), size=9,
                    line=dict(color=theme.SURFACE, width=2)),
        showlegend=False, hoverinfo="skip",
    ))
    if has_bench:
        bench_pct = nav_bench["Benchmark"] - 100.0
        fig.add_trace(go.Scatter(
            x=nav_bench["Date"], y=bench_pct, name="Benchmark", mode="lines",
            line=dict(color=theme.BENCHMARK, width=1.5, dash="dot"),
            hovertemplate="%{x|%d %b %Y}<br>%{y:.2f}%<extra></extra>",
        ))
    else:
        st.caption(f"Benchmark unavailable for this strategy ({scfg.get('benchmark_status', 'unknown')}) — see docs/momentum_strategies_plan.md.")
    theme.styled_layout(fig, height=360, showlegend=has_bench)
    fig.update_yaxes(ticksuffix="%", zeroline=True, zerolinecolor=theme.AXIS, zerolinewidth=1)
    fig.update_xaxes(range=[CHART_START_DATE, max(nav_bench["Date"].max(), CHART_START_DATE)])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.caption("No NAV history yet.")

st.markdown('<div class="section-title">Drawdown</div>', unsafe_allow_html=True)
dd = read_csv_safe(sdir / "drawdown.csv", parse_dates=["Date"])
if not dd.empty:
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd["Date"], y=dd["Drawdown"] * 100, mode="lines", fill="tozeroy",
        line=dict(color=theme.DRAWDOWN_LINE, width=1.5), fillcolor=theme.DRAWDOWN_FILL,
        hovertemplate="%{x|%d %b %Y}<br>%{y:.2f}%<extra></extra>", name="Drawdown",
    ))
    theme.styled_layout(fig_dd, height=200)
    fig_dd.update_yaxes(ticksuffix="%")
    fig_dd.update_xaxes(range=[CHART_START_DATE, max(dd["Date"].max(), CHART_START_DATE)])
    st.plotly_chart(fig_dd, use_container_width=True)

# ── Sector & Market-cap breakdown (sorted bars — precise, legend-free) ────────
col_sector, col_mcap = st.columns(2)
with col_sector:
    st.markdown('<div class="section-title">Sector Breakdown</div>', unsafe_allow_html=True)
    sec = read_csv_safe(sdir / "sector_breakdown.csv")
    if not sec.empty:
        fig_s = theme.hbar(sec["SectorName"].tolist(), sec["Weight"].tolist(),
                           height=max(300, 26 * len(sec) + 60))
        st.plotly_chart(fig_s, use_container_width=True)
    else:
        st.caption("No holdings yet.")
with col_mcap:
    st.markdown('<div class="section-title">Market Cap Breakdown (NSE rank-based)</div>', unsafe_allow_html=True)
    mcap = read_csv_safe(sdir / "mcap_breakdown.csv")
    if not mcap.empty:
        fig_m = theme.hbar(mcap["Category"].tolist(), mcap["Weight"].tolist(), height=300)
        st.plotly_chart(fig_m, use_container_width=True)
    else:
        st.caption("No mcap classification available yet — run scripts/refresh_mcap.py.")

# ── Current holdings ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Current Holdings</div>', unsafe_allow_html=True)
holdings = read_csv_safe(sdir / "current_holdings.csv")
if not holdings.empty:
    show_cols = [c for c in ["Symbol", "Sector", "MCapCategory", "Units", "CurrentPrice", "MarketValue", "Weight", "EntryDate", "DaysHeld", "EntryRank"] if c in holdings.columns]
    st.dataframe(holdings[show_cols].sort_values("Weight", ascending=False), use_container_width=True, hide_index=True)
else:
    st.caption("No current holdings.")

# ── Latest rebalance sheet ───────────────────────────────────────────────────
st.markdown('<div class="section-title">Latest Rebalance Sheet</div>', unsafe_allow_html=True)
if metrics.get("latest_rebalance_signal_date"):
    st.caption(f"Signal date: {metrics['latest_rebalance_signal_date']} · Executed: {metrics.get('latest_rebalance_exec_date') or 'pending'}")
sheet = read_csv_safe(sdir / "latest_rebalance_sheet.csv")
if not sheet.empty:
    st.dataframe(sheet, use_container_width=True, hide_index=True)
else:
    st.caption("No rebalance recorded yet.")

# ── Monthly returns heatmap ──────────────────────────────────────────────────
with st.expander("Monthly Returns Heatmap"):
    monthly = read_csv_safe(sdir / "monthly_returns.csv", index_col=0)
    if not monthly.empty:
        # Diverging scale: red (loss) → neutral gray at zero → green (gain), centered at 0.
        diverging = [[0.0, "#c0392b"], [0.5, "#2b3140"], [1.0, theme.GOOD]]
        fig_h = px.imshow(monthly * 100, text_auto=".1f", aspect="auto",
                          color_continuous_scale=diverging, zmin=-(monthly.abs().max().max()*100),
                          zmax=(monthly.abs().max().max()*100))
        fig_h.update_traces(textfont=dict(size=11), hovertemplate="%{y} · month %{x}<br>%{z:.2f}%<extra></extra>")
        theme.styled_layout(fig_h, height=300)
        fig_h.update_xaxes(showgrid=False, side="top")
        fig_h.update_yaxes(showgrid=False)
        fig_h.update_coloraxes(showscale=False)
        st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.caption("Not enough history yet for a monthly breakdown.")

# ── Trade log ─────────────────────────────────────────────────────────────────
with st.expander("Trade Log"):
    trades = read_csv_safe(sdir / "trades.csv")
    if not trades.empty:
        st.dataframe(trades, use_container_width=True, hide_index=True)
    else:
        st.caption("No trades recorded yet.")
