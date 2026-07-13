from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
STRATEGIES_DIR = BASE_DIR / "data" / "processed" / "strategies"

# theme.py lives in momentum_dashboard/ (one level up from pages/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import theme  # noqa: E402

st.set_page_config(page_title="Momentum Portfolio Dashboard — Comparison", layout="wide", initial_sidebar_state="collapsed")
st.markdown(theme.PAGE_CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="dash-header"><h1>Strategy Comparison</h1>'
    '<p>All four momentum strategies since inception — % return, rebased to 0 at each start</p></div>',
    unsafe_allow_html=True,
)

with open(BASE_DIR / "config" / "strategies.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
strategies = cfg["strategies"]

fig = go.Figure()
rows = []
any_live = False
end_points = []  # (y, sid, label) for direct end-labels

for sid, scfg in strategies.items():
    sdir = STRATEGIES_DIR / sid
    metrics_path = sdir / "metrics.json"
    if not metrics_path.exists():
        continue
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    color = theme.STRATEGY_COLORS.get(sid, theme.PRIMARY_HUE)
    if metrics.get("status") != "live":
        rows.append({"Strategy": f"{sid} — {scfg['name']}", "Status": "not live yet",
                     "Return": None, "CAGR": None, "Sharpe": None, "Max DD": None})
        continue
    any_live = True
    nav_path = sdir / "nav_vs_benchmark.csv"
    if nav_path.exists():
        nav = pd.read_csv(nav_path, parse_dates=["Date"])
        if not nav.empty:
            pct = nav["NAV"] - 100.0  # % return since inception (NAV is base-100)
            fig.add_trace(go.Scatter(
                x=nav["Date"], y=pct, name=f"{sid} — {scfg['name']}", mode="lines",
                line=dict(color=color, width=2),
                hovertemplate=f"<b>{sid}</b> %{{x|%d %b %Y}}<br>%{{y:.2f}}%<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=[nav["Date"].iloc[-1]], y=[pct.iloc[-1]], mode="markers",
                marker=dict(color=color, size=9, line=dict(color=theme.SURFACE, width=2)),
                showlegend=False, hoverinfo="skip",
            ))
            end_points.append((pct.iloc[-1], sid, f"{sid}  {pct.iloc[-1]:+.1f}%", color, nav["Date"].iloc[-1]))
    rows.append({
        "Strategy": f"{sid} — {scfg['name']}",
        "Status": f"live since {metrics.get('inception_date')}",
        "Return": metrics.get("period_return"),
        "CAGR": metrics.get("cagr"),
        "Sharpe": metrics.get("sharpe"),
        "Max DD": metrics.get("max_drawdown"),
    })

if any_live:
    theme.styled_layout(fig, height=460, showlegend=True)
    fig.update_yaxes(ticksuffix="%", zeroline=True, zerolinecolor=theme.AXIS, zerolinewidth=1)
    # direct end-labels supplement the legend (4 series, well separated at the right edge)
    for y, sid, label, color, x_end in end_points:
        fig.add_annotation(x=x_end, y=y, text=f"  {label}", showarrow=False,
                           xanchor="left", font=dict(color=color, size=12),
                           xshift=6)
    fig.update_layout(margin=dict(l=8, r=90, t=8, b=8))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No strategy is live yet. Run scripts/golive.py once you're ready to start tracking.")

# ── Metrics table ────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Metrics</div>', unsafe_allow_html=True)
mdf = pd.DataFrame(rows)
if not mdf.empty:
    fmt = mdf.copy()
    for col in ["Return", "CAGR", "Max DD"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].apply(lambda v: f"{v*100:+.2f}%" if pd.notna(v) else "—")
    if "Sharpe" in fmt.columns:
        fmt["Sharpe"] = fmt["Sharpe"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    st.dataframe(fmt, use_container_width=True, hide_index=True)
