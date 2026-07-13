"""Shared visual theme for the Momentum Portfolio Dashboard.

Colors are NOT eyeballed — the categorical data palette below was run through the
dataviz skill's palette validator against this dashboard's actual dark surface
(#0f1117): lightness band PASS, chroma floor PASS, CVD separation PASS (worst
adjacent ΔE 15.7 for the 4 strategy slots), contrast-vs-surface PASS. Every chart
pairs color with a secondary encoding (direct labels or a legend), so identity is
never carried by hue alone.

One place defines the palette + Plotly layout so light/dark tokens and mark specs
stay consistent across app.py and the comparison page.
"""
from __future__ import annotations

import plotly.graph_objects as go

# ── Surfaces & ink tokens (dark) ─────────────────────────────────────────────
SURFACE = "#0f1117"        # chart surface (matches .stApp background)
CARD = "#1a1f2e"
INK_PRIMARY = "#e8eaf0"
INK_SECONDARY = "#9aa3b8"
INK_MUTED = "#6b7488"       # axis ticks / de-emphasised labels
GRID = "#20263a"            # hairline gridline, one step off surface, recessive
AXIS = "#333c52"            # baseline / zero-line

# ── Categorical data palette (validated, fixed order) ────────────────────────
# Fixed slot order; never cycled, never repainted on filter. All subsets used
# (4 strategies, 5 funds) were run through the palette validator on this surface.
SERIES = ["#3987e5", "#199e70", "#c98500", "#9085e9", "#e66767", "#d55181"]  # blue, aqua, yellow, violet, red, magenta
STRATEGY_COLORS = {"S1": SERIES[0], "S2": SERIES[1], "S3": SERIES[2], "S4": SERIES[3]}

PRIMARY_HUE = "#3987e5"     # single hue for magnitude-by-category bars (sector, mcap)
BENCHMARK = "#8b93a7"       # benchmark line — muted, deliberately NOT a series color

# ── Status colors (reserved; used with a +/- sign as secondary encoding) ─────
GOOD = "#22b563"           # positive return text — legible on the dark surface
BAD = "#e06666"            # negative return text
NEUTRAL = "#60a5fa"
DRAWDOWN_LINE = "#e06666"
DRAWDOWN_FILL = "rgba(224, 102, 102, 0.12)"

PLOTLY_FONT = dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif",
                   color=INK_SECONDARY, size=12)


def styled_layout(fig: go.Figure, height: int = 360, showlegend: bool = False,
                  y_ticksuffix: str = "") -> go.Figure:
    """Apply the shared quiet-chart layout: recessive hairline grid, muted axis ink,
    transparent surface matching the page, tight margins."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=16, t=8, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=PLOTLY_FONT,
        showlegend=bool(showlegend),  # coerce numpy.bool_ → native bool (plotly rejects numpy)
        legend=dict(orientation="h", y=1.12, x=0, font=dict(color=INK_SECONDARY),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=CARD, font_size=12, font_family=PLOTLY_FONT["family"]),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color=INK_MUTED,
                     linecolor=AXIS, tickfont=dict(color=INK_MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
                     color=INK_MUTED, tickfont=dict(color=INK_MUTED))
    return fig


def hbar(categories, values, *, height: int = 300, color: str = PRIMARY_HUE,
         value_fmt: str = "{:.1f}%", value_scale: float = 100.0) -> go.Figure:
    """Sorted horizontal bar for magnitude-across-categories (replaces a pie).

    Single hue (magnitude is one metric, not multiple identities), sorted so the
    largest sits at the top, each bar direct-labeled at its tip — no legend, no
    per-slice color lottery. `values` are fractions (0-1); scaled by value_scale
    for display.
    """
    pairs = sorted(zip(categories, values), key=lambda p: p[1])  # ascending → largest on top
    cats = [p[0] for p in pairs]
    vals = [p[1] * value_scale for p in pairs]
    labels = [value_fmt.format(v) for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=cats, orientation="h",
        marker=dict(color=color, line=dict(width=0)),
        text=labels, textposition="outside",
        textfont=dict(color=INK_SECONDARY, size=12),
        cliponaxis=False,
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        width=0.62,  # leave air in the band; keeps bars thin
    ))
    fig = styled_layout(fig, height=height)
    # magnitude bars: hide the numeric x-axis (values are on the tips) and grid
    fig.update_xaxes(showticklabels=False, range=[0, max(vals) * 1.18 if vals else 1])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK_SECONDARY, size=12))
    fig.update_layout(margin=dict(l=8, r=48, t=8, b=8))
    return fig


# ── Shared page CSS (dark, blue-slate identity kept; tightened rhythm) ────────
PAGE_CSS = f"""
<style>
.stApp {{ background-color: {SURFACE}; color: {INK_PRIMARY}; }}
.dash-header {{
    background: linear-gradient(90deg, #1a1f2e 0%, #16213e 100%);
    border-bottom: 2px solid {PRIMARY_HUE}; padding: 18px 28px 14px 28px;
    border-radius: 10px; margin-bottom: 18px;
}}
.dash-header h1 {{ color: {INK_PRIMARY}; font-size: 1.7rem; font-weight: 700; margin: 0;
    letter-spacing: -0.01em; }}
.dash-header p  {{ color: {INK_SECONDARY}; font-size: 0.82rem; margin: 5px 0 0 0; }}
.kpi-card {{ background: {CARD}; border: 1px solid #262c3d; border-radius: 12px;
    padding: 15px 16px; text-align: left; }}
.kpi-label {{ color: {INK_MUTED}; font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.08em; margin-bottom: 7px; font-weight: 600; }}
.kpi-value {{ color: {INK_PRIMARY}; font-size: 1.5rem; font-weight: 700;
    font-variant-numeric: tabular-nums; line-height: 1.1; }}
.kpi-pos {{ color: {GOOD}; }} .kpi-neg {{ color: {BAD}; }} .kpi-neu {{ color: {NEUTRAL}; }}
.section-title {{ color: {INK_PRIMARY}; font-size: 1.0rem; font-weight: 600;
    border-left: 3px solid {PRIMARY_HUE}; padding-left: 10px; margin: 24px 0 8px 0; }}
.stale-banner {{ background: #1a2030; border: 1px solid #2b3345; color: {INK_SECONDARY};
    padding: 7px 14px; border-radius: 8px; font-size: 0.8rem; margin-bottom: 14px; }}
</style>
"""
