import ast
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


SECTOR_COLORS = {
    # 11 GICS sectors
    "Technology": "#3b82f6",
    "Healthcare": "#a855f7",
    "Financials": "#22c55e",
    "Energy": "#f97316",
    "Consumer Discretionary": "#ec4899",
    "Consumer Staples": "#f472b6",
    "Industrials": "#eab308",
    "Materials": "#fb923c",
    "Utilities": "#facc15",
    "Real Estate": "#84cc16",
    "Communication Services": "#06b6d4",
    # Legacy names from old cache
    "Finance": "#22c55e",
    "Industrial": "#eab308",
    "Consumer": "#ec4899",
    "Government": "#64748b",
    "Automotive": "#f43f5e",
    "Media": "#8b5cf6",
    "Retail": "#14b8a6",
    "Transportation": "#d946ef",
    "Aerospace": "#0ea5e9",
    "Other": "#6b7280",
}

_FALLBACK_COLORS = [
    "#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#ffeaa7",
    "#dfe6e9", "#fd79a8", "#6c5ce7", "#00b894", "#e17055",
]


def create_sector_timeseries(scored_df: pd.DataFrame, sector_daily: dict = None) -> go.Figure:
    if scored_df.empty or "sectors" not in scored_df.columns:
        return _empty_figure("No sector data")

    df = scored_df.copy()
    df["sectors"] = df["sectors"].apply(_parse_sectors)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.explode("sectors").dropna(subset=["sectors"])
    df = df[df["sectors"] != ""]

    if df.empty:
        return _empty_figure("No sector data")

    # Only keep sectors with at least 2 articles
    sector_counts = df["sectors"].value_counts()
    valid_sectors = sector_counts[sector_counts >= 2].index
    df = df[df["sectors"].isin(valid_sectors)]

    if df.empty:
        return _empty_figure("No sector data")

    pivot = df.pivot_table(values="recession_fear", index="date", columns="sectors", aggfunc="mean")
    pivot = pivot.sort_index()

    fig = go.Figure()
    for sector in pivot.columns:
        color = SECTOR_COLORS.get(sector)
        if color is None:
            color = _FALLBACK_COLORS[len([s for s in pivot.columns[:list(pivot.columns).index(sector)] if s not in SECTOR_COLORS]) % len(_FALLBACK_COLORS)]
        fig.add_trace(go.Scatter(
            x=pivot.index, y=pivot[sector],
            name=sector, line=dict(color=color, width=2),
            mode="lines+markers",
        ))

    fig.update_layout(
        title=dict(text="Sector Sentiment Over Time", font=dict(size=13)),
        template="plotly_dark",
        height=420,
        yaxis_title="Recession Fear",
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center", font=dict(size=9)),
        margin=dict(l=50, r=30, t=40, b=90),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=10),
    )
    return fig


def _parse_sectors(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            return parsed if isinstance(parsed, list) else [val]
        except (ValueError, SyntaxError):
            return [s.strip() for s in val.split(",") if s.strip()]
    return []


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14))
    fig.update_layout(template="plotly_dark", height=350)
    return fig
