import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def _normalize_to_range(series: pd.Series, target_min: float = 0, target_max: float = 10) -> pd.Series:
    """Normalize a series to a target range, handling NaN."""
    clean = series.dropna()
    if clean.empty or clean.max() == clean.min():
        return series.fillna(5)
    normalized = (series - clean.min()) / (clean.max() - clean.min()) * (target_max - target_min) + target_min
    return normalized


def create_sentiment_vs_sp500_chart(
    merged_df: pd.DataFrame,
    market_label: str = "S&P 500",
) -> go.Figure:
    if merged_df.empty:
        return _empty_figure("No data available")

    if len(merged_df) == 1:
        return _single_day_figure(merged_df)

    # Get fear range for Y-axis
    fear_vals = merged_df["avg_recession_fear"].dropna()
    if fear_vals.empty:
        y_min, y_max = 0, 10
    else:
        y_min = max(0, fear_vals.min() - 0.5)
        y_max = min(10, fear_vals.max() + 0.5)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Sentiment fear line
    fig.add_trace(go.Scatter(
        x=merged_df["date"],
        y=merged_df["avg_recession_fear"],
        name="Recession Fear",
        line=dict(color="#ef4444", width=2),
        mode="lines+markers",
    ), secondary_y=False)

    if "rolling_3d" in merged_df.columns:
        fig.add_trace(go.Scatter(
            x=merged_df["date"],
            y=merged_df["rolling_3d"],
            name="3-day MA",
            line=dict(color="#f97316", width=1, dash="dash"),
            mode="lines",
        ), secondary_y=False)

    if "rolling_7d" in merged_df.columns:
        fig.add_trace(go.Scatter(
            x=merged_df["date"],
            y=merged_df["rolling_7d"],
            name="7-day MA",
            line=dict(color="#eab308", width=1, dash="dot"),
            mode="lines",
        ), secondary_y=False)

    # S&P 500 on secondary axis
    if "sp500_close" in merged_df.columns:
        sp_clean = merged_df["sp500_close"].dropna()
        if not sp_clean.empty:
            fig.add_trace(go.Scatter(
                x=merged_df["date"],
                y=merged_df["sp500_close"],
                name=market_label,
                line=dict(color="#3b82f6", width=2),
                mode="lines",
            ), secondary_y=True)

    title = f"Sentiment vs {market_label}"

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center", font=dict(size=10)),
        margin=dict(l=55, r=55, t=45, b=55),
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=11),
    )

    # Fear axis: inverted, range from data
    fig.update_yaxes(
        title_text="Recession Fear",
        range=[y_max, y_min],  # inverted: high fear at bottom
        secondary_y=False,
    )

    fig.update_yaxes(
        title_text=market_label,
        secondary_y=True,
    )

    return fig


def _single_day_figure(df: pd.DataFrame) -> go.Figure:
    row = df.iloc[0]
    fear = row.get("avg_recession_fear", 0)
    d = row.get("date", "")
    articles = int(row.get("article_count", 0))

    color = "#ef4444" if fear >= 5 else "#f97316" if fear >= 3 else "#22c55e"
    sentiment = "High Fear" if fear >= 5 else "Moderate" if fear >= 3 else "Low Fear"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(d)], y=[fear],
        marker_color=color,
        text=[f"Fear: {fear:.1f}/10<br>{articles} articles<br>{sentiment}"],
        textposition="outside",
    ))

    fig.update_layout(
        title=f"Sector Summary — {d} — Recession Fear: {fear:.1f}/10 ({sentiment})",
        template="plotly_dark",
        height=450,
        yaxis=dict(range=[0, 10], title="Recession Fear (0-10)"),
        showlegend=False,
    )
    return fig


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
    fig.update_layout(template="plotly_dark", height=450)
    return fig
