import plotly.graph_objects as go
import pandas as pd
import numpy as np


def create_sector_heatmap(scored_df: pd.DataFrame) -> go.Figure:
    """Sector × days heatmap color-coded by average recession fear."""
    if scored_df.empty or "sectors" not in scored_df.columns:
        return _empty_figure("No sector data")

    # Explode sectors (each article may have multiple)
    df = scored_df.copy()

    # Handle sectors stored as strings (from cache)
    import ast
    def parse_sectors(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                return parsed if isinstance(parsed, list) else [val]
            except (ValueError, SyntaxError):
                return [s.strip() for s in val.split(",") if s.strip()]
        return []

    df["sectors"] = df["sectors"].apply(parse_sectors)
    df = df.explode("sectors")
    df = df.dropna(subset=["sectors"])
    df = df[df["sectors"] != ""]

    if df.empty:
        return _empty_figure("No sector data")

    # Only keep sectors with at least 2 articles
    sector_counts = df["sectors"].value_counts()
    valid_sectors = sector_counts[sector_counts >= 2].index
    df = df[df["sectors"].isin(valid_sectors)]

    if df.empty:
        return _empty_figure("No sector data")

    pivot = df.pivot_table(
        values="recession_fear",
        index="sectors",
        columns="date",
        aggfunc="mean",
    )

    if pivot.empty:
        return _empty_figure("No sector data")

    # Sort dates
    pivot = pivot[sorted(pivot.columns)]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(d) for d in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="YlOrRd_r",
        zmin=0,
        zmax=10,
        colorbar=dict(title="Fear<br>0=calm<br>10=panic"),
    ))

    fig.update_layout(
        title=dict(text="Sector Sentiment Heatmap", font=dict(size=13)),
        template="plotly_dark",
        height=350,
        margin=dict(l=90, r=30, t=40, b=40),
        xaxis=dict(tickangle=-45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=10),
    )

    return fig


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14))
    fig.update_layout(template="plotly_dark", height=300)
    return fig
