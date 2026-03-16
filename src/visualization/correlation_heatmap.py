import plotly.graph_objects as go
import pandas as pd
import numpy as np


def create_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Lag correlation heatmap showing Pearson and Spearman at each lag."""
    if corr_df.empty:
        return _empty_figure("No correlation data")

    lags = corr_df["lag"].tolist()
    lag_labels = [f"T+{l}" for l in lags]

    z = np.array([corr_df["pearson_r"].values, corr_df["spearman_r"].values])

    # Text annotations with r values
    text = [
        [f"r={v:.3f}" for v in corr_df["pearson_r"]],
        [f"r={v:.3f}" for v in corr_df["spearman_r"]],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=lag_labels,
        y=["Pearson", "Spearman"],
        text=text,
        texttemplate="%{text}",
        colorscale="RdYlGn_r",
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlation"),
    ))

    fig.update_layout(
        title=dict(text="Lag Correlation", font=dict(size=13)),
        template="plotly_dark",
        height=220,
        margin=dict(l=70, r=30, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=11),
    )

    return fig


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14))
    fig.update_layout(template="plotly_dark", height=250)
    return fig
