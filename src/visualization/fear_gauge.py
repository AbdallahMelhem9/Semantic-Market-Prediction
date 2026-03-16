import plotly.graph_objects as go


def create_fear_gauge(avg_fear: float) -> go.Figure:
    """Semicircular gauge showing overall market fear level."""

    if avg_fear <= 2:
        label = "Extreme Greed"
    elif avg_fear <= 4:
        label = "Greed"
    elif avg_fear <= 6:
        label = "Neutral"
    elif avg_fear <= 8:
        label = "Fear"
    else:
        label = "Extreme Fear"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_fear,
        title={"text": f"Market Fear Index<br><span style='font-size:0.7em;color:#94a3b8'>{label}</span>"},
        number={"suffix": "/10", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 10], "tickwidth": 1},
            "bar": {"color": "#334155"},
            "steps": [
                {"range": [0, 2], "color": "#22c55e"},
                {"range": [2, 4], "color": "#86efac"},
                {"range": [4, 6], "color": "#fbbf24"},
                {"range": [6, 8], "color": "#f97316"},
                {"range": [8, 10], "color": "#ef4444"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "thickness": 0.8,
                "value": avg_fear,
            },
        },
    ))

    fig.update_layout(
        template="plotly_dark",
        height=220,
        margin=dict(l=20, r=20, t=70, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"),
    )
    return fig
