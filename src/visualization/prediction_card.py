import plotly.graph_objects as go


def create_prediction_display(direction: str, confidence: float, accuracy: float) -> dict:
    """Build prediction card data for the dashboard."""
    arrow = "↑" if direction == "bullish" else "↓"
    color = "#22c55e" if direction == "bullish" else "#ef4444"

    return {
        "direction": direction.capitalize(),
        "arrow": arrow,
        "color": color,
        "confidence": f"{confidence:.0%}",
        "accuracy": f"{accuracy:.0%}",
    }


def create_feature_importance_chart(feature_names: list[str], importances: list[float]) -> go.Figure:
    """Horizontal bar chart of XGBoost feature importances."""
    if not feature_names or not importances:
        fig = go.Figure()
        fig.add_annotation(text="No model trained", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark", height=250)
        return fig

    sorted_pairs = sorted(zip(importances, feature_names), reverse=True)
    sorted_imp, sorted_names = zip(*sorted_pairs)

    fig = go.Figure(go.Bar(
        x=list(sorted_imp),
        y=list(sorted_names),
        orientation="h",
        marker_color="#3b82f6",
    ))

    fig.update_layout(
        title="Feature Importance",
        template="plotly_dark",
        height=250,
        margin=dict(l=120, r=20, t=40, b=30),
        yaxis=dict(autorange="reversed"),
    )

    return fig
