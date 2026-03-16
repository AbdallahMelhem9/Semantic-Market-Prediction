import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date


def _make_merged_df(n=15):
    dates = [date(2026, 3, d + 1) for d in range(n)]
    return pd.DataFrame({
        "date": dates,
        "avg_recession_fear": np.random.uniform(2, 8, n),
        "rolling_3d": np.random.uniform(3, 7, n),
        "rolling_7d": np.random.uniform(3, 7, n),
        "sp500_close": np.random.uniform(4400, 4600, n),
        "sp500_return": np.random.uniform(-0.02, 0.02, n),
    })


def _make_scored_df():
    return pd.DataFrame({
        "date": [date(2026, 3, 1)] * 4,
        "recession_fear": [7.0, 3.0, 5.0, 8.0],
        "market_sentiment": ["bearish", "bullish", "neutral", "bearish"],
        "sectors": [["Technology"], ["Finance"], ["Energy", "Technology"], ["Healthcare"]],
    })


def _make_corr_df():
    return pd.DataFrame({
        "lag": [0, 1, 3, 5],
        "pearson_r": [0.15, 0.32, -0.10, 0.05],
        "pearson_p": [0.3, 0.05, 0.6, 0.8],
        "spearman_r": [0.12, 0.28, -0.08, 0.03],
        "spearman_p": [0.35, 0.08, 0.65, 0.85],
        "n_samples": [20, 19, 17, 15],
    })


# --- Sentiment chart ---

def test_sentiment_chart_returns_figure():
    from src.visualization.sentiment_chart import create_sentiment_vs_sp500_chart
    fig = create_sentiment_vs_sp500_chart(_make_merged_df())
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2  # at least sentiment + S&P lines


def test_sentiment_chart_empty():
    from src.visualization.sentiment_chart import create_sentiment_vs_sp500_chart
    fig = create_sentiment_vs_sp500_chart(pd.DataFrame())
    assert isinstance(fig, go.Figure)


# --- Correlation heatmap ---

def test_correlation_heatmap_returns_figure():
    from src.visualization.correlation_heatmap import create_correlation_heatmap
    fig = create_correlation_heatmap(_make_corr_df())
    assert isinstance(fig, go.Figure)


def test_correlation_heatmap_empty():
    from src.visualization.correlation_heatmap import create_correlation_heatmap
    fig = create_correlation_heatmap(pd.DataFrame())
    assert isinstance(fig, go.Figure)


# --- Sector heatmap ---

def test_sector_heatmap_returns_figure():
    from src.visualization.sector_heatmap import create_sector_heatmap
    fig = create_sector_heatmap(_make_scored_df())
    assert isinstance(fig, go.Figure)


def test_sector_heatmap_empty():
    from src.visualization.sector_heatmap import create_sector_heatmap
    fig = create_sector_heatmap(pd.DataFrame())
    assert isinstance(fig, go.Figure)


# --- Prediction card ---

def test_prediction_display():
    from src.visualization.prediction_card import create_prediction_display
    result = create_prediction_display("bullish", 0.72, 0.58)
    assert result["direction"] == "Bullish"
    assert result["arrow"] == "↑"
    assert result["color"] == "#22c55e"


def test_prediction_display_bearish():
    from src.visualization.prediction_card import create_prediction_display
    result = create_prediction_display("bearish", 0.65, 0.55)
    assert result["arrow"] == "↓"
    assert result["color"] == "#ef4444"


def test_feature_importance_chart():
    from src.visualization.prediction_card import create_feature_importance_chart
    fig = create_feature_importance_chart(
        ["fear", "momentum", "volume"],
        [0.4, 0.35, 0.25],
    )
    assert isinstance(fig, go.Figure)


def test_feature_importance_empty():
    from src.visualization.prediction_card import create_feature_importance_chart
    fig = create_feature_importance_chart([], [])
    assert isinstance(fig, go.Figure)


# --- Dashboard app ---

def test_dash_app_creates():
    from src.dashboard.app import create_app
    app = create_app({})
    assert app is not None
    assert app.layout is not None
