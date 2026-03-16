import pandas as pd
import numpy as np
from datetime import date
from unittest.mock import patch, MagicMock


def _make_scored_df(n_days: int = 10, articles_per_day: int = 3) -> pd.DataFrame:
    rows = []
    base = date(2026, 3, 1)
    for d in range(n_days):
        current_date = date(2026, 3, 1 + d)
        for a in range(articles_per_day):
            rows.append({
                "title": f"Article {d}-{a}",
                "url": f"https://example.com/{d}/{a}",
                "source_tier": "tier-1" if a == 0 else "tier-2",
                "recession_fear": 3.0 + d * 0.5 + a * 0.1,
                "market_sentiment": ["bearish", "neutral", "bullish"][a % 3],
                "confidence": "high",
                "rationale": "test",
                "date": current_date,
            })
    return pd.DataFrame(rows)


# --- Aggregator tests ---

def test_aggregate_returns_daily_rows():
    from src.timeseries.aggregator import aggregate_daily_sentiment
    df = _make_scored_df(10, 3)
    daily = aggregate_daily_sentiment(df)
    assert len(daily) == 10
    assert "avg_recession_fear" in daily.columns
    assert "article_count" in daily.columns


def test_aggregate_rolling_averages():
    from src.timeseries.aggregator import aggregate_daily_sentiment
    df = _make_scored_df(10, 2)
    daily = aggregate_daily_sentiment(df, rolling_windows=[3, 7])
    assert "rolling_3d" in daily.columns
    assert "rolling_7d" in daily.columns
    assert not daily["rolling_3d"].isna().all()


def test_aggregate_momentum():
    from src.timeseries.aggregator import aggregate_daily_sentiment
    df = _make_scored_df(5, 2)
    daily = aggregate_daily_sentiment(df)
    assert "momentum" in daily.columns
    # First day should be NaN
    assert pd.isna(daily.iloc[0]["momentum"])


def test_aggregate_volatility():
    from src.timeseries.aggregator import aggregate_daily_sentiment
    df = _make_scored_df(10, 2)
    daily = aggregate_daily_sentiment(df)
    assert "volatility" in daily.columns


def test_aggregate_article_count():
    from src.timeseries.aggregator import aggregate_daily_sentiment
    df = _make_scored_df(5, 4)
    daily = aggregate_daily_sentiment(df)
    assert all(daily["article_count"] == 4)


def test_aggregate_weighted_by_tier():
    from src.timeseries.aggregator import aggregate_daily_sentiment
    # tier-1 articles with fear=10, tier-2 with fear=0
    df = pd.DataFrame({
        "date": [date(2026, 3, 1)] * 2,
        "recession_fear": [10.0, 0.0],
        "market_sentiment": ["bearish", "bullish"],
        "source_tier": ["tier-1", "tier-2"],
    })
    daily = aggregate_daily_sentiment(df)
    # tier-1 weight 1.5, tier-2 weight 1.0 → (10*1.5 + 0*1.0) / 2.5 = 6.0
    assert daily.iloc[0]["avg_recession_fear"] == 6.0


def test_aggregate_empty():
    from src.timeseries.aggregator import aggregate_daily_sentiment
    result = aggregate_daily_sentiment(pd.DataFrame())
    assert len(result) == 0


def test_aggregate_marks_trading_days():
    from src.timeseries.aggregator import aggregate_daily_sentiment
    # March 1, 2026 is a Sunday
    df = pd.DataFrame({
        "date": [date(2026, 3, 1), date(2026, 3, 2)],  # Sun, Mon
        "recession_fear": [5.0, 5.0],
        "market_sentiment": ["neutral", "neutral"],
        "source_tier": ["tier-2", "tier-2"],
    })
    daily = aggregate_daily_sentiment(df)
    assert daily.iloc[0]["is_trading_day"] == False  # Sunday
    assert daily.iloc[1]["is_trading_day"] == True   # Monday


def test_export_timeseries(tmp_path):
    from src.timeseries.aggregator import aggregate_daily_sentiment, export_timeseries
    df = _make_scored_df(5, 2)
    daily = aggregate_daily_sentiment(df)
    out = str(tmp_path / "test_output.csv")
    export_timeseries(daily, out)
    loaded = pd.read_csv(out)
    assert len(loaded) == 5


# --- Correlation tests ---

def test_correlation_computes_at_lags():
    from src.timeseries.correlation import compute_lag_correlations
    np.random.seed(42)
    n = 30
    df = pd.DataFrame({
        "avg_recession_fear": np.random.randn(n).cumsum() + 5,
        "sp500_return": np.random.randn(n) * 0.01,
    })
    result = compute_lag_correlations(df, lags=[0, 1, 3])
    assert len(result) == 3
    assert "pearson_r" in result.columns
    assert "spearman_r" in result.columns
    assert all(result["n_samples"] > 0)


def test_correlation_empty_df():
    from src.timeseries.correlation import compute_lag_correlations
    result = compute_lag_correlations(pd.DataFrame())
    assert len(result) == 0


def test_correlation_values_in_range():
    from src.timeseries.correlation import compute_lag_correlations
    np.random.seed(42)
    df = pd.DataFrame({
        "avg_recession_fear": np.random.randn(20) + 5,
        "sp500_return": np.random.randn(20) * 0.01,
    })
    result = compute_lag_correlations(df, lags=[0, 1])
    assert all((-1 <= result["pearson_r"]) & (result["pearson_r"] <= 1))
    assert all((-1 <= result["spearman_r"]) & (result["spearman_r"] <= 1))


# --- Market data tests (mocked) ---

@patch("yfinance.Ticker")
def test_fetch_sp500_returns_dataframe(mock_ticker_cls):
    from src.timeseries.market_data import fetch_sp500

    mock_ticker = MagicMock()
    mock_hist = pd.DataFrame({
        "Close": [4500.0, 4520.0, 4480.0],
    }, index=pd.to_datetime(["2026-03-10", "2026-03-11", "2026-03-12"]))
    mock_hist.index.name = "Date"
    mock_ticker.history.return_value = mock_hist
    mock_ticker_cls.return_value = mock_ticker

    result = fetch_sp500(date(2026, 3, 10), date(2026, 3, 12))
    assert len(result) == 3
    assert "sp500_close" in result.columns
    assert "sp500_return" in result.columns
    assert "sp500_direction" in result.columns


@patch("yfinance.Ticker")
def test_fetch_sp500_handles_failure(mock_ticker_cls):
    from src.timeseries.market_data import fetch_sp500
    mock_ticker_cls.side_effect = Exception("Network error")
    result = fetch_sp500(date(2026, 3, 10), date(2026, 3, 12))
    assert len(result) == 0


def test_merge_sentiment_and_market():
    from src.timeseries.market_data import merge_sentiment_and_market
    sentiment = pd.DataFrame({
        "date": [date(2026, 3, 10), date(2026, 3, 11), date(2026, 3, 12)],
        "avg_recession_fear": [5.0, 6.0, 7.0],
    })
    market = pd.DataFrame({
        "date": [date(2026, 3, 10), date(2026, 3, 11)],
        "sp500_close": [4500, 4520],
    })
    merged = merge_sentiment_and_market(sentiment, market)
    assert len(merged) == 2  # inner join


def test_merge_empty_market():
    from src.timeseries.market_data import merge_sentiment_and_market
    sentiment = pd.DataFrame({"date": [date(2026, 3, 10)], "avg_recession_fear": [5.0]})
    merged = merge_sentiment_and_market(sentiment, pd.DataFrame())
    assert len(merged) == 1  # returns sentiment as-is
