import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fetch_historical_training_data(years: int = 2) -> pd.DataFrame:
    """Fetch historical S&P 500 + VIX data for model training.

    VIX serves as a fear/sentiment proxy since we don't have historical LLM scores.
    """
    import yfinance as yf

    end = date.today()
    start = end - timedelta(days=years * 365)

    logger.info(f"Fetching {years} years of historical data ({start} to {end})")

    try:
        sp500 = yf.Ticker("^GSPC").history(start=str(start), end=str(end))
        vix = yf.Ticker("^VIX").history(start=str(start), end=str(end))
    except Exception as e:
        logger.error(f"Failed to fetch historical data: {e}")
        return pd.DataFrame()

    if sp500.empty or vix.empty:
        logger.warning("Empty historical data returned")
        return pd.DataFrame()

    df = pd.DataFrame(index=sp500.index)
    df["sp500_close"] = sp500["Close"]
    df["sp500_return"] = sp500["Close"].pct_change()
    df["sp500_direction"] = (df["sp500_return"] > 0).astype(int)

    # VIX as fear proxy (higher VIX = more fear)
    df["vix_close"] = vix["Close"].reindex(df.index, method="ffill")
    df["vix_scaled"] = df["vix_close"] / df["vix_close"].max() * 10  # scale to 0-10 like our fear score

    df["avg_recession_fear"] = df["vix_scaled"]
    df["rolling_3d"] = df["avg_recession_fear"].rolling(3, min_periods=1).mean()
    df["rolling_7d"] = df["avg_recession_fear"].rolling(7, min_periods=1).mean()
    df["volatility"] = df["avg_recession_fear"].rolling(7, min_periods=2).std()
    df["momentum"] = df["avg_recession_fear"].diff()

    if "Volume" in sp500.columns:
        df["article_count"] = sp500["Volume"] / sp500["Volume"].mean() * 10  # normalize
    else:
        df["article_count"] = 10

    df["sentiment_numeric"] = -df["vix_close"].pct_change().apply(
        lambda x: 1 if x < -0.02 else (-1 if x > 0.02 else 0)
    )

    df["date"] = df.index.date
    df = df.reset_index(drop=True)
    df = df.dropna()

    logger.info(f"Built {len(df)} days of historical training data")
    return df
