import logging
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_sp500(start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch S&P 500 daily close prices from yfinance."""
    import yfinance as yf

    # Pad dates slightly to handle alignment
    start = (start_date - timedelta(days=5)).isoformat()
    end = (end_date + timedelta(days=1)).isoformat()

    try:
        ticker = yf.Ticker("^GSPC")
        hist = ticker.history(start=start, end=end)
    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 data: {e}")
        return pd.DataFrame()

    if hist.empty:
        logger.warning("No S&P 500 data returned")
        return pd.DataFrame()

    df = hist[["Close"]].reset_index()
    df.columns = ["date", "sp500_close"]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["sp500_return"] = df["sp500_close"].pct_change()
    df["sp500_direction"] = (df["sp500_return"] > 0).astype(int)

    logger.info(f"Fetched {len(df)} days of S&P 500 data")
    return df


def fetch_sector_etfs(start_date: date, end_date: date, etf_map: dict[str, str]) -> pd.DataFrame:
    """Fetch sector ETF daily close prices."""
    import yfinance as yf

    start = (start_date - timedelta(days=5)).isoformat()
    end = (end_date + timedelta(days=1)).isoformat()

    all_data = []

    for sector, symbol in etf_map.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start, end=end)
            if hist.empty:
                continue

            df = hist[["Close"]].reset_index()
            df.columns = ["date", "close"]
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df["sector"] = sector
            df["symbol"] = symbol
            df["daily_return"] = df["close"].pct_change()
            all_data.append(df)
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} ({sector}): {e}")

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    logger.info(f"Fetched ETF data for {len(etf_map)} sectors")
    return result


def merge_sentiment_and_market(
    sentiment_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge daily sentiment with market data on matching dates."""
    if sentiment_df.empty or market_df.empty:
        logger.warning("Cannot merge — one or both DataFrames are empty")
        return sentiment_df

    merged = pd.merge(sentiment_df, market_df, on="date", how="left")
    for col in ["sp500_close", "sp500_return", "sp500_direction"]:
        if col in merged.columns:
            merged[col] = merged[col].ffill()
    logger.info(f"Merged {len(merged)} days with sentiment data (including non-trading days)")
    return merged
