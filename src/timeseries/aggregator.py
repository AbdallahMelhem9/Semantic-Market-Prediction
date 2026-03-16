import logging
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TIER_WEIGHTS = {"tier-1": 1.5, "tier-2": 1.0}


def aggregate_daily_sentiment(scored_df: pd.DataFrame, rolling_windows: list[int] = None) -> pd.DataFrame:
    """Aggregate scored articles into a daily sentiment timeseries."""
    if scored_df.empty:
        logger.warning("No scored articles to aggregate")
        return pd.DataFrame()

    if rolling_windows is None:
        rolling_windows = [3, 7]

    df = scored_df.copy()

    # Ensure date column exists and is proper date type
    if "date" not in df.columns and "published_at" in df.columns:
        df["date"] = pd.to_datetime(df["published_at"]).dt.date
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    # Compute weights
    df["weight"] = df.get("source_tier", "tier-2").map(TIER_WEIGHTS).fillna(1.0)

    # Weighted daily aggregation
    daily = df.groupby("date").apply(_weighted_daily_stats, include_groups=False).reset_index()

    daily = daily.sort_values("date").reset_index(drop=True)

    # Rolling averages
    for window in rolling_windows:
        daily[f"rolling_{window}d"] = daily["avg_recession_fear"].rolling(window, min_periods=1).mean()

    # Sentiment momentum (day-over-day change)
    daily["momentum"] = daily["avg_recession_fear"].diff()

    # Volatility (rolling std dev over 7 days)
    daily["volatility"] = daily["avg_recession_fear"].rolling(7, min_periods=2).std()

    # Mark weekends/holidays
    daily["is_trading_day"] = daily["date"].apply(_is_trading_day)

    logger.info(f"Aggregated {len(daily)} days from {len(scored_df)} articles")
    return daily


def _weighted_daily_stats(group: pd.DataFrame) -> pd.Series:
    weights = group["weight"]
    total_weight = weights.sum()

    if total_weight == 0:
        avg_fear = group["recession_fear"].mean()
    else:
        avg_fear = (group["recession_fear"] * weights).sum() / total_weight

    # Map sentiment to numeric for averaging
    sentiment_map = {"bearish": -1, "neutral": 0, "bullish": 1}
    sentiment_vals = group["market_sentiment"].map(sentiment_map).fillna(0)
    avg_sentiment_num = (sentiment_vals * weights).sum() / total_weight

    if avg_sentiment_num < -0.3:
        avg_sentiment = "bearish"
    elif avg_sentiment_num > 0.3:
        avg_sentiment = "bullish"
    else:
        avg_sentiment = "neutral"

    return pd.Series({
        "avg_recession_fear": round(avg_fear, 2),
        "avg_sentiment": avg_sentiment,
        "avg_sentiment_numeric": round(avg_sentiment_num, 2),
        "article_count": len(group),
    })


def _is_trading_day(d) -> bool:
    if isinstance(d, date):
        return d.weekday() < 5  # Mon-Fri
    return True


def export_timeseries(daily_df: pd.DataFrame, output_path: str) -> None:
    if daily_df.empty:
        return
    daily_df.to_csv(output_path, index=False)
    logger.info(f"Exported timeseries to {output_path}")
