import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def engineer_features(daily_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (features_df, target_series). Target is next-day S&P 500 direction: 1=up, 0=down."""
    if daily_df.empty:
        logger.warning("No data for feature engineering")
        return pd.DataFrame(), pd.Series(dtype=int)

    df = daily_df.copy()

    features = pd.DataFrame(index=df.index)
    features["avg_recession_fear"] = df["avg_recession_fear"]

    if "rolling_3d" in df.columns:
        features["rolling_3d"] = df["rolling_3d"]
    if "rolling_7d" in df.columns:
        features["rolling_7d"] = df["rolling_7d"]
    if "volatility" in df.columns:
        features["volatility"] = df["volatility"]
    if "article_count" in df.columns:
        features["article_count"] = df["article_count"]
    if "momentum" in df.columns:
        features["momentum"] = df["momentum"]
    if "avg_sentiment_numeric" in df.columns:
        features["sentiment_numeric"] = df["avg_sentiment_numeric"]

    # Target: next-day S&P direction (shift -1 to look ahead)
    if "sp500_direction" not in df.columns:
        if "sp500_return" in df.columns:
            df["sp500_direction"] = (df["sp500_return"] > 0).astype(int)
        else:
            logger.warning("No market data for target variable")
            return pd.DataFrame(), pd.Series(dtype=int)

    target = df["sp500_direction"].shift(-1)  # next day's direction

    # Drop rows with NaN in features or target
    valid = features.notna().all(axis=1) & target.notna()
    features = features[valid]
    target = target[valid].astype(int)

    logger.info(f"Engineered {len(features.columns)} features, {len(features)} samples")
    return features, target
