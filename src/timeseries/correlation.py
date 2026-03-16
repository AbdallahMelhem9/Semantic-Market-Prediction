import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_lag_correlations(
    merged_df: pd.DataFrame,
    sentiment_col: str = "avg_recession_fear",
    market_col: str = "sp500_return",
    lags: list[int] = None,
) -> pd.DataFrame:
    """Compute Pearson and Spearman correlation at various lag offsets.

    Lag N means: sentiment at day T correlated with market at day T+N.
    """
    if lags is None:
        lags = [0, 1, 2, 3, 5]

    if merged_df.empty or sentiment_col not in merged_df.columns or market_col not in merged_df.columns:
        logger.warning("Insufficient data for correlation analysis")
        return pd.DataFrame()

    results = []
    sentiment = merged_df[sentiment_col].values
    market = merged_df[market_col].values

    for lag in lags:
        if lag == 0:
            s, m = sentiment, market
        else:
            s = sentiment[:-lag]
            m = market[lag:]

        if len(s) < 2:
            continue

        # Drop NaN pairs
        mask = ~(np.isnan(s) | np.isnan(m))
        s_clean, m_clean = s[mask], m[mask]

        if len(s_clean) < 2:
            continue

        pearson_r, pearson_p = stats.pearsonr(s_clean, m_clean)
        spearman_r, spearman_p = stats.spearmanr(s_clean, m_clean)

        results.append({
            "lag": lag,
            "pearson_r": round(pearson_r, 4),
            "pearson_p": round(pearson_p, 4),
            "spearman_r": round(spearman_r, 4),
            "spearman_p": round(spearman_p, 4),
            "n_samples": len(s_clean),
        })

    if not results:
        return pd.DataFrame()

    corr_df = pd.DataFrame(results)
    logger.info(f"Computed correlations at {len(results)} lag offsets")
    return corr_df
