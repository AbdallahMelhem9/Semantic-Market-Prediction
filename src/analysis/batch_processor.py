import logging

import pandas as pd

from src.analysis.scorer import BaseSentimentScorer, SentimentScore
from src.config.settings import Settings

logger = logging.getLogger(__name__)


def score_articles_in_batches(
    df: pd.DataFrame,
    scorer: BaseSentimentScorer,
    settings: Settings,
) -> pd.DataFrame:
    """Score all articles, processing in configurable batch sizes.

    Adds sentiment columns to the DataFrame and returns the enriched result.
    """
    if df.empty:
        logger.warning("No articles to score")
        return df

    batch_size = settings.llm.batch_size
    articles = df.to_dict(orient="records")
    total = len(articles)
    all_scores: list[SentimentScore] = []

    logger.info(f"Scoring {total} articles (parallel processing)")
    all_scores = scorer.score_batch(articles)

    # Attach scores to dataframe
    result = df.copy()
    result["recession_fear"] = [s.recession_fear for s in all_scores]
    result["market_sentiment"] = [s.market_sentiment for s in all_scores]
    result["confidence"] = [s.confidence for s in all_scores]
    result["rationale"] = [s.rationale for s in all_scores]
    result["sectors"] = [s.sectors for s in all_scores]

    scored_count = sum(1 for s in all_scores if s.confidence != "low" or s.rationale != "Scoring failed — default neutral applied")
    logger.info(f"Scoring complete: {scored_count}/{total} successfully scored")

    return result
