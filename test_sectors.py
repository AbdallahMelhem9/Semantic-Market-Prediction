"""
Test script — loads articles from cache, cleans them, runs Finance/Energy
sector daily assessment. Does NOT write to cache. Just prints results.
"""
import copy
import json
import logging
from datetime import timedelta

import pandas as pd

from src.config.settings import load_settings
from src.config.logging_setup import setup_logging
from src.analysis.daily_assessor import assess_daily_sentiment
from src.ingestion.text_cleaner import clean_article, is_usable_article
from src.pipeline import _fetch_market_index

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_and_clean(scored_path: str) -> pd.DataFrame:
    with open(scored_path, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data["articles"])
    logger.info(f"Loaded {len(df)} articles from {scored_path}")

    # Clean text fields
    cleaned_count = 0
    for idx, row in df.iterrows():
        article = row.to_dict()
        cleaned = clean_article(article)
        if cleaned.get("title") != article.get("title"):
            cleaned_count += 1
        for field in ("title", "description", "content"):
            df.at[idx, field] = cleaned.get(field, "")

    # Filter unusable
    before = len(df)
    df = df[df.apply(lambda r: is_usable_article(r.to_dict()), axis=1)].reset_index(drop=True)
    logger.info(f"Cleaned {cleaned_count} articles, filtered {before - len(df)} unusable, {len(df)} remaining")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    return df


def run_test(region_key: str, scored_path: str, market_index: str):
    logger.info(f"\n{'='*50}")
    logger.info(f"TESTING SECTOR ASSESSMENT: {region_key.upper()}")

    scored_df = load_and_clean(scored_path)

    # Fetch market data
    dates = scored_df["date"].dropna().tolist()
    min_date = min(dates) - timedelta(days=14)
    max_date = max(dates)
    market_df = _fetch_market_index(market_index, min_date, max_date)

    settings = load_settings()
    gpt_settings = copy.deepcopy(settings)
    gpt_settings.llm.backend = "openrouter"
    gpt_settings.llm.model = "openai/gpt-5.4"

    # Test only last 3 days to save cost
    from datetime import date
    cutoff = date.today() - timedelta(days=3)
    test_df = scored_df[scored_df["date"] >= cutoff].copy()
    logger.info(f"Testing on {test_df['date'].nunique()} days ({cutoff} to today), {len(test_df)} articles")

    for sec in ["Financials", "Energy"]:
        logger.info(f"\n--- {sec} ---")
        result = assess_daily_sentiment(copy.deepcopy(gpt_settings), test_df.copy(), market_df, sector=sec)
        if not result:
            logger.info(f"  No {sec} articles found in test window")
            continue
        for a in result:
            d = str(a.get("date", ""))[:10]
            fear = a.get("daily_fear", "?")
            driver = a.get("key_driver", "")[:100]
            reasoning = a.get("reasoning", "")[:120]
            logger.info(f"  {d}: fear={fear}, driver: {driver}")
            logger.info(f"    reasoning: {reasoning}")


def main():
    settings = load_settings()
    setup_logging(settings)

    run_test("us", "data/cache/scored/gpt54_us.json", "^GSPC")

    logger.info(f"\n{'='*50}")
    logger.info("Test complete! Results above are NOT saved to cache.")


if __name__ == "__main__":
    main()
