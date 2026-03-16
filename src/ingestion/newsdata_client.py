import logging
import os
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://newsdata.io/api/1/latest"


def fetch_newsdata(keywords: list[str], country: str = "", days: int = 30) -> pd.DataFrame:
    api_key = os.getenv("NEWSDATA_API_KEY", "")
    if not api_key:
        logger.info("NEWSDATA_API_KEY not set — skipping NewsData.io")
        return pd.DataFrame()

    query = " OR ".join(keywords[:5])  # NewsData limits query complexity

    params = {
        "apikey": api_key,
        "q": query,
        "language": "en",
        "category": "business",
    }
    if country:
        params["country"] = country

    all_articles = []

    # Fewer keyword batches to stay under rate limit (200 req/day)
    keyword_batches = [keywords[:5]]
    extra_terms = ["stock market economy", "oil energy crisis"]
    for term in extra_terms:
        keyword_batches.append([term])

    for batch_keywords in keyword_batches:
        batch_query = " OR ".join(batch_keywords)
        batch_params = {**params, "q": batch_query}
        next_page = None

        for page_num in range(2):
            if next_page:
                batch_params["page"] = next_page

            try:
                resp = requests.get(BASE_URL, params=batch_params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.error(f"NewsData.io request failed: {e}")
                break

            if data.get("status") != "success":
                break

            results = data.get("results", [])
            if not results:
                break

            all_articles.extend(results)
            next_page = data.get("nextPage")
            if not next_page:
                break

            time.sleep(0.3)
        time.sleep(0.3)

    if not all_articles:
        logger.info("No articles from NewsData.io")
        return pd.DataFrame()

    rows = []
    for a in all_articles:
        rows.append({
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "content": a.get("content", "") or a.get("description", ""),
            "url": a.get("link", ""),
            "source": a.get("source_name", ""),
            "source_tier": "tier-2",
            "published_at": a.get("pubDate", ""),
            "api_source": "newsdata",
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        df["date"] = df["published_at"].dt.date

    logger.info(f"NewsData.io: fetched {len(df)} articles")
    return df
