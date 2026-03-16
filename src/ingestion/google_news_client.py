import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def fetch_google_news(keywords: list[str], max_articles: int = 150) -> pd.DataFrame:
    """Fetch news via Google News RSS — free, no API key. Capped to last 28 days."""
    all_articles = []

    queries = [
        " ".join(keywords[:3]),
        " ".join(keywords[3:6]) if len(keywords) > 3 else "economy market",
        "stock market S&P 500 economy",
    ]

    for query in queries:
        if len(all_articles) >= max_articles:
            break

        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        try:
            resp = requests.get(url, headers={"User-Agent": "SemanticMarketPrediction/1.0"}, timeout=15)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            items = root.findall(".//item")

            for item in items:
                title_el = item.find("title")
                pub_el = item.find("pubDate")
                link_el = item.find("link")
                source_el = item.find("source")

                if title_el is None or pub_el is None:
                    continue

                all_articles.append({
                    "title": title_el.text or "",
                    "description": title_el.text or "",
                    "content": title_el.text or "",
                    "url": link_el.text if link_el is not None else "",
                    "source": source_el.text if source_el is not None else "Google News",
                    "source_tier": "tier-2",
                    "published_at": pub_el.text or "",
                    "api_source": "google_news",
                })

            logger.info(f"  Google News '{query[:30]}': {len(items)} articles")
        except Exception as e:
            logger.warning(f"Google News RSS failed for '{query[:30]}': {e}")

        time.sleep(0.5)

    if not all_articles:
        logger.info("No articles from Google News")
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["date"] = df["published_at"].dt.date
    df = df.dropna(subset=["date"])
    df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)

    # Keep articles from last 35 days (slightly wider than 28 to catch edge cases)
    from datetime import timedelta
    cutoff = datetime.now(tz=timezone.utc).date() - timedelta(days=35)
    df = df[df["date"] >= cutoff].reset_index(drop=True)

    # Filter out non-financial articles
    noise_words = ["recipe", "sports", "celebrity", "movie", "music", "game score",
                   "weather", "horoscope", "oral health", "petty", "savage", "dating"]
    before_filter = len(df)
    df = df[~df["title"].str.lower().str.contains("|".join(noise_words), na=False)]
    if len(df) < before_filter:
        logger.debug(f"Filtered {before_filter - len(df)} non-financial articles")

    if len(df) > max_articles:
        df = df.head(max_articles)

    logger.info(f"Google News: {len(df)} articles across {df['date'].nunique()} days")
    return df
