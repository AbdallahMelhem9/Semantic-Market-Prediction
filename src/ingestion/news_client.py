import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

from src.config.settings import Settings

logger = logging.getLogger(__name__)

TIER1_SOURCES = {"reuters.com", "bloomberg.com", "wsj.com", "cnbc.com"}
BASE_URL_EVERYTHING = "https://newsapi.org/v2/everything"
BASE_URL_HEADLINES = "https://newsapi.org/v2/top-headlines"


def _classify_source_tier(source_url: str | None) -> str:
    if not source_url:
        return "tier-2"
    url_lower = source_url.lower()
    for tier1 in TIER1_SOURCES:
        if tier1 in url_lower:
            return "tier-1"
    return "tier-2"


def _build_query(keywords: list[str]) -> str:
    return " OR ".join(f'"{kw}"' for kw in keywords)


def _api_request(url: str, params: dict, api_key: str, silent_fail: bool = False) -> dict | None:
    headers = {"X-Api-Key": api_key, "User-Agent": "SemanticMarketPrediction/1.0"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 429:
            if not silent_fail:
                logger.warning("NewsAPI rate limit hit — pausing")
            time.sleep(2)
            return None
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "ok":
            return data
        if not silent_fail:
            logger.error(f"NewsAPI error: {data.get('message', 'Unknown')}")
        return None
    except requests.RequestException as e:
        if not silent_fail:
            logger.error(f"NewsAPI request failed: {e}")
        return None


def fetch_news(settings: Settings) -> pd.DataFrame:
    api_key = settings.secrets.newsapi_key
    if not api_key:
        logger.error("NEWSAPI_KEY is not set — cannot fetch news")
        return pd.DataFrame()

    news_config = settings.news
    now = datetime.now(tz=timezone.utc)
    to_date = now.strftime("%Y-%m-%d")

    logger.info(f"Fetching news: {len(news_config.keywords)} keywords, {news_config.days} days")

    all_articles: list[dict[str, Any]] = []

    # Strategy 1: Try 'everything' with full date range
    query = _build_query(news_config.keywords)
    from_date = (now - timedelta(days=news_config.days)).strftime("%Y-%m-%d")
    data = _api_request(BASE_URL_EVERYTHING, {
        "q": query, "from": from_date, "to": to_date,
        "language": "en", "sortBy": "publishedAt", "pageSize": 100,
    }, api_key, silent_fail=True)

    if data and data.get("totalResults", 0) > 0:
        all_articles.extend(data.get("articles", []))
        logger.info(f"  Everything endpoint: {len(data.get('articles', []))} articles")
    else:
        # Strategy 2: Minimal API calls to stay under rate limit
        logger.info("Falling back to headlines + historical...")

        # A: Business headlines (1 call)
        data = _api_request(BASE_URL_HEADLINES, {
            "country": "us", "category": "business", "pageSize": 100,
        }, api_key)
        if data and data.get("articles"):
            all_articles.extend(data["articles"])
            logger.info(f"  Business headlines: {len(data['articles'])} articles")

        # B: Top 3 keywords only (3 calls)
        for keyword in news_config.keywords[:3]:
            time.sleep(0.5)
            data = _api_request(BASE_URL_HEADLINES, {
                "q": keyword, "pageSize": 100,
            }, api_key)
            if data and data.get("articles"):
                all_articles.extend(data["articles"])
                logger.info(f"  '{keyword}': {len(data['articles'])} articles")

        # C: Historical via 'everything' — 2 calls for different date windows
        for days_back, kw in [(7, "recession OR inflation"), (21, "economy OR market crash")]:
            time.sleep(0.5)
            from_d = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")
            to_d = (now - timedelta(days=max(0, days_back - 7))).strftime("%Y-%m-%d")
            data = _api_request(BASE_URL_EVERYTHING, {
                "q": kw, "from": from_d, "to": to_d,
                "language": "en", "sortBy": "relevancy", "pageSize": 50,
            }, api_key, silent_fail=True)
            if data and data.get("articles"):
                all_articles.extend(data["articles"])
                logger.info(f"  Historical({days_back}d): {len(data['articles'])} articles")

    if not all_articles:
        logger.warning("No articles fetched from NewsAPI")
        return pd.DataFrame()

    df = _articles_to_dataframe(all_articles)

    # Deduplicate
    before = len(df)
    df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    if before != len(df):
        logger.info(f"Deduplicated: {before} → {len(df)} articles")

    # Cap at 100
    if len(df) > 100:
        df = df.sort_values("published_at", ascending=False).head(100).reset_index(drop=True)
        logger.info("Capped at 100 articles")

    logger.info(f"Fetched {len(df)} unique articles")
    return df


def _articles_to_dataframe(articles: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for article in articles:
        source_name = article.get("source", {}).get("name", "")
        source_url = article.get("url", "")
        rows.append({
            "title": article.get("title", ""),
            "description": article.get("description", ""),
            "content": article.get("content", ""),
            "url": article.get("url", ""),
            "source": source_name,
            "source_tier": _classify_source_tier(source_url),
            "published_at": article.get("publishedAt", ""),
        })

    df = pd.DataFrame(rows)
    if not df.empty and "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        df["date"] = df["published_at"].dt.date

    return df
