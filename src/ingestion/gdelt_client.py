import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch_gdelt(keywords: list[str], days: int = 30, max_articles: int = 100) -> pd.DataFrame:
    """Fetch historical news from GDELT — fully free, no API key needed."""
    all_articles = []

    # Split into weekly windows for better date coverage
    now = datetime.now(tz=timezone.utc)
    windows = []
    for week in range(0, min(days, 28), 7):
        end = now - timedelta(days=week)
        start = now - timedelta(days=week + 7)
        windows.append((start, end))

    articles_per_window = max(max_articles // len(windows), 10)

    for start, end in windows:
        if len(all_articles) >= max_articles:
            break

        start_str = start.strftime("%Y%m%d%H%M%S")
        end_str = end.strftime("%Y%m%d%H%M%S")
        query = " OR ".join(keywords[:4])

        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": str(articles_per_window),
            "format": "json",
            "startdatetime": start_str,
            "enddatetime": end_str,
            "sourcelang": "english",
        }

        import time
        time.sleep(5)  # GDELT needs spacing between requests

        try:
            resp = requests.get(GDELT_DOC_API, params=params, timeout=30)
            if resp.status_code == 429:
                logger.info("GDELT rate limited, waiting 10s...")
                time.sleep(10)
                resp = requests.get(GDELT_DOC_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            articles = data.get("articles", [])
            if articles:
                all_articles.extend(articles)
                week_label = f"{start.strftime('%m/%d')}-{end.strftime('%m/%d')}"
                logger.info(f"  GDELT {week_label}: {len(articles)} articles")
        except Exception as e:
            logger.warning(f"GDELT request failed: {e}")

    if not all_articles:
        logger.info("No articles from GDELT")
        return pd.DataFrame()

    rows = []
    for a in all_articles:
        rows.append({
            "title": a.get("title", ""),
            "description": a.get("seendate", ""),
            "content": a.get("title", ""),
            "url": a.get("url", ""),
            "source": a.get("domain", ""),
            "source_tier": "tier-2",
            "published_at": a.get("seendate", ""),
            "api_source": "gdelt",
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        df["date"] = df["published_at"].dt.date
        df = df.dropna(subset=["date"])

    logger.info(f"GDELT: fetched {len(df)} articles across {df['date'].nunique() if not df.empty else 0} days")
    return df
