import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)

FINNHUB_BASE = "https://finnhub.io/api/v1"

# US tickers across sectors
TICKERS_US = [
    "AAPL", "MSFT", "AMZN", "GOOGL",  # Tech
    "JPM", "GS", "BAC",                 # Finance
    "XOM", "CVX",                        # Energy
    "UNH", "JNJ",                        # Healthcare
    "WMT", "HD",                         # Consumer
    "CAT", "BA",                         # Industrial
]

# EU companies via US-listed ADRs (free tier supports US exchange tickers only)
TICKERS_EU = [
    "SAP", "ASML",       # Tech (already US-listed)
    "UL", "DEO",         # Consumer (Unilever, Diageo)
    "SHEL", "BP",        # Energy (Shell, BP)
    "AZN", "GSK", "NVO", # Healthcare (AstraZeneca, GSK, Novo Nordisk)
    "BCS", "ING",        # Finance (Barclays, ING Group)
    "TM", "SNY",         # Industrial/Pharma (Toyota, Sanofi)
]


def fetch_finnhub_news(
    api_key: str,
    keywords: list[str],
    days: int = 30,
    max_articles: int = 300,
    region: str = "us",
) -> pd.DataFrame:
    """Fetch market news from Finnhub — free tier, headline + summary."""
    if not api_key:
        logger.info("FINNHUB_API_KEY not set — skipping Finnhub")
        return pd.DataFrame()

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    now = datetime.now(tz=timezone.utc)
    all_articles = []

    # General market news (returns latest ~1-2 days)
    articles = _fetch_general_news(api_key, cutoff)
    all_articles.extend(articles)
    logger.info(f"Finnhub general news: {len(articles)} articles")

    # Company news across ALL tickers — no early exit, process everything
    tickers = TICKERS_US if region == "us" else TICKERS_EU
    from_date = (now - timedelta(days=days)).strftime("%Y-%m-%d")
    to_date = now.strftime("%Y-%m-%d")

    for ticker in tickers:
        ticker_articles = _fetch_company_news(api_key, ticker, from_date, to_date)
        if ticker_articles:
            logger.info(f"  Finnhub {ticker}: {len(ticker_articles)} articles")
        all_articles.extend(ticker_articles)
        time.sleep(0.15)

    if not all_articles:
        logger.info("No articles from Finnhub")
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["date"] = df["published_at"].dt.date
    df = df.dropna(subset=["date"])

    # Filter to requested date range
    cutoff_date = cutoff.date()
    df = df[df["date"] >= cutoff_date]

    # Deduplicate by URL
    df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)

    # Spread across days: keep up to 10 articles per day for even coverage
    if not df.empty:
        sampled = []
        for d, group in df.groupby("date"):
            sampled.append(group.head(10))
        df = pd.concat(sampled, ignore_index=True)

    # Final cap and sort
    df = df.sort_values("published_at", ascending=False).head(max_articles).reset_index(drop=True)

    logger.info(f"Finnhub: {len(df)} articles across {df['date'].nunique()} days")
    return df


def _fetch_general_news(api_key: str, cutoff: datetime) -> list[dict]:
    """Fetch general market news."""
    articles = []
    try:
        resp = requests.get(
            f"{FINNHUB_BASE}/news",
            params={"category": "general", "token": api_key},
            timeout=15,
        )
        if resp.status_code == 429:
            logger.warning("Finnhub rate limited, waiting 2s...")
            time.sleep(2)
            resp = requests.get(
                f"{FINNHUB_BASE}/news",
                params={"category": "general", "token": api_key},
                timeout=15,
            )
        resp.raise_for_status()
        data = resp.json()

        for item in data:
            dt = datetime.fromtimestamp(item.get("datetime", 0), tz=timezone.utc)
            if dt < cutoff:
                continue
            articles.append(_normalize(item, dt))

    except Exception as e:
        logger.warning(f"Finnhub general news failed: {e}")

    return articles


def _fetch_company_news(api_key: str, symbol: str, from_date: str, to_date: str) -> list[dict]:
    """Fetch company-specific news with date range."""
    articles = []
    try:
        resp = requests.get(
            f"{FINNHUB_BASE}/company-news",
            params={
                "symbol": symbol,
                "from": from_date,
                "to": to_date,
                "token": api_key,
            },
            timeout=15,
        )
        if resp.status_code == 429:
            time.sleep(2)
            resp = requests.get(
                f"{FINNHUB_BASE}/company-news",
                params={
                    "symbol": symbol,
                    "from": from_date,
                    "to": to_date,
                    "token": api_key,
                },
                timeout=15,
            )
        if resp.status_code == 403:
            logger.debug(f"Finnhub {symbol}: 403 (not available on free tier)")
            return []
        resp.raise_for_status()
        data = resp.json()

        for item in data[:100]:
            dt = datetime.fromtimestamp(item.get("datetime", 0), tz=timezone.utc)
            articles.append(_normalize(item, dt))

    except Exception as e:
        logger.warning(f"Finnhub company news ({symbol}) failed: {e}")

    return articles


def _normalize(item: dict, dt: datetime) -> dict:
    return {
        "title": item.get("headline", ""),
        "description": item.get("summary", ""),
        "content": item.get("summary", ""),
        "url": item.get("url", ""),
        "source": item.get("source", ""),
        "source_tier": "tier-2",
        "published_at": dt.isoformat(),
        "api_source": "finnhub",
    }
