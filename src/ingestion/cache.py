"""Local JSON cache for news and scored articles — prevents redundant API calls."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.settings import Settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages local file caching for news and scored articles."""

    def __init__(self, settings: Settings) -> None:
        self.news_raw_dir = Path(settings.paths.cache) / "news_raw"
        self.scored_dir = Path(settings.paths.cache) / "scored"
        self.news_raw_dir.mkdir(parents=True, exist_ok=True)
        self.scored_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, days: int, keywords_hash: str) -> str:
        return f"news_{days}d_{keywords_hash}"

    def _keywords_hash(self, keywords: list[str]) -> str:
        import hashlib
        joined = "|".join(sorted(keywords))
        return hashlib.md5(joined.encode()).hexdigest()[:8]

    def has_fresh_cached_news(self, settings: Settings, max_age_hours: int = 24) -> bool:
        key = self._cache_key(
            settings.news.days,
            self._keywords_hash(settings.news.keywords),
        )
        cache_file = self.news_raw_dir / f"{key}.json"
        if not cache_file.exists():
            return False

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_at = pd.Timestamp(data.get("cached_at", ""))
            age_hours = (pd.Timestamp.now() - cached_at).total_seconds() / 3600
            if age_hours < max_age_hours:
                logger.info(f"Cache is {age_hours:.1f}h old (fresh, max {max_age_hours}h)")
                return True
            else:
                logger.info(f"Cache is {age_hours:.1f}h old (stale, refetching)")
                return False
        except Exception:
            return False

    def save_news(self, df: pd.DataFrame, settings: Settings) -> None:
        if df.empty:
            logger.debug("Empty DataFrame — skipping cache save")
            return

        key = self._cache_key(
            settings.news.days,
            self._keywords_hash(settings.news.keywords),
        )
        cache_file = self.news_raw_dir / f"{key}.json"

        existing = self.load_news(settings)
        if not existing.empty:
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
            logger.info(f"Accumulated cache: {len(existing)} existing + new = {len(df)} total")

        records = df.copy()
        for col in records.columns:
            if pd.api.types.is_datetime64_any_dtype(records[col]):
                records[col] = records[col].astype(str)
            elif records[col].dtype == "object":
                records[col] = records[col].apply(
                    lambda x: str(x) if not isinstance(x, str) else x
                )

        data = {
            "cached_at": pd.Timestamp.now().isoformat(),
            "article_count": len(records),
            "articles": records.to_dict(orient="records"),
        }

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Cached {len(records)} articles to {cache_file.name}")

    def load_news(self, settings: Settings) -> pd.DataFrame:
        key = self._cache_key(
            settings.news.days,
            self._keywords_hash(settings.news.keywords),
        )
        cache_file = self.news_raw_dir / f"{key}.json"

        if not cache_file.exists():
            logger.debug("No cached news found")
            return pd.DataFrame()

        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        articles = data.get("articles", [])
        if not articles:
            return pd.DataFrame()

        df = pd.DataFrame(articles)

        if "published_at" in df.columns:
            df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

        logger.info(
            f"Loaded {len(df)} articles from cache "
            f"(cached at {data.get('cached_at', 'unknown')})"
        )
        return df

    def save_scored(self, df: pd.DataFrame, cache_name: str = "scored_articles") -> None:
        if df.empty:
            return

        cache_file = self.scored_dir / f"{cache_name}.json"
        records = df.copy()
        for col in records.columns:
            if pd.api.types.is_datetime64_any_dtype(records[col]):
                records[col] = records[col].astype(str)
            elif records[col].dtype == "object":
                records[col] = records[col].apply(
                    lambda x: str(x) if not isinstance(x, str) else x
                )

        data = {
            "cached_at": pd.Timestamp.now().isoformat(),
            "count": len(records),
            "articles": records.to_dict(orient="records"),
        }

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Cached {len(records)} scored articles to {cache_file.name}")

    def load_scored(self, cache_name: str = "scored_articles") -> pd.DataFrame:
        cache_file = self.scored_dir / f"{cache_name}.json"

        if not cache_file.exists():
            return pd.DataFrame()

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"Corrupt cache file {cache_file.name} — deleting")
            cache_file.unlink()
            return pd.DataFrame()

        articles = data.get("articles", [])
        if not articles:
            return pd.DataFrame()

        df = pd.DataFrame(articles)
        logger.info(f"Loaded {len(df)} scored articles from cache")
        return df

    def clear(self) -> None:
        import shutil
        for d in [self.news_raw_dir, self.scored_dir]:
            if d.exists():
                shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)
        logger.info("Cache cleared")
