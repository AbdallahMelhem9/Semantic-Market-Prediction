"""Tests for the cache manager."""

import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture
def settings(tmp_path):
    """Create settings pointing to a temp cache directory."""
    from src.config.settings import load_settings
    s = load_settings()
    s.paths.cache = str(tmp_path / "cache")
    return s


@pytest.fixture
def cache(settings):
    """Create a CacheManager with temp directory."""
    from src.ingestion.cache import CacheManager
    return CacheManager(settings)


@pytest.fixture
def sample_df():
    """Sample news DataFrame for testing."""
    return pd.DataFrame({
        "title": ["Article 1", "Article 2"],
        "description": ["Desc 1", "Desc 2"],
        "content": ["Content 1", "Content 2"],
        "url": ["https://reuters.com/1", "https://bloomberg.com/2"],
        "source": ["Reuters", "Bloomberg"],
        "source_tier": ["tier-1", "tier-1"],
        "published_at": pd.to_datetime(["2026-03-10", "2026-03-11"], utc=True),
        "date": pd.to_datetime(["2026-03-10", "2026-03-11"]).date,
    })


def test_cache_dirs_created(cache, settings):
    """Cache directories should be created on init."""
    assert Path(settings.paths.cache, "news_raw").exists()
    assert Path(settings.paths.cache, "scored").exists()


def test_no_cached_news_initially(cache, settings):
    """Should report no cache before any save."""
    assert cache.has_fresh_cached_news(settings) is False


def test_save_and_load_news(cache, settings, sample_df):
    """Should save and load news articles correctly."""
    cache.save_news(sample_df, settings)
    assert cache.has_fresh_cached_news(settings) is True

    loaded = cache.load_news(settings)
    assert len(loaded) == 2
    assert "title" in loaded.columns
    assert loaded.iloc[0]["title"] == "Article 1"


def test_load_news_empty_when_no_cache(cache, settings):
    """Should return empty DataFrame when no cache exists."""
    loaded = cache.load_news(settings)
    assert isinstance(loaded, pd.DataFrame)
    assert len(loaded) == 0


def test_save_empty_df_skips(cache, settings):
    """Should not create cache file for empty DataFrame."""
    cache.save_news(pd.DataFrame(), settings)
    assert cache.has_fresh_cached_news(settings) is False


def test_save_and_load_scored(cache):
    """Should save and load scored articles."""
    scored_df = pd.DataFrame({
        "title": ["Art 1"],
        "recession_fear": [7.5],
        "market_sentiment": ["bearish"],
        "confidence": ["high"],
    })

    cache.save_scored(scored_df, "test_scored")
    loaded = cache.load_scored("test_scored")

    assert len(loaded) == 1
    assert loaded.iloc[0]["recession_fear"] == 7.5
    assert loaded.iloc[0]["market_sentiment"] == "bearish"


def test_load_scored_empty_when_no_cache(cache):
    """Should return empty DataFrame for non-existent scored cache."""
    loaded = cache.load_scored("nonexistent")
    assert isinstance(loaded, pd.DataFrame)
    assert len(loaded) == 0


def test_clear_cache(cache, settings, sample_df):
    """Should remove all cached data."""
    cache.save_news(sample_df, settings)
    assert cache.has_fresh_cached_news(settings) is True

    cache.clear()
    assert cache.has_fresh_cached_news(settings) is False


def test_different_params_different_cache(cache, settings, sample_df):
    """Different query params should create different cache files."""
    cache.save_news(sample_df, settings)
    assert cache.has_fresh_cached_news(settings) is True

    # Change days — should be a different cache key
    settings.news.days = 7
    assert cache.has_fresh_cached_news(settings) is False
