"""Tests for NewsAPI client."""

from unittest.mock import patch, MagicMock
import pandas as pd


def _make_settings(api_key: str = "test_key"):
    """Create a minimal Settings object for testing."""
    from src.config.settings import load_settings
    settings = load_settings()
    settings.secrets.newsapi_key = api_key
    settings.news.days = 7
    settings.news.keywords = ["recession", "inflation"]
    settings.news.page_size = 100
    return settings


def _mock_response(articles: list[dict], total: int = None, status: str = "ok"):
    """Create a mock requests.Response."""
    if total is None:
        total = len(articles)
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "status": status,
        "totalResults": total,
        "articles": articles,
    }
    mock.raise_for_status.return_value = None
    return mock


SAMPLE_ARTICLES = [
    {
        "source": {"name": "Reuters"},
        "title": "Fed signals patience on rate cuts",
        "description": "The Federal Reserve indicated...",
        "content": "Full article content here...",
        "url": "https://reuters.com/article/1",
        "publishedAt": "2026-03-10T12:00:00Z",
    },
    {
        "source": {"name": "Bloomberg"},
        "title": "Recession fears mount amid weak data",
        "description": "Economic indicators suggest...",
        "content": "Full article content here...",
        "url": "https://bloomberg.com/article/2",
        "publishedAt": "2026-03-11T08:00:00Z",
    },
    {
        "source": {"name": "Some Blog"},
        "title": "Market outlook uncertain",
        "description": "Analysis of current conditions...",
        "content": "Full article content here...",
        "url": "https://someblog.com/article/3",
        "publishedAt": "2026-03-12T15:00:00Z",
    },
]


@patch("src.ingestion.news_client.requests.get")
def test_fetch_news_returns_dataframe(mock_get):
    """fetch_news should return a DataFrame with expected columns."""
    mock_get.return_value = _mock_response(SAMPLE_ARTICLES)
    settings = _make_settings()

    from src.ingestion.news_client import fetch_news
    df = fetch_news(settings)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    expected_cols = ["title", "description", "content", "url", "source", "source_tier", "published_at", "date"]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"


@patch("src.ingestion.news_client.requests.get")
def test_fetch_news_source_tier_classification(mock_get):
    """tier-1 sources should be correctly classified."""
    mock_get.return_value = _mock_response(SAMPLE_ARTICLES)
    settings = _make_settings()

    from src.ingestion.news_client import fetch_news
    df = fetch_news(settings)

    reuters_row = df[df["url"].str.contains("reuters")]
    assert reuters_row.iloc[0]["source_tier"] == "tier-1"

    blog_row = df[df["url"].str.contains("someblog")]
    assert blog_row.iloc[0]["source_tier"] == "tier-2"


@patch("src.ingestion.news_client.requests.get")
def test_fetch_news_deduplicates_by_url(mock_get):
    """Duplicate articles (same URL) should be removed."""
    duped = SAMPLE_ARTICLES + [SAMPLE_ARTICLES[0]]  # duplicate first article
    mock_get.return_value = _mock_response(duped, total=len(duped))
    settings = _make_settings()

    from src.ingestion.news_client import fetch_news
    df = fetch_news(settings)

    assert len(df) == 3  # deduped to 3 unique URLs


def test_fetch_news_empty_when_no_api_key():
    """Should return empty DataFrame when API key is missing."""
    settings = _make_settings(api_key="")

    from src.ingestion.news_client import fetch_news
    df = fetch_news(settings)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


@patch("src.ingestion.news_client.requests.get")
def test_fetch_news_handles_api_error(mock_get):
    """Should handle API errors gracefully."""
    mock_get.return_value = _mock_response([], status="error")
    mock_get.return_value.json.return_value = {
        "status": "error",
        "message": "API rate limit exceeded",
    }
    settings = _make_settings()

    from src.ingestion.news_client import fetch_news
    df = fetch_news(settings)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


@patch("src.ingestion.news_client.requests.get")
def test_fetch_news_handles_request_exception(mock_get):
    """Should handle network errors gracefully."""
    import requests as req
    mock_get.side_effect = req.ConnectionError("Connection refused")
    settings = _make_settings()

    from src.ingestion.news_client import fetch_news
    df = fetch_news(settings)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


@patch("src.ingestion.news_client.requests.get")
def test_fetch_news_paginates(mock_get):
    """Should paginate when total results exceed page size."""
    page1_articles = SAMPLE_ARTICLES[:2]
    page2_articles = [SAMPLE_ARTICLES[2]]

    responses = [
        _mock_response(page1_articles, total=3),
        _mock_response(page2_articles, total=3),
    ]
    mock_get.side_effect = responses
    settings = _make_settings()
    settings.news.page_size = 2

    from src.ingestion.news_client import fetch_news
    df = fetch_news(settings)

    assert len(df) >= 2
    assert mock_get.call_count >= 1


def test_build_query():
    """Query builder should OR-join keywords in quotes."""
    from src.ingestion.news_client import _build_query
    query = _build_query(["recession", "inflation"])
    assert '"recession"' in query
    assert '"inflation"' in query
    assert " OR " in query


def test_classify_source_tier():
    """Source tier classification should work correctly."""
    from src.ingestion.news_client import _classify_source_tier
    assert _classify_source_tier("https://reuters.com/article/1") == "tier-1"
    assert _classify_source_tier("https://bloomberg.com/news/2") == "tier-1"
    assert _classify_source_tier("https://wsj.com/article/3") == "tier-1"
    assert _classify_source_tier("https://cnbc.com/article/4") == "tier-1"
    assert _classify_source_tier("https://randomsite.com/article") == "tier-2"
    assert _classify_source_tier(None) == "tier-2"
    assert _classify_source_tier("") == "tier-2"
