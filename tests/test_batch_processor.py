import pandas as pd
from unittest.mock import MagicMock

from src.analysis.scorer import SentimentScore


def _make_settings(batch_size: int = 3):
    from src.config.settings import load_settings
    s = load_settings()
    s.llm.batch_size = batch_size
    return s


def _make_articles_df(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "title": [f"Article {i}" for i in range(n)],
        "description": [f"Desc {i}" for i in range(n)],
        "content": [f"Content {i}" for i in range(n)],
        "url": [f"https://example.com/{i}" for i in range(n)],
    })


def _make_mock_scorer(score: SentimentScore | None = None):
    if score is None:
        score = SentimentScore(recession_fear=6, market_sentiment="bearish", confidence="high", rationale="Weak data")

    scorer = MagicMock()
    scorer.score_batch.side_effect = lambda batch: [score] * len(batch)
    return scorer


def test_batch_processor_adds_score_columns():
    from src.analysis.batch_processor import score_articles_in_batches

    df = _make_articles_df(5)
    scorer = _make_mock_scorer()
    settings = _make_settings(batch_size=3)

    result = score_articles_in_batches(df, scorer, settings)

    assert "recession_fear" in result.columns
    assert "market_sentiment" in result.columns
    assert "confidence" in result.columns
    assert "rationale" in result.columns
    assert "sectors" in result.columns
    assert len(result) == 5


def test_batch_processor_calls_scorer():
    from src.analysis.batch_processor import score_articles_in_batches

    df = _make_articles_df(7)
    scorer = _make_mock_scorer()
    settings = _make_settings(batch_size=3)

    score_articles_in_batches(df, scorer, settings)

    # All articles sent in one score_batch call now (parallel internally)
    assert scorer.score_batch.call_count == 1
    assert len(scorer.score_batch.call_args[0][0]) == 7


def test_batch_processor_preserves_original_columns():
    from src.analysis.batch_processor import score_articles_in_batches

    df = _make_articles_df(2)
    scorer = _make_mock_scorer()
    settings = _make_settings(batch_size=5)

    result = score_articles_in_batches(df, scorer, settings)

    assert "title" in result.columns
    assert "url" in result.columns
    assert result.iloc[0]["title"] == "Article 0"


def test_batch_processor_correct_scores():
    from src.analysis.batch_processor import score_articles_in_batches

    score = SentimentScore(recession_fear=8.5, market_sentiment="bearish", confidence="high", rationale="Crisis imminent")
    df = _make_articles_df(3)
    scorer = _make_mock_scorer(score)
    settings = _make_settings(batch_size=10)

    result = score_articles_in_batches(df, scorer, settings)

    assert all(result["recession_fear"] == 8.5)
    assert all(result["market_sentiment"] == "bearish")
    assert all(result["rationale"] == "Crisis imminent")


def test_batch_processor_empty_df():
    from src.analysis.batch_processor import score_articles_in_batches

    df = pd.DataFrame()
    scorer = _make_mock_scorer()
    settings = _make_settings()

    result = score_articles_in_batches(df, scorer, settings)

    assert len(result) == 0
    assert scorer.score_batch.call_count == 0


def test_batch_processor_single_article():
    from src.analysis.batch_processor import score_articles_in_batches

    df = _make_articles_df(1)
    scorer = _make_mock_scorer()
    settings = _make_settings(batch_size=5)

    result = score_articles_in_batches(df, scorer, settings)

    assert len(result) == 1
    assert scorer.score_batch.call_count == 1
