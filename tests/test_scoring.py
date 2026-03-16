import json
from unittest.mock import patch, MagicMock
import pytest

from src.analysis.scorer import SentimentScore, BaseSentimentScorer
from src.analysis.response_validator import validate_score_response, _extract_json


# --- SentimentScore tests ---

def test_sentiment_score_default():
    score = SentimentScore.default()
    assert score.recession_fear == 5.0
    assert score.market_sentiment == "neutral"
    assert score.confidence == "low"


# --- Response validator tests ---

def test_validate_valid_json():
    raw = json.dumps({
        "recession_fear": 7.5,
        "market_sentiment": "bearish",
        "confidence": "high",
        "rationale": "Weak economic data"
    })
    score = validate_score_response(raw)
    assert score is not None
    assert score.recession_fear == 7.5
    assert score.market_sentiment == "bearish"
    assert score.confidence == "high"


def test_validate_json_in_markdown_fence():
    raw = '```json\n{"recession_fear": 3, "market_sentiment": "bullish", "confidence": "low", "rationale": "Strong jobs"}\n```'
    score = validate_score_response(raw)
    assert score is not None
    assert score.recession_fear == 3.0
    assert score.market_sentiment == "bullish"


def test_validate_json_with_surrounding_text():
    raw = 'Here is my analysis:\n{"recession_fear": 6, "market_sentiment": "neutral", "confidence": "medium", "rationale": "Mixed signals"}\nDone.'
    score = validate_score_response(raw)
    assert score is not None
    assert score.recession_fear == 6.0


def test_validate_rejects_missing_field():
    raw = json.dumps({"market_sentiment": "bearish", "confidence": "high"})
    assert validate_score_response(raw) is None


def test_validate_rejects_out_of_range():
    raw = json.dumps({"recession_fear": 15, "market_sentiment": "bearish", "confidence": "high"})
    assert validate_score_response(raw) is None


def test_validate_rejects_invalid_sentiment():
    raw = json.dumps({"recession_fear": 5, "market_sentiment": "maybe", "confidence": "high"})
    assert validate_score_response(raw) is None


def test_validate_rejects_garbage():
    assert validate_score_response("this is not json at all") is None


def test_validate_handles_sectors_string():
    raw = json.dumps({
        "recession_fear": 4,
        "market_sentiment": "neutral",
        "confidence": "medium",
        "rationale": "ok",
        "sectors": "Technology, Finance"
    })
    score = validate_score_response(raw)
    assert score is not None
    assert "Technology" in score.sectors
    assert "Finance" in score.sectors


def test_extract_json_plain():
    result = _extract_json('{"key": "value"}')
    assert result == {"key": "value"}


def test_extract_json_returns_none_for_invalid():
    assert _extract_json("no json here") is None


# --- Factory tests ---

def test_factory_creates_ollama_scorer():
    from src.config.settings import load_settings
    settings = load_settings()
    settings.llm.backend = "ollama"

    with patch("src.analysis.ollama_scorer.load_prompt", return_value="{{ARTICLE}}"):
        from src.analysis.scorer_factory import create_scorer
        scorer = create_scorer(settings)
        assert isinstance(scorer, BaseSentimentScorer)


def test_factory_rejects_unknown_backend():
    from src.config.settings import load_settings
    from src.analysis.scorer_factory import create_scorer
    settings = load_settings()
    settings.llm.backend = "unknown"
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        create_scorer(settings)


def test_factory_openai_not_implemented():
    from src.config.settings import load_settings
    from src.analysis.scorer_factory import create_scorer
    settings = load_settings()
    settings.llm.backend = "openai"
    with pytest.raises(NotImplementedError):
        create_scorer(settings)


# --- OllamaScorer tests (mocked) ---

@patch("src.analysis.ollama_scorer.load_prompt", return_value="Analyze: {{ARTICLE}}")
@patch("src.analysis.ollama_scorer.requests.post")
def test_ollama_scorer_scores_article(mock_post, mock_prompt):
    from src.config.settings import load_settings
    from src.analysis.ollama_scorer import OllamaScorer

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "response": json.dumps({
            "recession_fear": 8,
            "market_sentiment": "bearish",
            "confidence": "high",
            "rationale": "Severe downturn expected"
        })
    }
    mock_resp.raise_for_status.return_value = None
    mock_post.return_value = mock_resp

    settings = load_settings()
    scorer = OllamaScorer(settings)
    score = scorer.score_article({"title": "Test", "description": "Test", "content": "Test"})

    assert score.recession_fear == 8
    assert score.market_sentiment == "bearish"


@patch("src.analysis.ollama_scorer.load_prompt", return_value="Analyze: {{ARTICLE}}")
@patch("src.analysis.ollama_scorer.requests.post")
def test_ollama_scorer_returns_default_on_failure(mock_post, mock_prompt):
    from src.config.settings import load_settings
    from src.analysis.ollama_scorer import OllamaScorer

    mock_post.side_effect = Exception("Connection refused")

    settings = load_settings()
    scorer = OllamaScorer(settings)
    score = scorer.score_article({"title": "Test"})

    assert score.recession_fear == 5.0
    assert score.confidence == "low"


@patch("src.analysis.ollama_scorer.load_prompt", return_value="Analyze: {{ARTICLE}}")
@patch("src.analysis.ollama_scorer.requests.post")
def test_ollama_scorer_batch(mock_post, mock_prompt):
    from src.config.settings import load_settings
    from src.analysis.ollama_scorer import OllamaScorer

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "response": json.dumps({
            "recession_fear": 5,
            "market_sentiment": "neutral",
            "confidence": "medium",
            "rationale": "Mixed data"
        })
    }
    mock_resp.raise_for_status.return_value = None
    mock_post.return_value = mock_resp

    settings = load_settings()
    scorer = OllamaScorer(settings)
    articles = [{"title": f"Art {i}"} for i in range(3)]
    scores = scorer.score_batch(articles)

    assert len(scores) == 3
    assert all(s.recession_fear == 5 for s in scores)
