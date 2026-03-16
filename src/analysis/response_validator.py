import json
import logging
import re

from src.analysis.scorer import SentimentScore

logger = logging.getLogger(__name__)

VALID_SENTIMENTS = {"bearish", "neutral", "bullish"}
VALID_CONFIDENCES = {"low", "medium", "high"}


def validate_score_response(raw_text: str) -> SentimentScore | None:
    """Parse and validate LLM response into a SentimentScore. Returns None if invalid."""
    parsed = _extract_json(raw_text)
    if parsed is None:
        return None

    # Check required fields
    recession_fear = parsed.get("recession_fear")
    sentiment = parsed.get("market_sentiment", "").lower()
    confidence = parsed.get("confidence", "").lower()
    rationale = parsed.get("rationale", "")

    if recession_fear is None:
        logger.debug("Missing recession_fear in response")
        return None

    try:
        recession_fear = float(recession_fear)
    except (ValueError, TypeError):
        return None

    if not (0 <= recession_fear <= 10):
        logger.debug(f"recession_fear out of range: {recession_fear}")
        return None

    if sentiment not in VALID_SENTIMENTS:
        logger.debug(f"Invalid market_sentiment: {sentiment}")
        return None

    if confidence not in VALID_CONFIDENCES:
        confidence = _normalize_confidence(confidence)
        if confidence is None:
            return None

    sectors = parsed.get("sectors", [])
    if isinstance(sectors, str):
        sectors = [s.strip() for s in sectors.split(",")]

    return SentimentScore(
        recession_fear=recession_fear,
        market_sentiment=sentiment,
        confidence=confidence,
        rationale=str(rationale),
        sectors=sectors,
    )


def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from LLM output, handling markdown fences."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*\n?({.*?})\s*\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _normalize_confidence(value: str) -> str | None:
    mapping = {"med": "medium", "hi": "high", "lo": "low"}
    return mapping.get(value)
