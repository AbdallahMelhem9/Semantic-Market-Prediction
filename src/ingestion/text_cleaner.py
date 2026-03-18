import re
import logging

logger = logging.getLogger(__name__)


def clean_article_text(text: str) -> str:
    """Clean raw article text for LLM input.

    Strips HTML tags, NewsAPI truncation markers, encoding artifacts,
    and normalises whitespace.
    """
    if not text or not isinstance(text, str):
        return ""

    # Strip HTML tags (<p>, <li>, <a href=...>, etc.)
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove NewsAPI truncation marker: [+1234 chars]
    text = re.sub(r"\[\+\d+ chars?\]", "", text)

    # Remove common encoding artifacts
    text = text.replace("\u00a0", " ")   # non-breaking space
    text = text.replace("\u200b", "")    # zero-width space
    text = text.replace("\ufeff", "")    # BOM
    text = text.replace("\r\n", "\n")    # Windows newlines

    # Collapse HTML entities that sometimes survive
    text = re.sub(r"&(amp|lt|gt|quot|apos|nbsp);", " ", text)
    text = re.sub(r"&#\d+;", " ", text)

    # Remove URLs (not useful for sentiment, can confuse LLM)
    text = re.sub(r"https?://\S+", "", text)

    # Collapse multiple whitespace / newlines into single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_article(article: dict) -> dict:
    """Clean title, description, and content fields of an article dict.

    Returns a new dict (does not mutate the original).
    """
    cleaned = article.copy()
    for field in ("title", "description", "content"):
        raw = cleaned.get(field, "")
        cleaned[field] = clean_article_text(raw or "")
    return cleaned


def is_usable_article(article: dict, min_length: int = 15) -> bool:
    """Check if an article has enough text to be worth scoring."""
    title = article.get("title", "")
    content = article.get("content", "") or article.get("description", "")
    combined = f"{title} {content}".strip()
    return len(combined) >= min_length
