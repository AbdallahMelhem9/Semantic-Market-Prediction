import json
import logging
from typing import Any

import requests

from src.analysis.scorer import BaseSentimentScorer, SentimentScore
from src.analysis.prompt_loader import load_prompt
from src.analysis.response_validator import validate_score_response
from src.config.settings import Settings

logger = logging.getLogger(__name__)


class OllamaScorer(BaseSentimentScorer):

    def __init__(self, settings: Settings, region: str = "") -> None:
        self.model = settings.llm.model
        self.base_url = settings.llm.ollama_url
        self.temperature = settings.llm.temperature
        self.batch_size = settings.llm.batch_size
        self.prompt_template = load_prompt(settings, "sentiment_scoring", region)

    def _call_ollama(self, prompt: str) -> str | None:
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            return None

    def _build_prompt(self, article: dict) -> str:
        title = article.get("title", "")
        description = article.get("description", "")
        content = article.get("content", "")

        article_text = f"Title: {title}\nDescription: {description}\nContent: {content}"
        return self.prompt_template.replace("{{ARTICLE}}", article_text)

    def score_article(self, article: dict) -> SentimentScore:
        prompt = self._build_prompt(article)

        for attempt in range(3):
            raw = self._call_ollama(prompt)
            if raw is None:
                continue

            score = validate_score_response(raw)
            if score is not None:
                return score

            logger.warning(f"Invalid LLM response (attempt {attempt + 1}/3)")

        logger.warning(f"Scoring failed after 3 attempts for: {article.get('title', '')[:60]}")
        return SentimentScore.default()

    def score_batch(self, articles: list[dict]) -> list[SentimentScore]:
        results = []
        for article in articles:
            score = self.score_article(article)
            results.append(score)
        return results
