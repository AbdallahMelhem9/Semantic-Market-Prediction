import json
import logging

import requests

from src.analysis.scorer import BaseSentimentScorer, SentimentScore
from src.analysis.prompt_loader import load_prompt
from src.analysis.response_validator import validate_score_response
from src.config.settings import Settings

logger = logging.getLogger(__name__)


class GroqScorer(BaseSentimentScorer):

    def __init__(self, settings: Settings, region: str = "") -> None:
        self.model = settings.llm.model or "llama-3.3-70b-versatile"
        self.api_key = settings.secrets.groq_api_key
        self.temperature = settings.llm.temperature
        self.prompt_template = load_prompt(settings, "sentiment_scoring", region)

        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set in .env")

    def _call_groq(self, prompt: str) -> str | None:
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": 300,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Groq API failed: {e}")
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
            raw = self._call_groq(prompt)
            if raw is None:
                continue
            score = validate_score_response(raw)
            if score is not None:
                return score
            logger.warning(f"Invalid Groq response (attempt {attempt + 1}/3)")

        logger.warning(f"Scoring failed for: {article.get('title', '')[:60]}")
        return SentimentScore.default()

    def score_batch(self, articles: list[dict]) -> list[SentimentScore]:
        return [self.score_article(a) for a in articles]
