import logging

import requests

from src.analysis.scorer import BaseSentimentScorer, SentimentScore
from src.analysis.prompt_loader import load_prompt
from src.analysis.response_validator import validate_score_response
from src.config.settings import Settings

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
API_URL = "https://router.huggingface.co/together/v1/chat/completions"


class HuggingFaceScorer(BaseSentimentScorer):

    def __init__(self, settings: Settings, region: str = "") -> None:
        self.model = settings.llm.model or DEFAULT_MODEL
        self.token = settings.secrets.huggingface_token
        self.temperature = settings.llm.temperature
        self.prompt_template = load_prompt(settings, "sentiment_scoring", region)
        self._quota_exhausted = False

        if not self.token:
            raise ValueError("HUGGINGFACE_TOKEN not set in .env")

    def _call_hf(self, prompt: str) -> str | None:
        try:
            resp = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": self.temperature,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if "402" in str(e) or "Payment Required" in str(e):
                logger.error("HuggingFace quota exhausted — switching to defaults for remaining articles")
                self._quota_exhausted = True
            else:
                logger.error(f"HuggingFace API failed: {e}")
            return None

    def _build_prompt(self, article: dict) -> str:
        title = article.get("title", "")
        description = article.get("description", "")
        content = article.get("content", "")
        article_text = f"Title: {title}\nDescription: {description}\nContent: {content}"
        return self.prompt_template.replace("{{ARTICLE}}", article_text)

    def score_article(self, article: dict) -> SentimentScore:
        if self._quota_exhausted:
            return SentimentScore.default()

        prompt = self._build_prompt(article)

        for attempt in range(2):
            raw = self._call_hf(prompt)
            if raw is None:
                continue
            score = validate_score_response(raw)
            if score is not None:
                return score
            logger.warning(f"Invalid HF response (attempt {attempt + 1}/2)")

        logger.warning(f"HF scoring failed for: {article.get('title', '')[:60]}")
        return SentimentScore.default()

    def score_batch(self, articles: list[dict]) -> list[SentimentScore]:
        return [self.score_article(a) for a in articles]
