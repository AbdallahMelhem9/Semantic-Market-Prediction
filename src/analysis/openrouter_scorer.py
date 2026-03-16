import logging

import requests

from src.analysis.scorer import BaseSentimentScorer, SentimentScore
from src.analysis.prompt_loader import load_prompt
from src.analysis.response_validator import validate_score_response
from src.config.settings import Settings

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct"


class OpenRouterScorer(BaseSentimentScorer):

    def __init__(self, settings: Settings, region: str = "") -> None:
        self.model = settings.llm.model or DEFAULT_MODEL
        self.api_key = settings.secrets.openrouter_api_key
        self.temperature = settings.llm.temperature
        self.prompt_template = load_prompt(settings, "sentiment_scoring", region)

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set in .env")

    def _call_api(self, prompt: str) -> str | None:
        import time

        for retry in range(3):
            try:
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "Return ONLY valid JSON. No markdown, no explanation, no extra text."},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": self.temperature,
                        "max_tokens": 200,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                if retry < 2:
                    time.sleep(2 ** retry)  # 1s, 2s backoff
                    continue
                logger.error(f"OpenRouter API failed after 3 retries: {e}")
                return None

    def _build_prompt(self, article: dict) -> str:
        title = article.get("title", "")
        description = article.get("description", "")
        content = article.get("content", "")
        article_text = f"Title: {title}\nDescription: {description}\nContent: {content}"
        return self.prompt_template.replace("{{ARTICLE}}", article_text)

    def score_article(self, article: dict) -> SentimentScore:
        prompt = self._build_prompt(article)

        for attempt in range(2):
            raw = self._call_api(prompt)
            if raw is None:
                continue
            score = validate_score_response(raw)
            if score is not None:
                return score
            logger.debug(f"Invalid response (attempt {attempt + 1}/2), retrying")

        logger.warning(f"Scoring failed for: {article.get('title', '')[:60]}")
        return SentimentScore.default()

    def score_batch(self, articles: list[dict]) -> list[SentimentScore]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(articles)
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(self.score_article, a): i for i, a in enumerate(articles)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = SentimentScore.default()

        return results
