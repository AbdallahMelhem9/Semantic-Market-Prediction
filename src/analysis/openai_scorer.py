from src.analysis.scorer import BaseSentimentScorer, SentimentScore


class OpenAIScorer(BaseSentimentScorer):
    """Placeholder — will be implemented when OpenAI subscription is active."""

    def __init__(self, settings) -> None:
        raise NotImplementedError("OpenAI scorer not yet implemented. Use 'ollama' backend.")

    def score_article(self, article: dict) -> SentimentScore:
        raise NotImplementedError

    def score_batch(self, articles: list[dict]) -> list[SentimentScore]:
        raise NotImplementedError
