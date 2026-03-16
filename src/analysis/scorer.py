from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SentimentScore:
    recession_fear: float = 5.0
    market_sentiment: str = "neutral"
    confidence: str = "medium"
    rationale: str = ""
    sectors: list[str] = field(default_factory=list)

    @staticmethod
    def default() -> "SentimentScore":
        return SentimentScore(
            recession_fear=5.0,
            market_sentiment="neutral",
            confidence="low",
            rationale="Scoring failed — default neutral applied",
        )


class BaseSentimentScorer(ABC):

    @abstractmethod
    def score_article(self, article: dict) -> SentimentScore:
        ...

    @abstractmethod
    def score_batch(self, articles: list[dict]) -> list[SentimentScore]:
        ...
