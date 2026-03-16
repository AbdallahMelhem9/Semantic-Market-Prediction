from src.analysis.scorer import BaseSentimentScorer
from src.config.settings import Settings


def create_scorer(settings: Settings, region: str = "") -> BaseSentimentScorer:
    backend = settings.llm.backend.lower()

    if backend == "ollama":
        from src.analysis.ollama_scorer import OllamaScorer
        return OllamaScorer(settings, region)
    elif backend == "openai":
        from src.analysis.openai_scorer import OpenAIScorer
        return OpenAIScorer(settings)
    elif backend == "huggingface":
        from src.analysis.huggingface_scorer import HuggingFaceScorer
        return HuggingFaceScorer(settings, region)
    elif backend == "groq":
        from src.analysis.groq_scorer import GroqScorer
        return GroqScorer(settings, region)
    elif backend == "openrouter":
        from src.analysis.openrouter_scorer import OpenRouterScorer
        return OpenRouterScorer(settings, region)
    else:
        raise ValueError(f"Unknown LLM backend: {backend}. Use 'ollama', 'openai', 'huggingface', 'groq', or 'openrouter'.")
