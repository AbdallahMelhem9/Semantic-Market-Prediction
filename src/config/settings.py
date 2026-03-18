"""Loads YAML config and .env secrets into typed Settings."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


@dataclass
class NewsConfig:
    keywords: list[str] = field(default_factory=lambda: ["recession", "economic downturn"])
    days: int = 30
    page_size: int = 100


@dataclass
class LLMConfig:
    backend: str = "ollama"
    model: str = "llama3.1"
    temperature: float = 0.0
    batch_size: int = 5
    prompt_version: str = "v1"
    ollama_url: str = "http://localhost:11434"


@dataclass
class VisualizationConfig:
    theme: str = "dark"
    rolling_windows: list[int] = field(default_factory=lambda: [3, 7])


@dataclass
class PredictionConfig:
    test_split: float = 0.3
    features: list[str] = field(default_factory=list)


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8050
    debug: bool = True


@dataclass
class PathsConfig:
    root: str = ""
    data: str = ""
    cache: str = ""
    processed: str = ""
    models: str = ""
    prompts: str = ""


@dataclass
class SecretsConfig:
    newsapi_key: str = ""
    openai_api_key: str = ""
    huggingface_token: str = ""
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    finnhub_api_key: str = ""


@dataclass
class Settings:
    """Typed configuration object for the entire application."""

    news: NewsConfig = field(default_factory=NewsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    secrets: SecretsConfig = field(default_factory=SecretsConfig)
    sectors: list[str] = field(default_factory=list)
    sector_etfs: dict[str, str] = field(default_factory=dict)
    logging_level: str = "INFO"
    logging_format: str = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"

    def validate(self) -> list[str]:
        warnings = []
        if not self.secrets.newsapi_key:
            warnings.append("NEWSAPI_KEY is not set in .env — news ingestion will fail")
        if self.llm.backend == "openai" and not self.secrets.openai_api_key:
            warnings.append("OPENAI_API_KEY is not set in .env — OpenAI scoring will fail")
        if self.news.days < 1 or self.news.days > 30:
            warnings.append(f"news.days={self.news.days} — NewsAPI free tier supports max 30 days")
        return warnings


def _build_settings(raw: dict[str, Any], root: Path) -> Settings:
    news_raw = raw.get("news", {})
    llm_raw = raw.get("llm", {})
    viz_raw = raw.get("visualization", {})
    pred_raw = raw.get("prediction", {})
    server_raw = raw.get("server", {})
    log_raw = raw.get("logging", {})

    return Settings(
        news=NewsConfig(
            keywords=news_raw.get("keywords", ["recession", "economic downturn"]),
            days=news_raw.get("days", 30),
            page_size=news_raw.get("page_size", 100),
        ),
        llm=LLMConfig(
            backend=llm_raw.get("backend", "ollama"),
            model=llm_raw.get("model", "llama3.1"),
            temperature=llm_raw.get("temperature", 0.0),
            batch_size=llm_raw.get("batch_size", 5),
            prompt_version=llm_raw.get("prompt_version", "v1"),
            ollama_url=llm_raw.get("ollama_url", "http://localhost:11434"),
        ),
        visualization=VisualizationConfig(
            theme=viz_raw.get("theme", "dark"),
            rolling_windows=viz_raw.get("rolling_windows", [3, 7]),
        ),
        prediction=PredictionConfig(
            test_split=pred_raw.get("test_split", 0.3),
            features=pred_raw.get("features", []),
        ),
        server=ServerConfig(
            host=server_raw.get("host", "127.0.0.1"),
            port=server_raw.get("port", 8050),
            debug=server_raw.get("debug", True),
        ),
        paths=PathsConfig(
            root=str(root),
            data=str(root / "data"),
            cache=str(root / "data" / "cache"),
            processed=str(root / "data" / "processed"),
            models=str(root / "data" / "models"),
            prompts=str(root / "prompts"),
        ),
        secrets=SecretsConfig(
            newsapi_key=os.getenv("NEWSAPI_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            huggingface_token=os.getenv("HUGGINGFACE_TOKEN", ""),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            finnhub_api_key=os.getenv("FINNHUB_API_KEY", ""),
        ),
        sectors=raw.get("sectors", []),
        sector_etfs=raw.get("sector_etfs", {}),
        logging_level=log_raw.get("level", "INFO"),
        logging_format=log_raw.get("format", "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"),
    )


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load raw config dict (backwards-compatible)."""
    root = get_project_root()
    load_dotenv(root / ".env")

    if config_path is None:
        config_path = root / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["secrets"] = {
        "newsapi_key": os.getenv("NEWSAPI_KEY", ""),
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "huggingface_token": os.getenv("HUGGINGFACE_TOKEN", ""),
        "groq_api_key": os.getenv("GROQ_API_KEY", ""),
        "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", ""),
    }
    config["paths"] = {
        "root": str(root),
        "data": str(root / "data"),
        "cache": str(root / "data" / "cache"),
        "processed": str(root / "data" / "processed"),
        "models": str(root / "data" / "models"),
        "prompts": str(root / "prompts"),
    }

    return config


def load_settings(config_path: str | None = None) -> Settings:
    """Load config.yaml and return a typed Settings object."""
    root = get_project_root()
    load_dotenv(root / ".env")

    if config_path is None:
        config_path = root / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    settings = _build_settings(raw, root)

    warnings = settings.validate()
    for w in warnings:
        logger.warning(w)

    return settings
