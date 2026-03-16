"""Tests for the configuration system."""

import os
from pathlib import Path

import pytest


def test_load_config_returns_dict():
    """load_config returns a dictionary with all expected sections."""
    from src.config.settings import load_config
    config = load_config()
    assert isinstance(config, dict)
    assert "news" in config
    assert "llm" in config
    assert "visualization" in config
    assert "prediction" in config
    assert "secrets" in config
    assert "paths" in config


def test_load_settings_returns_settings_object():
    """load_settings returns a typed Settings object."""
    from src.config.settings import load_settings
    settings = load_settings()
    assert settings.news.days == 30
    assert settings.news.page_size == 100
    assert settings.llm.backend in ("ollama", "huggingface", "openai", "openrouter", "groq")
    assert settings.llm.temperature == 0.0
    assert settings.visualization.theme == "dark"
    assert settings.prediction.test_split == 0.3


def test_settings_news_keywords():
    """News keywords should be loaded from config."""
    from src.config.settings import load_settings
    settings = load_settings()
    assert len(settings.news.keywords) >= 1


def test_settings_paths_are_valid():
    """All paths in settings should point to existing directories or be valid paths."""
    from src.config.settings import load_settings
    settings = load_settings()
    assert Path(settings.paths.root).exists()
    assert Path(settings.paths.data).exists()
    assert Path(settings.paths.prompts).exists()


def test_settings_secrets_from_env(monkeypatch):
    """Secrets should be loaded from environment variables."""
    monkeypatch.setenv("NEWSAPI_KEY", "test_key_123")
    from src.config.settings import load_settings
    settings = load_settings()
    assert settings.secrets.newsapi_key == "test_key_123"


def test_settings_validate_warns_missing_newsapi(monkeypatch):
    """Validation should warn when NEWSAPI_KEY is missing."""
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    from src.config.settings import load_settings
    settings = load_settings()
    settings.secrets.newsapi_key = ""
    warnings = settings.validate()
    assert any("NEWSAPI_KEY" in w for w in warnings)


def test_settings_validate_warns_openai_when_needed():
    """Validation should warn when backend is openai but key is missing."""
    from src.config.settings import load_settings
    settings = load_settings()
    settings.llm.backend = "openai"
    settings.secrets.openai_api_key = ""
    warnings = settings.validate()
    assert any("OPENAI_API_KEY" in w for w in warnings)


def test_missing_config_raises_error():
    """Loading a non-existent config should raise FileNotFoundError."""
    from src.config.settings import load_settings
    with pytest.raises(FileNotFoundError):
        load_settings(config_path="/nonexistent/config.yaml")


def test_settings_server_config():
    """Server config should have defaults."""
    from src.config.settings import load_settings
    settings = load_settings()
    assert settings.server.host == "127.0.0.1"
    assert settings.server.port == 8050


def test_settings_sectors_loaded():
    """Sectors should be loaded from config."""
    from src.config.settings import load_settings
    settings = load_settings()
    assert len(settings.sectors) >= 4
    assert "Technology" in settings.sectors


def test_settings_sector_etfs_loaded():
    """Config should load without sector_etfs error (now in regions)."""
    from src.config.settings import load_settings
    settings = load_settings()
    # sector_etfs now lives under regions config, not top-level
    assert isinstance(settings.sector_etfs, dict)
