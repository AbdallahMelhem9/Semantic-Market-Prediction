from pathlib import Path
from src.config.settings import load_settings


def test_sentiment_prompt_exists():
    settings = load_settings()
    prompt_file = Path(settings.paths.prompts) / "sentiment_scoring_v1.txt"
    assert prompt_file.exists()


def test_sector_prompt_exists():
    settings = load_settings()
    prompt_file = Path(settings.paths.prompts) / "sector_classification_v1.txt"
    assert prompt_file.exists()


def test_chatbot_prompt_exists():
    settings = load_settings()
    prompt_file = Path(settings.paths.prompts) / "chatbot_system_v1.txt"
    assert prompt_file.exists()


def test_sentiment_prompt_has_placeholder():
    settings = load_settings()
    content = (Path(settings.paths.prompts) / "sentiment_scoring_v1.txt").read_text()
    assert "{{ARTICLE}}" in content


def test_sentiment_prompt_has_few_shot_examples():
    settings = load_settings()
    content = (Path(settings.paths.prompts) / "sentiment_scoring_v1.txt").read_text()
    assert "Example 1:" in content
    assert "Example 2:" in content
    assert "Example 3:" in content
    assert "recession_fear" in content
    assert "market_sentiment" in content


def test_sentiment_prompt_has_role():
    settings = load_settings()
    content = (Path(settings.paths.prompts) / "sentiment_scoring_v1.txt").read_text()
    assert "senior financial analyst" in content


def test_prompt_loader_loads_versioned_file():
    from src.analysis.prompt_loader import load_prompt
    settings = load_settings()
    settings.llm.prompt_version = "v1"
    prompt = load_prompt(settings, "sentiment_scoring")
    assert "recession_fear" in prompt
    assert "{{ARTICLE}}" in prompt


def test_prompt_loader_fallback_on_missing():
    from src.analysis.prompt_loader import load_prompt
    settings = load_settings()
    settings.llm.prompt_version = "v999"
    prompt = load_prompt(settings, "sentiment_scoring")
    assert "{{ARTICLE}}" in prompt  # fallback still has placeholder


def test_prompt_loader_fallback_unknown_name():
    from src.analysis.prompt_loader import load_prompt
    settings = load_settings()
    settings.llm.prompt_version = "v999"
    prompt = load_prompt(settings, "totally_unknown")
    assert "{{ARTICLE}}" in prompt
