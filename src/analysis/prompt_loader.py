import logging
from pathlib import Path

from src.config.settings import Settings

logger = logging.getLogger(__name__)


def load_prompt(settings: Settings, prompt_name: str, region: str = "") -> str:
    version = settings.llm.prompt_version
    prompts_dir = Path(settings.paths.prompts)

    # Try region-specific prompt first
    if region:
        regional_file = prompts_dir / f"{prompt_name}_{region}_{version}.txt"
        if regional_file.exists():
            logger.debug(f"Loaded regional prompt: {regional_file.name}")
            return regional_file.read_text(encoding="utf-8")

    # Fall back to generic prompt
    generic_file = prompts_dir / f"{prompt_name}_{version}.txt"
    if generic_file.exists():
        return generic_file.read_text(encoding="utf-8")

    logger.warning(f"Prompt file not found, using fallback for: {prompt_name}")
    return _fallback_prompt(prompt_name)


def _fallback_prompt(prompt_name: str) -> str:
    if "sentiment" in prompt_name:
        return (
            "You are a senior financial analyst. Analyze the following article and return "
            "a JSON object with: recession_fear (0-10), market_sentiment (bearish/neutral/bullish), "
            "confidence (low/medium/high), rationale (1 sentence), sectors (list).\n\n{{ARTICLE}}"
        )
    return "Analyze the following text:\n\n{{ARTICLE}}"
