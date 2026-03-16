import json
import logging
import re
import time

import pandas as pd
import requests

from src.config.settings import Settings

logger = logging.getLogger(__name__)


def get_llm_forecast(settings: Settings, merged_df: pd.DataFrame, market_name: str = "S&P 500") -> dict:
    if merged_df.empty:
        return {}

    recent = merged_df.tail(7)
    data_summary = []
    for _, row in recent.iterrows():
        line = f"Date: {row.get('date')}, Fear: {row.get('avg_recession_fear', 'N/A'):.1f}, " \
               f"Sentiment: {row.get('avg_sentiment', 'N/A')}, " \
               f"{market_name}: {row.get('sp500_close', 'N/A')}"
        if "sp500_return" in row and pd.notna(row["sp500_return"]):
            line += f", Return: {row['sp500_return']:.2%}"
        data_summary.append(line)

    prompt = f"""You are a quantitative financial analyst. Based on the last 7 days of sentiment and market data, predict whether {market_name} will go UP or DOWN tomorrow.

Data:
{chr(10).join(data_summary)}

Rules:
- Analyze the trend in recession fear scores
- Consider the relationship between fear and market movement
- Factor in momentum and recent direction
- Return ONLY a JSON object with: direction ("bullish" or "bearish"), confidence (0.0 to 1.0), reasoning (1-2 sentences)

Return ONLY valid JSON, no other text."""

    response = _call_llm(settings, prompt)
    if not response:
        return {}

    try:
        match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return {
                "direction": data.get("direction", "neutral"),
                "confidence": float(data.get("confidence", 0.5)),
                "reasoning": data.get("reasoning", ""),
            }
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse LLM forecast: {e}")

    return {}


def _call_llm(settings: Settings, prompt: str) -> str | None:
    backend = settings.llm.backend
    api_key = settings.secrets.openrouter_api_key if backend == "openrouter" else settings.secrets.huggingface_token

    if not api_key:
        return None

    if backend == "openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    elif backend == "huggingface":
        url = "https://router.huggingface.co/together/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    else:
        return None

    for attempt in range(2):
        try:
            resp = requests.post(url, headers=headers, json={
                "model": settings.llm.model,
                "messages": [
                    {"role": "system", "content": "Return ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 200,
            }, timeout=45)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 0:
                time.sleep(2)
                continue
            logger.warning(f"LLM forecast failed after 2 attempts: {e}")
            return None
