import json
import logging
import re
from pathlib import Path

import pandas as pd
import requests

from src.config.settings import Settings

logger = logging.getLogger(__name__)


def assess_daily_sentiment(
    settings: Settings,
    scored_df: pd.DataFrame,
    market_df: pd.DataFrame = None,
) -> list[dict]:
    """For each day, give the LLM ALL articles + YESTERDAY's market data.

    Strictly no future leakage: day N only sees market data up to day N-1.
    """
    if scored_df.empty:
        return []

    prompt_template = _load_prompt(settings)
    scored_df = scored_df.copy()
    scored_df["date"] = pd.to_datetime(scored_df["date"]).dt.date

    # Build market lookup from actual data
    market_lookup = {}
    if market_df is not None and not market_df.empty:
        mdf = market_df.copy()
        mdf["date"] = pd.to_datetime(mdf["date"]).dt.date
        for _, row in mdf.iterrows():
            market_lookup[row["date"]] = {
                "close": row.get("sp500_close"),
                "return": row.get("sp500_return"),
            }

    dates = sorted(scored_df["date"].unique())
    assessments = []
    prev_sentiment = None

    for i, d in enumerate(dates):
        day_articles = scored_df[scored_df["date"] == d]

        # Build article summary for THIS day only
        article_lines = []
        for _, row in day_articles.iterrows():
            title = str(row.get("title", ""))[:120]
            fear = row.get("recession_fear", 5)
            sentiment = row.get("market_sentiment", "neutral")
            sectors = row.get("sectors", [])
            article_lines.append(f"- \"{title}\" (fear={fear}, {sentiment}, sectors={sectors})")

        articles_text = "\n".join(article_lines)

        # Build market context: all market days BEFORE current date (no future leakage)
        market_lines = []
        all_market_dates = sorted([md for md in market_lookup.keys() if md < d])
        for prev_date in all_market_dates[-5:]:  # last 5 trading days before this date
            m = market_lookup[prev_date]
            close = m.get("close")
            ret = m.get("return")
            if close is not None:
                line = f"  {prev_date}: Close={close:.0f}"
                if ret is not None and not pd.isna(ret):
                    line += f", Return={ret:.2%}"
                market_lines.append(line)

        if prev_sentiment is not None:
            market_lines.append(f"  Previous day's LLM sentiment assessment: {prev_sentiment:.1f}/10")

        if market_lines:
            market_text = "Recent market history (last 5 trading days before today):\n" + "\n".join(market_lines)
        else:
            market_text = "No previous market data available."

        prompt = prompt_template.replace("{{ARTICLES}}", articles_text).replace("{{MARKET_DATA}}", market_text)

        result = _call_llm(settings, prompt)
        if result:
            result["date"] = d
            result["article_count"] = len(day_articles)
            assessments.append(result)
            prev_sentiment = result.get("daily_fear", 5)
        else:
            avg_fear = float(day_articles["recession_fear"].mean())
            assessments.append({
                "date": d,
                "daily_fear": avg_fear,
                "direction": "neutral",
                "confidence": 0.3,
                "key_driver": "LLM assessment unavailable, using average",
                "reasoning": "",
                "article_count": len(day_articles),
            })
            prev_sentiment = avg_fear

    logger.info(f"Daily assessment: {len(assessments)} days assessed by LLM")
    return assessments


def _load_prompt(settings: Settings) -> str:
    path = Path(settings.paths.prompts) / "daily_assessment_v1.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "Assess these articles:\n{{ARTICLES}}\nMarket: {{MARKET_DATA}}\nReturn JSON with daily_fear, direction, confidence, key_driver, reasoning."


def _call_llm(settings: Settings, prompt: str) -> dict | None:
    backend = settings.llm.backend

    if backend == "openrouter":
        api_key = settings.secrets.openrouter_api_key
        if not api_key:
            return None
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": settings.llm.model,
                    "messages": [
                        {"role": "system", "content": "Return ONLY valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 300,
                },
                timeout=60,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            return _parse_response(raw)
        except Exception as e:
            logger.error(f"Daily assessment LLM call failed: {e}")
            return None

    return None


def _parse_response(raw: str) -> dict | None:
    try:
        match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return {
                "daily_fear": float(data.get("daily_fear", 5)),
                "direction": data.get("direction", "neutral"),
                "confidence": float(data.get("confidence", 0.5)),
                "key_driver": data.get("key_driver", ""),
                "reasoning": data.get("reasoning", ""),
            }
    except (json.JSONDecodeError, ValueError):
        pass
    return None
