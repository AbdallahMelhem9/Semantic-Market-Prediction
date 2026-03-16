import logging
from pathlib import Path

import pandas as pd

from src.config.settings import Settings

logger = logging.getLogger(__name__)


class ChatEngine:

    def __init__(self, settings: Settings, pipeline_data: dict) -> None:
        self.settings = settings
        self.pipeline_data = pipeline_data
        self.prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        prompt_path = Path(self.settings.paths.prompts) / "chatbot_system_v1.txt"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return "You are a financial sentiment assistant. Context: {{CONTEXT}}\n\nQuestion: {{QUESTION}}"

    def _build_context(self) -> str:
        merged_df = self.pipeline_data.get("merged_df", pd.DataFrame())
        scored_df = self.pipeline_data.get("scored_df", pd.DataFrame())
        prediction = self.pipeline_data.get("prediction", {})

        parts = []

        if not merged_df.empty:
            latest = merged_df.iloc[-1]
            parts.append(f"Latest date: {latest.get('date', 'N/A')}")
            parts.append(f"Latest recession fear: {latest.get('avg_recession_fear', 'N/A')}")
            parts.append(f"Latest S&P 500: {latest.get('sp500_close', 'N/A')}")

            parts.append(f"\nDaily summary (last {len(merged_df)} days):")
            for _, row in merged_df.iterrows():
                parts.append(f"  {row.get('date')}: fear={row.get('avg_recession_fear', 'N/A')}, "
                             f"sentiment={row.get('avg_sentiment', 'N/A')}, "
                             f"articles={row.get('article_count', 'N/A')}")

        # Include actual article headlines grouped by date so LLM can reference real events
        if not scored_df.empty:
            parts.append("\nArticle headlines by date (with individual scores):")
            scored_copy = scored_df.copy()
            scored_copy["date"] = pd.to_datetime(scored_copy["date"]).dt.date
            for d, group in scored_copy.groupby("date"):
                parts.append(f"\n  {d}:")
                for _, row in group.iterrows():
                    title = str(row.get("title", ""))[:100]
                    fear = row.get("recession_fear", "N/A")
                    sentiment = row.get("market_sentiment", "N/A")
                    sectors = row.get("sectors", [])
                    parts.append(f"    - \"{title}\" (fear={fear}, {sentiment}, sectors={sectors})")

        if prediction:
            parts.append(f"\nPrediction: {prediction.get('direction', 'N/A')} "
                         f"(confidence: {prediction.get('confidence', 0):.0%})")
            parts.append(f"Model accuracy: {prediction.get('accuracy', 0):.0%}")

        return "\n".join(parts) if parts else "No data loaded yet."

    def ask(self, question: str) -> str:
        context = self._build_context()
        prompt = self.prompt_template.replace("{{CONTEXT}}", context).replace("{{QUESTION}}", question)

        # Try LLM call
        response = self._call_llm(prompt)
        if response:
            return response

        # Fallback to simple response
        return self._simple_response(question)

    def _call_llm(self, prompt: str) -> str | None:
        """Always uses Claude 4.6 for chat, regardless of selected scoring model."""
        import requests

        api_key = self.settings.secrets.openrouter_api_key
        if not api_key:
            return None

        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "anthropic/claude-sonnet-4.6",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 600,
                    "temperature": 0.3,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"Chat (Claude 4.6) failed: {e}")
            return None

    def _simple_response(self, question: str) -> str:
        merged_df = self.pipeline_data.get("merged_df", pd.DataFrame())
        if merged_df.empty:
            return "No data loaded yet. Run the pipeline first."

        q = question.lower()
        latest = merged_df.iloc[-1]

        if any(w in q for w in ["latest", "today", "current", "now"]):
            return (f"Latest ({latest['date']}): Recession fear = {latest.get('avg_recession_fear', 'N/A'):.1f}, "
                    f"S&P 500 = {latest.get('sp500_close', 'N/A'):.0f}")

        if any(w in q for w in ["trend", "direction", "going"]):
            if "rolling_7d" in merged_df.columns and len(merged_df) >= 2:
                recent = merged_df["rolling_7d"].iloc[-1]
                prev = merged_df["rolling_7d"].iloc[-2]
                trend = "rising" if recent > prev else "falling"
                return f"7-day sentiment trend is {trend} ({prev:.1f} → {recent:.1f})."

        if "predict" in q or "tomorrow" in q or "forecast" in q:
            pred = self.pipeline_data.get("prediction", {})
            if pred:
                return (f"Model predicts: {pred.get('direction', 'N/A')} "
                        f"(confidence: {pred.get('confidence', 0):.0%}, "
                        f"accuracy: {pred.get('accuracy', 0):.0%})")

        avg = merged_df["avg_recession_fear"].mean()
        return (f"Based on {len(merged_df)} days of data, average recession fear is {avg:.1f}/10. "
                f"Ask about trends, sectors, latest values, or predictions.")
