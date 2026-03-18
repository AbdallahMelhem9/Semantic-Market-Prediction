import ast
import json
import logging
import re
import threading
from pathlib import Path

import pandas as pd
import requests

from src.config.settings import Settings

logger = logging.getLogger(__name__)

METHODOLOGY_ANSWERS = {
    "scoring": (
        "**How article scoring works:**\n\n"
        "Each article is sent to GPT-5.4 with a sector-specific prompt. The LLM returns:\n"
        "- **recession_fear** (0-10): How much recession risk this article signals\n"
        "- **market_sentiment**: bearish / neutral / bullish\n"
        "- **confidence**: low / medium / high\n"
        "- **sectors**: Which GICS sectors are impacted\n\n"
        "Articles are scored in parallel (10 concurrent workers) and cached per model."
    ),
    "daily": (
        "**How daily assessment works:**\n\n"
        "For each day, ALL articles + the last 5 days of S&P 500 data + the previous day's sentiment "
        "are sent to the LLM. It produces ONE daily fear score (0-10) that weighs article importance — "
        "a single Fed emergency rate cut article matters more than 10 routine earnings reports.\n\n"
        "This runs twice (GPT-5.4 + Claude 4.6) and the scores are averaged (ensemble). "
        "Financials and Energy also get dedicated sector assessments via GPT-5.4."
    ),
    "prediction": (
        "**How predictions work:**\n\n"
        "Two independent methods:\n"
        "- **XGBoost**: Trained on 2 years of VIX + S&P 500 data (~500 trading days). "
        "Features: daily fear, 3/7-day rolling averages, volatility, article count, momentum.\n"
        "- **LLM Forecast**: Claude sees the last 7 days of fear scores + S&P prices and reasons about direction.\n\n"
        "Both output next-day direction (bullish/bearish) with confidence."
    ),
}


class ChatEngine:

    def __init__(self, settings: Settings, pipeline_data: dict) -> None:
        self.settings = settings
        self.all_pipeline_data = pipeline_data
        self.pipeline_data = {}
        self.prompts = self._load_prompts()
        self.frontend_context = {"region": "United States", "region_key": "us", "sector": "All sectors", "time_window": "last week"}
        self._stream_buffer = ""
        self._stream_done = False

    def _load_prompts(self) -> dict:
        prompts_dir = Path(self.settings.paths.prompts)
        result = {}
        for name in ["resolver", "factual", "sector", "prediction", "correlation",
                      "comparison", "summary", "other"]:
            path = prompts_dir / f"chat_{name}_v1.txt"
            if path.exists():
                result[name] = path.read_text(encoding="utf-8")
            else:
                result[name] = f"Answer the user's question.\n\n{{{{DATA}}}}\n\n{{{{QUERY}}}}"
        return result

    def _get_pipeline_data(self, region_override: str = None) -> dict:
        region_key = region_override or self.frontend_context.get("region_key", "us")
        return self.all_pipeline_data.get(region_key, {})

    def _get_comparison_data(self) -> dict:
        merged = {"scored_df": pd.DataFrame(), "merged_df": pd.DataFrame(),
                  "corr_df": pd.DataFrame(), "prediction": {}, "llm_forecast": {},
                  "sector_daily": {}}
        all_scored, all_merged = [], []
        for key, data in self.all_pipeline_data.items():
            s = data.get("scored_df", pd.DataFrame())
            m = data.get("merged_df", pd.DataFrame())
            if not s.empty:
                sc = s.copy(); sc["_region"] = data.get("region_name", key); all_scored.append(sc)
            if not m.empty:
                mc = m.copy(); mc["_region"] = data.get("region_name", key); all_merged.append(mc)
            for sec, vals in data.get("sector_daily", {}).items():
                merged["sector_daily"].setdefault(sec, []).extend(vals)
        if all_scored: merged["scored_df"] = pd.concat(all_scored, ignore_index=True)
        if all_merged: merged["merged_df"] = pd.concat(all_merged, ignore_index=True)
        for key, data in self.all_pipeline_data.items():
            if not merged.get("prediction"): merged["prediction"] = data.get("prediction", {})
            if not merged.get("llm_forecast"): merged["llm_forecast"] = data.get("llm_forecast", {})
            if merged.get("corr_df", pd.DataFrame()).empty: merged["corr_df"] = data.get("corr_df", pd.DataFrame())
        return merged

    def ask(self, question: str, chat_history: list = None) -> str:
        if chat_history is None: chat_history = []
        resolved_query, task_type, region_override, sector_override = self._resolve_and_classify(question, chat_history)
        logger.info(f"Chat: type={task_type}, region={region_override}, sector={sector_override}, resolved='{resolved_query[:80]}'")
        if task_type == "methodology": return self._handle_methodology(resolved_query)
        if task_type == "comparison" and region_override is None:
            self.pipeline_data = self._get_comparison_data()
        else:
            self.pipeline_data = self._get_pipeline_data(region_override)
        data_context = self._slice_data(task_type, resolved_query)
        history_text = self._format_history(chat_history[-6:])
        fc = self.frontend_context
        data_region = region_override or fc.get("region_key", "us")
        data_region_label = {"us": "United States", "europe": "Europe"}.get(data_region, data_region)
        data_sector = sector_override or fc.get("sector", "All sectors")
        dashboard_note = f"Data provided below is for: {data_region_label} / {data_sector} / {fc.get('time_window', 'last week')}."
        prompt_template = self.prompts.get(task_type, self.prompts["other"])
        prompt = (prompt_template.replace("{{DATA}}", f"{dashboard_note}\n\n{data_context}")
                  .replace("{{QUERY}}", resolved_query).replace("{{HISTORY}}", history_text))
        response = self._call_llm(prompt)
        return response or "I couldn't generate a response. Please try again."

    def ask_streaming(self, question: str, chat_history: list = None):
        self._stream_buffer = ""
        self._stream_done = False
        def _run():
            hist = chat_history if chat_history else []
            resolved_query, task_type, region_override, sector_override = self._resolve_and_classify(question, hist)
            logger.info(f"Chat: type={task_type}, region={region_override}, sector={sector_override}, resolved='{resolved_query[:80]}'")
            if task_type == "methodology":
                self._stream_buffer = self._handle_methodology(resolved_query)
                self._stream_done = True
                return
            if task_type == "comparison" and region_override is None:
                self.pipeline_data = self._get_comparison_data()
            else:
                self.pipeline_data = self._get_pipeline_data(region_override)
            data_context = self._slice_data(task_type, resolved_query)
            history_text = self._format_history(hist[-6:])
            fc = self.frontend_context
            data_region = region_override or fc.get("region_key", "us")
            data_region_label = {"us": "United States", "europe": "Europe"}.get(data_region, data_region)
            data_sector = sector_override or fc.get("sector", "All sectors")
            dashboard_note = f"Data provided below is for: {data_region_label} / {data_sector} / {fc.get('time_window', 'last week')}."
            prompt_template = self.prompts.get(task_type, self.prompts["other"])
            prompt = (prompt_template.replace("{{DATA}}", f"{dashboard_note}\n\n{data_context}")
                      .replace("{{QUERY}}", resolved_query).replace("{{HISTORY}}", history_text))
            self._call_llm_streaming(prompt)
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def get_stream_chunk(self) -> tuple[str, bool]:
        return self._stream_buffer, self._stream_done

    def _resolve_and_classify(self, message: str, chat_history: list) -> tuple[str, str, str | None, str | None]:
        history_text = self._format_history(chat_history[-6:])
        fc = self.frontend_context
        prompt = (self.prompts["resolver"]
                  .replace("{{MESSAGE}}", message)
                  .replace("{{HISTORY}}", history_text if history_text else "No prior conversation.")
                  .replace("{{REGION}}", fc.get("region", "United States"))
                  .replace("{{SECTOR}}", fc.get("sector", "All sectors"))
                  .replace("{{TIME_WINDOW}}", fc.get("time_window", "last week")))
        raw = self._call_llm_fast(prompt)
        if raw:
            try:
                match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    resolved = data.get("resolved_query", message)
                    task_type = data.get("task_type", "other")
                    region_override = data.get("region")
                    sector_override = data.get("sector")
                    if region_override in ("null", "", "none", "None"): region_override = None
                    if sector_override in ("null", "", "none", "None"): sector_override = None
                    valid_types = ["factual", "sector", "prediction", "correlation", "comparison", "summary", "methodology", "other"]
                    if task_type not in valid_types: task_type = "other"
                    return resolved, task_type, region_override, sector_override
            except (json.JSONDecodeError, ValueError):
                pass
        return message, "other", None, None

    # ── Data Slicing ──

    def _slice_data(self, task_type: str, query: str) -> str:
        merged_df = self.pipeline_data.get("merged_df", pd.DataFrame())
        scored_df = self.pipeline_data.get("scored_df", pd.DataFrame())
        prediction = self.pipeline_data.get("prediction", {})
        llm_forecast = self.pipeline_data.get("llm_forecast", {})
        corr_df = self.pipeline_data.get("corr_df", pd.DataFrame())
        sector_daily = self.pipeline_data.get("sector_daily", {})
        if task_type == "factual": return self._data_articles(scored_df, merged_df)
        elif task_type == "sector": return self._data_sectors(scored_df, sector_daily, merged_df)
        elif task_type == "prediction": return self._data_prediction(merged_df, prediction, llm_forecast)
        elif task_type == "correlation": return self._data_correlation(corr_df, merged_df)
        elif task_type == "comparison": return self._data_articles(scored_df, merged_df)
        elif task_type == "summary": return self._data_summary(scored_df, merged_df, prediction, llm_forecast, sector_daily)
        else: return self._data_summary(scored_df, merged_df, prediction, llm_forecast, sector_daily)

    def _data_articles(self, scored_df, merged_df):
        parts = []
        if not merged_df.empty:
            parts.append("**Daily sentiment timeseries:**")
            for _, row in merged_df.iterrows():
                d = row.get("date", ""); fear = row.get("avg_recession_fear", "N/A")
                sp = row.get("sp500_close", "")
                line = f"  {d}: fear={fear:.1f}" if isinstance(fear, (int, float)) else f"  {d}: fear={fear}"
                if sp and not pd.isna(sp): line += f", S&P={sp:.0f}"
                parts.append(line)
        if not scored_df.empty:
            parts.append("\n**Articles by date:**")
            sc = scored_df.copy(); sc["date"] = pd.to_datetime(sc["date"]).dt.date
            for d, group in sc.groupby("date"):
                parts.append(f"\n  {d} ({len(group)} articles):")
                for _, row in group.head(8).iterrows():
                    parts.append(f"    - \"{str(row.get('title', ''))[:100]}\" (fear={row.get('recession_fear', '?')}, {row.get('market_sentiment', '?')}, sectors={row.get('sectors', [])})")
                if len(group) > 8: parts.append(f"    ... and {len(group) - 8} more")
        return "\n".join(parts) if parts else "No data available."

    def _data_sectors(self, scored_df, sector_daily, merged_df):
        parts = []
        if not merged_df.empty:
            latest = merged_df.iloc[-1]
            parts.append(f"**Overall market:** Latest fear={latest.get('avg_recession_fear', 'N/A')}")
        if not scored_df.empty and "sectors" in scored_df.columns:
            df = scored_df.copy(); df["sectors"] = df["sectors"].apply(self._parse_sectors)
            exploded = df.explode("sectors").dropna(subset=["sectors"])
            if not exploded.empty:
                sector_avg = exploded.groupby("sectors")["recession_fear"].agg(["mean", "count"]).sort_values("mean", ascending=False)
                parts.append("\n**Sector fear averages (from article scores):**")
                for sec, row in sector_avg.iterrows():
                    parts.append(f"  {sec}: fear={row['mean']:.1f} ({int(row['count'])} articles)")
        for sec, assessments in sector_daily.items():
            parts.append(f"\n**{sec} — LLM daily assessment (GPT-5.4):**")
            for a in sorted(assessments, key=lambda x: str(x.get("date", ""))):
                parts.append(f"  {str(a.get('date', ''))[:10]}: fear={a.get('daily_fear', '?')}, driver: {a.get('key_driver', '')[:100]}")
        return "\n".join(parts) if parts else "No sector data available."

    def _data_prediction(self, merged_df, prediction, llm_forecast):
        parts = []
        if prediction: parts.append(f"**XGBoost prediction:** {prediction.get('direction', 'N/A')} (confidence: {prediction.get('confidence', 0):.0%}, accuracy: {prediction.get('accuracy', 0):.0%})")
        if llm_forecast:
            parts.append(f"**LLM forecast:** {llm_forecast.get('direction', 'N/A')} (confidence: {llm_forecast.get('confidence', 0):.0%})")
            r = llm_forecast.get("reasoning", "")
            if r: parts.append(f"  Reasoning: {r}")
        if not merged_df.empty:
            parts.append("\n**Recent sentiment (last 7 days):**")
            for _, row in merged_df.tail(7).iterrows():
                d = row.get("date", ""); fear = row.get("avg_recession_fear", "N/A"); sp = row.get("sp500_close", "")
                line = f"  {d}: fear={fear:.1f}" if isinstance(fear, (int, float)) else f"  {d}: fear={fear}"
                if sp and not pd.isna(sp): line += f", S&P={sp:.0f}"
                parts.append(line)
        return "\n".join(parts) if parts else "No prediction data available."

    def _data_correlation(self, corr_df, merged_df):
        parts = []
        if not corr_df.empty: parts.append("**Lag correlation (sentiment vs S&P 500 returns):**\n" + corr_df.to_string())
        if not merged_df.empty: parts.append(f"\n**Data points:** {len(merged_df)} days")
        return "\n".join(parts) if parts else "No correlation data available."

    def _data_summary(self, scored_df, merged_df, prediction, llm_forecast, sector_daily):
        parts = []
        if not merged_df.empty:
            latest = merged_df.iloc[-1]
            parts.append(f"**Latest:** {latest.get('date')}, fear={latest.get('avg_recession_fear', 'N/A')}, S&P={latest.get('sp500_close', 'N/A')}")
            parts.append(f"**Period:** {len(merged_df)} days, avg fear={merged_df['avg_recession_fear'].mean():.1f}")
            parts.append("\n**Daily trend:**")
            for _, row in merged_df.tail(7).iterrows():
                fear = row.get("avg_recession_fear", "N/A")
                parts.append(f"  {row.get('date', '')}: fear={fear:.1f}" if isinstance(fear, (int, float)) else f"  {row.get('date', '')}: fear={fear}")
        if prediction: parts.append(f"\n**XGBoost:** {prediction.get('direction')} ({prediction.get('confidence', 0):.0%})")
        if llm_forecast: parts.append(f"**LLM forecast:** {llm_forecast.get('direction')} ({llm_forecast.get('confidence', 0):.0%})")
        if not scored_df.empty:
            top = scored_df.nlargest(5, "recession_fear")
            parts.append("\n**Top 5 highest-fear articles:**")
            for _, row in top.iterrows():
                parts.append(f"  - \"{str(row.get('title', ''))[:80]}\" (fear={row.get('recession_fear')}, {row.get('date')})")
        for sec, assessments in sector_daily.items():
            if assessments:
                la = max(assessments, key=lambda x: str(x.get("date", "")))
                parts.append(f"\n**{sec} (latest):** fear={la.get('daily_fear')}, driver: {la.get('key_driver', '')[:80]}")
        return "\n".join(parts) if parts else "No data available."

    def _handle_methodology(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["score", "scoring", "article", "grade"]): return METHODOLOGY_ANSWERS["scoring"]
        if any(w in q for w in ["daily", "ensemble", "assessment", "day"]): return METHODOLOGY_ANSWERS["daily"]
        if any(w in q for w in ["predict", "xgboost", "forecast", "model"]): return METHODOLOGY_ANSWERS["prediction"]
        return "\n\n---\n\n".join(METHODOLOGY_ANSWERS.values())

    # ── LLM Calls ──

    def _call_llm_fast(self, prompt: str) -> str | None:
        api_key = self.settings.secrets.openrouter_api_key
        if not api_key: return None
        try:
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "openai/gpt-5.4", "messages": [{"role": "system", "content": "Return ONLY valid JSON."}, {"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 150}, timeout=15)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"Chat classifier failed: {e}"); return None

    def _call_llm(self, prompt: str) -> str | None:
        api_key = self.settings.secrets.openrouter_api_key
        if not api_key: return None
        try:
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "anthropic/claude-sonnet-4.6", "messages": [{"role": "user", "content": prompt}], "max_tokens": 800, "temperature": 0.3}, timeout=30)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"Chat response failed: {e}"); return None

    def _call_llm_streaming(self, prompt: str):
        api_key = self.settings.secrets.openrouter_api_key
        if not api_key:
            self._stream_buffer = "API key not configured."; self._stream_done = True; return
        try:
            resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "anthropic/claude-sonnet-4.6", "messages": [{"role": "user", "content": prompt}], "max_tokens": 800, "temperature": 0.3, "stream": True}, timeout=60, stream=True)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line: continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]": break
                    try:
                        chunk = json.loads(data_str)
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content: self._stream_buffer += content
                    except json.JSONDecodeError: pass
        except Exception as e:
            logger.warning(f"Chat streaming failed: {e}")
            if not self._stream_buffer: self._stream_buffer = "Streaming failed. Please try again."
        self._stream_done = True

    def _format_history(self, chat_history: list) -> str:
        if not chat_history: return ""
        return "\n".join(f"{msg.get('role', 'user')}: {msg.get('text', '')[:200]}" for msg in chat_history)

    @staticmethod
    def _parse_sectors(val):
        if isinstance(val, list): return val
        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                return parsed if isinstance(parsed, list) else [val]
            except (ValueError, SyntaxError):
                return [s.strip() for s in val.split(",") if s.strip()]
        return []
