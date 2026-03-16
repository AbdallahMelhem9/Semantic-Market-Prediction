# Semantic Market Prediction

LLM-Powered Sentiment Alpha Discovery Platform — analyzes US and European financial news to quantify recession fears, correlate sentiment with market movements, and predict next-day market direction.

## Quick Start

```bash
# 1. Clone and setup
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Configure API keys in .env
cp .env.example .env
# Edit .env with your keys (NEWSAPI_KEY required, OPENROUTER_API_KEY for LLM scoring)

# 3. Run
python run.py
# Open http://127.0.0.1:8050
```

## Architecture

```
NewsAPI / Google News RSS
        ↓
  Article Collection (170 per region, cached 24h)
        ↓
  Individual Article Scoring (LLM: recession_fear 0-10, sentiment, sectors)
        ↓
  Daily Assessment (LLM sees ALL articles + yesterday's market → smart daily score)
        ↓
  Timeseries Construction (rolling averages, momentum, volatility)
        ↓
  Market Data Merge (S&P 500, Euro Stoxx 50, sector ETFs via yfinance)
        ↓
  Correlation Analysis (Pearson/Spearman at lag T+0 to T+5)
        ↓
  Prediction (XGBoost trained on 2yr VIX data + LLM forecast)
        ↓
  Dash Dashboard (interactive, multi-region, sector drill-down)
```

## Pipeline Steps

1. **Fetch News** — NewsAPI (headlines) + Google News RSS (historical) + NewsData.io (optional)
2. **Score Articles** — Each article scored by LLM with recession_fear (0-10), market_sentiment, confidence, sectors
3. **Fetch Market Data** — S&P 500 / Euro Stoxx 50 + sector ETFs via yfinance
4. **Daily Assessment** — LLM receives ALL articles per day + prior market context → produces one daily score weighing article importance
5. **Aggregate Timeseries** — Rolling averages, momentum, volatility
6. **Merge with Market** — Align sentiment with trading days
7. **Correlation** — Lag analysis to assess predictive power
8. **XGBoost Prediction** — Trained on 2 years of VIX/S&P historical data
9. **LLM Forecast** — LLM analyzes last 7 days and predicts direction

## Dashboard Features

- **Region Toggle** — US / Europe with region-specific prompts and market indices
- **Fear Gauge** — At-a-glance market fear level
- **Dual Forecast** — XGBoost model prediction + LLM analysis side by side
- **Sentiment vs Market Chart** — Dual-axis: recession fear vs S&P 500 / sector ETFs
- **Lag Correlation Heatmap** — Does sentiment predict the market at T+1, T+3, T+5?
- **Sector Heatmap** — Fear levels across sectors over time
- **Sector vs ETF** — Compare sector sentiment against actual ETF performance
- **Stock Mentions** — Individual companies extracted from headlines with fear scores
- **Article Browser** — Hover to expand, sorted by fear score
- **Chatbot** — LLM-powered Q&A with full data context
- **Model Selector** — Switch between DeepSeek, Llama, GPT-4o, Claude, Ollama

## LLM Backends Supported

| Backend | Config value | Cost |
|---------|-------------|------|
| Llama 3.3 70B (OpenRouter) | `meta-llama/llama-3.3-70b-instruct` | Free |
| DeepSeek R1 (OpenRouter) | `deepseek/deepseek-r1` | Free |
| GPT-4o (OpenRouter) | `openai/gpt-4o` | ~$0.35/run |
| Claude Sonnet (OpenRouter) | `anthropic/claude-sonnet-4.6` | ~$0.50/run |
| Ollama (local) | `ollama` | Free |

## Project Structure

```
src/
├── config/          # Settings, logging
├── ingestion/       # NewsAPI, Google News, NewsData.io, caching
├── analysis/        # LLM scoring, daily assessment, batch processing
├── timeseries/      # Aggregation, market data, correlation
├── prediction/      # XGBoost, feature engineering, LLM forecast
├── visualization/   # Plotly charts (sentiment, sectors, stocks, gauge)
├── chatbot/         # LLM-powered Q&A
├── dashboard/       # Dash app, layout, callbacks
└── pipeline.py      # Orchestrator
```

## Configuration

All settings in `config.yaml`. API keys in `.env` (gitignored).

## Tests

```bash
python -m pytest tests/ -v
```
